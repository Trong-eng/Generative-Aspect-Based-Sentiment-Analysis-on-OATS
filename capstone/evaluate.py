"""`python -m capstone.evaluate` evaluation entry point."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from capstone.config import load_config
from capstone.data.serialization import validate_prepared_metadata
from capstone.logging_utils import configure_logging
from capstone.models.base import Seq2SeqGenerator, resolve_checkpoint
from capstone.models.gas import parse_gas_generation
from capstone.models.paraphrase import parse_paraphrase_generation
from capstone.types import Quadruple
from capstone.evaluation import evaluate_records


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            records.append(record)
    return records


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _experiment_root(checkpoint_arg: str | Path, selected_checkpoint: Path) -> Path:
    starts = [Path(checkpoint_arg).expanduser().resolve(), selected_checkpoint.resolve()]
    seen: set[Path] = set()
    for start in starts:
        if start.is_file():
            start = start.parent
        for candidate in (start, *start.parents):
            if candidate in seen:
                continue
            seen.add(candidate)
            if (candidate / "config.yaml").is_file() and (candidate / "metadata.json").is_file():
                try:
                    selected_checkpoint.resolve().relative_to(candidate.resolve())
                except ValueError:
                    raise ValueError(
                        f"Selected checkpoint {selected_checkpoint} is outside its experiment directory {candidate}"
                    )
                return candidate.resolve()
    raise ValueError(
        "Could not bind checkpoint to an experiment directory containing config.yaml and metadata.json"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validated_fingerprint_map(value: Any, *, name: str, required_keys: Sequence[str] = ()) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Saved provenance is missing {name} mapping")
    normalized = {str(key): item for key, item in value.items()}
    missing = [key for key in required_keys if key not in normalized]
    if missing:
        raise ValueError(f"Saved provenance {name} is missing split(s): {', '.join(missing)}")
    for key, fingerprint in normalized.items():
        if not isinstance(fingerprint, str) or re.fullmatch(r"[0-9a-fA-F]{64}", fingerprint) is None:
            raise ValueError(f"Saved provenance {name}[{key!r}] is not a SHA-256 fingerprint")
        normalized[key] = fingerprint.lower()
    return normalized


def _validate_config_provenance(
    requested: dict[str, Any],
    saved: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    requested_values = {
        "model.type": requested["model"]["type"],
        "model.model_name": requested["model"]["model_name"],
        "model.tokenizer_name": requested["model"]["tokenizer_name"],
        "experiment.domain": requested["experiment"]["domain"],
        "experiment.split_mode": requested["experiment"]["split_mode"],
        "experiment.seed": requested["experiment"]["seed"],
        "dataset.sentiment_mode": requested["dataset"]["sentiment_mode"],
        "dataset.version": requested["dataset"]["version"],
    }
    saved_values = {
        "model.type": saved["model"]["type"],
        "model.model_name": saved["model"]["model_name"],
        "model.tokenizer_name": saved["model"]["tokenizer_name"],
        "experiment.domain": saved["experiment"]["domain"],
        "experiment.split_mode": saved["experiment"]["split_mode"],
        "experiment.seed": saved["experiment"]["seed"],
        "dataset.sentiment_mode": saved["dataset"]["sentiment_mode"],
        "dataset.version": saved["dataset"]["version"],
    }
    metadata_values = {
        "model.type": metadata.get("model_type"),
        "model.model_name": metadata.get("model_backbone"),
        "model.tokenizer_name": metadata.get("tokenizer", metadata.get("tokenizer_name")),
        "experiment.domain": metadata.get("domain"),
        "experiment.split_mode": metadata.get("split_mode"),
        "experiment.seed": metadata.get("seed"),
        "dataset.sentiment_mode": metadata.get("sentiment_mode"),
        "dataset.version": metadata.get("dataset_version"),
    }
    for field, requested_value in requested_values.items():
        saved_value = saved_values[field]
        if requested_value != saved_value:
            raise ValueError(
                f"Requested config {field}={requested_value!r} does not match checkpoint experiment config {saved_value!r}"
            )
        provenance_value = metadata_values[field]
        if provenance_value is None or provenance_value != saved_value:
            raise ValueError(
                f"Checkpoint experiment metadata {field}={provenance_value!r} does not match saved config {saved_value!r}"
            )

    for section in ("sequence", "generation"):
        if requested.get(section) != saved.get(section):
            raise ValueError(
                f"Requested config {section} settings do not match checkpoint experiment config"
            )
    requested_cleaning = requested.get("cleaning", {})
    saved_cleaning = saved.get("cleaning", {})
    if requested_cleaning != saved_cleaning:
        raise ValueError("Requested config cleaning settings do not match checkpoint experiment config")
    if metadata.get("cleaning_configuration") != saved_cleaning:
        raise ValueError("Checkpoint experiment metadata cleaning settings do not match saved config")


def _known_aspects_from_inventory(metadata: Mapping[str, Any]) -> set[str]:
    inventory = metadata.get("aspect_inventory")
    if isinstance(inventory, Mapping):
        values = inventory.keys()
    elif isinstance(inventory, Sequence) and not isinstance(inventory, (str, bytes, bytearray)):
        values = inventory
    else:
        raise ValueError("Prepared-data metadata is missing an aspect_inventory list or mapping")
    aspects: set[str] = set()
    for value in values:
        aspect = value.get("aspect") if isinstance(value, Mapping) else value
        if not isinstance(aspect, str) or not aspect.strip():
            raise ValueError("Prepared-data metadata aspect_inventory contains an invalid aspect")
        aspects.add(aspect.strip())
    if not aspects:
        raise ValueError("Prepared-data metadata aspect_inventory must be non-empty")
    return aspects


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def _quadruple_dict(value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        return {key: str(value[key]) for key in ("target", "aspect", "opinion", "sentiment")}
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return {
        "target": value.target,
        "aspect": value.aspect,
        "opinion": value.opinion,
        "sentiment": value.sentiment,
    }


def run_evaluation(config_path: str | Path, checkpoint: str | Path, split: str = "test") -> Path:
    config = load_config(config_path)
    if split not in {"train", "dev", "validation", "test"}:
        raise ValueError("split must be train, dev, validation, or test")
    data_split = "dev" if split == "validation" else split

    selected_checkpoint = resolve_checkpoint(checkpoint).expanduser().resolve()
    experiment_root = _experiment_root(checkpoint, selected_checkpoint)
    experiment_config_path = experiment_root / "config.yaml"
    experiment_config = load_config(experiment_config_path)
    experiment_metadata = _read_json(experiment_root / "metadata.json")
    _validate_config_provenance(config, experiment_config, experiment_metadata)

    domain = config["experiment"]["domain"]
    processed_root = Path(config["paths"]["processed_data"]) / domain
    prepared_metadata = validate_prepared_metadata(
        processed_root,
        domain=domain,
        split_mode=config["experiment"]["split_mode"],
        sentiment_mode=config["dataset"]["sentiment_mode"],
        dataset_version=config.get("dataset", {}).get("version", "local_oats_xml"),
        seed=config["experiment"]["seed"],
        cleaning_configuration=config.get("cleaning"),
    )
    prepared_split_fingerprints = _validated_fingerprint_map(
        prepared_metadata.get("split_fingerprints"),
        name="prepared split_fingerprints",
        required_keys=("train", "dev", "test"),
    )
    experiment_split_fingerprints = _validated_fingerprint_map(
        experiment_metadata.get("split_fingerprints"),
        name="checkpoint split_fingerprints",
        required_keys=("train", "dev", "test"),
    )
    if prepared_split_fingerprints != experiment_split_fingerprints:
        raise ValueError("Prepared split fingerprints do not match checkpoint experiment provenance")

    prepared_source_fingerprints = _validated_fingerprint_map(
        prepared_metadata.get("source_fingerprints"),
        name="prepared source_fingerprints",
    )
    experiment_source_fingerprints = _validated_fingerprint_map(
        experiment_metadata.get("source_fingerprints"),
        name="checkpoint source_fingerprints",
    )
    if not prepared_source_fingerprints:
        raise ValueError("Prepared-data metadata has no source_fingerprints")
    if prepared_source_fingerprints != experiment_source_fingerprints:
        raise ValueError("Prepared source fingerprints do not match checkpoint experiment provenance")

    prepared_aspects = _known_aspects_from_inventory(prepared_metadata)
    experiment_aspects = _known_aspects_from_inventory(experiment_metadata)
    if prepared_aspects != experiment_aspects:
        raise ValueError("Prepared aspect inventory does not match checkpoint experiment provenance")

    dataset_path = processed_root / f"{data_split}.jsonl"
    split_fingerprint = _sha256_file(dataset_path)
    expected_split_fingerprint = prepared_split_fingerprints[data_split]
    if split_fingerprint != expected_split_fingerprint:
        raise ValueError(
            f"Prepared {data_split} split fingerprint mismatch: "
            f"metadata={expected_split_fingerprint}, actual={split_fingerprint}"
        )
    records = _read_jsonl(dataset_path)
    known_aspects = experiment_aspects
    model = Seq2SeqGenerator.from_checkpoint(selected_checkpoint)
    model_type = config["model"]["type"]
    parser = parse_gas_generation if model_type == "gas" else parse_paraphrase_generation
    batch_size = int(config.get("evaluation", {}).get("batch_size", config.get("training", {}).get("batch_size", 16)))
    max_input_length = int(config["sequence"]["max_input_length"])
    generation = config["generation"]

    logger = configure_logging(
        experiment_root / "logs" / "evaluation.log",
        level=str(config.get("logging", {}).get("level", "INFO")),
    )
    logger.info("Evaluating %s checkpoint on %s/%s (%d records)", model_type, domain, data_split, len(records))
    cases: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        raw_generations = model.generate(
            [str(record.get("text", "")) for record in batch],
            max_input_length=max_input_length,
            num_beams=int(generation["num_beams"]),
            max_new_tokens=int(generation["max_new_tokens"]),
            early_stopping=generation["early_stopping"],
        )
        for record, raw_generation in zip(batch, raw_generations):
            parsed = parser(raw_generation, known_aspects=known_aspects)
            gold = [Quadruple.from_dict(q) if hasattr(Quadruple, "from_dict") else Quadruple(**q)
                    for q in record.get("quadruples", [])]
            normalized_gold = [q.to_dict() if hasattr(q, "to_dict") else _quadruple_dict(q) for q in gold]
            normalized_prediction = [_quadruple_dict(q) for q in parsed.quadruples]
            case = {
                "review_id": record.get("review_id"),
                "sentence_id": record.get("sentence_id"),
                "text": record.get("text", ""),
                "gold_quadruples": gold,
                "predicted_quadruples": parsed.quadruples,
                "raw_generation": raw_generation,
                "invalid_predictions": parsed.invalid_predictions,
            }
            cases.append(case)
            prediction_rows.append({
                "review_id": record.get("review_id"),
                "sentence_id": record.get("sentence_id"),
                "text": record.get("text", ""),
                "gold_quadruples": normalized_gold,
                "raw_generation": raw_generation,
                "parsed_quadruples": normalized_prediction,
                "normalized_quadruples": normalized_prediction,
                "invalid_predictions": parsed.invalid_predictions,
            })
            for invalid in parsed.invalid_predictions:
                invalid_rows.append({
                    "review_id": record.get("review_id"),
                    "sentence_id": record.get("sentence_id"),
                    "text": record.get("text", ""),
                    "raw_generation": raw_generation,
                    **invalid,
                })

    metrics, diagnostics, error_cases = evaluate_records(cases)
    output_dir = experiment_root / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "diagnostic_metrics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_jsonl(output_dir / "predictions.jsonl", prediction_rows)
    _write_jsonl(output_dir / "invalid_predictions.jsonl", invalid_rows)
    _write_jsonl(output_dir / "error_cases.jsonl", error_cases)
    (output_dir / "run_metadata.json").write_text(json.dumps({
        "experiment_dir": str(experiment_root),
        "experiment_config": str(experiment_config_path),
        "requested_config": str(Path(config_path).expanduser().resolve()),
        "model_type": model_type,
        "model_backbone": config["model"]["model_name"],
        "tokenizer": config["model"]["tokenizer_name"],
        "domain": domain,
        "split_mode": config["experiment"]["split_mode"],
        "dataset_version": config.get("dataset", {}).get("version", "local_oats_xml"),
        "sentiment_mode": config["dataset"]["sentiment_mode"],
        "seed": config["experiment"]["seed"],
        "split": data_split,
        "split_fingerprint": split_fingerprint,
        "split_fingerprints": experiment_split_fingerprints,
        "dataset_identity": {
            "domain": domain,
            "dataset_version": config.get("dataset", {}).get("version", "local_oats_xml"),
            "split_mode": config["experiment"]["split_mode"],
            "sentiment_mode": config["dataset"]["sentiment_mode"],
            "seed": config["experiment"]["seed"],
            "cleaning_configuration": config.get("cleaning", {}),
            "split_fingerprints": experiment_split_fingerprints,
            "source_fingerprints": experiment_source_fingerprints,
        },
        "evaluation_settings": {
            "sequence": config["sequence"],
            "generation": config["generation"],
        },
        "sample_count": len(records),
        "invalid_generation_count": len(invalid_rows),
        "aspect_inventory": sorted(experiment_aspects),
        "checkpoint": selected_checkpoint.relative_to(experiment_root).as_posix(),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Evaluation artifacts written to %s", output_dir)
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a GAS-T5 or Paraphrase-T5 OATS ASQP checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="checkpoint directory or experiment directory with best_checkpoint.json")
    parser.add_argument("--split", default="test", choices=("train", "dev", "validation", "test"))
    args = parser.parse_args(argv)
    output = run_evaluation(args.config, args.checkpoint, args.split)
    print(json.dumps({"evaluation_dir": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
