"""Single-sentence and file-based inference entry point."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from capstone.config import load_config
from capstone.models.base import Seq2SeqGenerator, resolve_checkpoint
from capstone.models.gas import parse_gas_generation
from capstone.models.paraphrase import parse_paraphrase_generation


def _load_optional_config(config_path: str | Path | None, checkpoint: Path) -> dict[str, Any] | None:
    candidates = [Path(config_path).expanduser()] if config_path else [
        checkpoint / "config.yaml",
        checkpoint.parent / "config.yaml",
        checkpoint.parent.parent / "config.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return load_config(candidate)
    return None


def _input_rows(text: str | None, input_file: str | Path | None) -> list[dict[str, Any]]:
    if text is not None:
        return [{"text": text}]
    if input_file is None:
        raise ValueError("Provide either --text or --input-file")
    rows: list[dict[str, Any]] = []
    path = Path(input_file).expanduser()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.rstrip("\r\n")
            if not stripped:
                continue
            if path.suffix.casefold() == ".jsonl":
                try:
                    value = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL input at {path}:{line_number}: {exc}") from exc
                if not isinstance(value, dict) or not isinstance(value.get("text"), str):
                    raise ValueError(f"Each JSONL input row needs a string 'text' at {path}:{line_number}")
                rows.append(value)
            else:
                rows.append({"text": stripped})
    return rows


def _validated_inventory(metadata: Any, source: Path) -> set[str]:
    inventory = metadata.get("aspect_inventory") if isinstance(metadata, dict) else None
    if (
        not isinstance(inventory, list)
        or not inventory
        or any(not isinstance(item, str) or not item.strip() for item in inventory)
    ):
        raise ValueError(f"Metadata at {source} is missing a valid non-empty aspect_inventory")
    return set(inventory)


def _known_aspects(config: dict[str, Any] | None, checkpoint: Path) -> set[str]:
    experiment_root = checkpoint.parent.parent if checkpoint.parent.name == "checkpoints" else checkpoint.parent
    metadata_path = experiment_root / "metadata.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not read experiment metadata at {metadata_path}: {error}") from error
        return _validated_inventory(metadata, metadata_path)
    if config is None:
        raise ValueError(
            "Cannot constrain inference to the domain aspect inventory: checkpoint metadata "
            "is absent and no configured prepared-data metadata is available"
        )
    domain = config.get("experiment", {}).get("domain")
    processed = config.get("paths", {}).get("processed_data")
    if not domain or processed is None:
        raise ValueError(
            "Inference requires experiment.domain and paths.processed_data when checkpoint "
            "metadata does not contain an aspect inventory"
        )
    prepared_metadata_path = Path(processed) / str(domain) / "metadata.json"
    if not prepared_metadata_path.is_file():
        raise ValueError(f"Prepared-data metadata is required for inference: {prepared_metadata_path}")
    try:
        prepared_metadata = json.loads(prepared_metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Could not read prepared-data metadata at {prepared_metadata_path}: {error}"
        ) from error
    return _validated_inventory(prepared_metadata, prepared_metadata_path)


@dataclass
class LoadedPredictor:
    """A checkpoint kept in memory with its decoding settings and aspect inventory.

    The CLI loads one per invocation; a long-running service loads one per
    (domain, model) and reuses it across requests.
    """

    model_type: str
    checkpoint_path: Path
    model: Any
    parser: Any
    known_aspects: set[str]
    max_input_length: int
    num_beams: int
    max_new_tokens: int
    early_stopping: Any
    batch_size: int

    def predict(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        for start in range(0, len(rows), self.batch_size):
            batch = rows[start:start + self.batch_size]
            generations = self.model.generate(
                [str(row["text"]) for row in batch],
                max_input_length=self.max_input_length,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
                early_stopping=self.early_stopping,
            )
            for row, raw_generation in zip(batch, generations):
                parsed = self.parser(raw_generation, known_aspects=self.known_aspects)
                output = {
                    "text": row["text"],
                    "quadruples": [quad.to_dict() if hasattr(quad, "to_dict") else {
                        "target": quad.target,
                        "aspect": quad.aspect,
                        "opinion": quad.opinion,
                        "sentiment": quad.sentiment,
                    } for quad in parsed.quadruples],
                    "raw_generation": raw_generation,
                    "invalid_predictions": parsed.invalid_predictions,
                }
                for identifier in ("review_id", "sentence_id", "id"):
                    if identifier in row:
                        output[identifier] = row[identifier]
                outputs.append(output)
        return outputs


def load_predictor(
    checkpoint: str | Path,
    model_type: str,
    config_path: str | Path | None = None,
) -> LoadedPredictor:
    """Resolve a checkpoint, validate it against its config, and load the model once."""
    if model_type not in {"gas", "paraphrase"}:
        raise ValueError("model_type must be gas or paraphrase")
    checkpoint_path = resolve_checkpoint(checkpoint)
    config = _load_optional_config(config_path, checkpoint_path)
    if config is not None and config.get("model", {}).get("type") != model_type:
        raise ValueError(
            f"--model-type {model_type!r} does not match the checkpoint config model type "
            f"{config.get('model', {}).get('type')!r}"
        )
    generation = config.get("generation", {}) if config else {}
    sequence = config.get("sequence", {}) if config else {}
    evaluation = config.get("evaluation", {}) if config else {}
    inference = config.get("inference", {}) if config else {}
    batch_size = int(inference.get("batch_size", evaluation.get("batch_size", 16)))
    if batch_size < 1:
        raise ValueError("Inference batch size must be positive")
    known_aspects = _known_aspects(config, checkpoint_path)
    model = Seq2SeqGenerator.from_checkpoint(checkpoint_path)
    return LoadedPredictor(
        model_type=model_type,
        checkpoint_path=checkpoint_path,
        model=model,
        parser=parse_gas_generation if model_type == "gas" else parse_paraphrase_generation,
        known_aspects=known_aspects,
        max_input_length=int(sequence.get("max_input_length", 256)),
        num_beams=int(generation.get("num_beams", 1)),
        max_new_tokens=int(generation.get("max_new_tokens", 256)),
        early_stopping=generation.get("early_stopping", False),
        batch_size=batch_size,
    )


def predict_rows(
    checkpoint: str | Path,
    model_type: str,
    rows: list[dict[str, Any]],
    config_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    return load_predictor(checkpoint, model_type, config_path).predict(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict OATS ASQP quadruples from text")
    parser.add_argument("--checkpoint", required=True, help="checkpoint directory or experiment directory with best_checkpoint.json")
    parser.add_argument("--model-type", required=True, choices=("gas", "paraphrase"))
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text")
    input_group.add_argument("--input-file")
    parser.add_argument("--config", help="Optional YAML config for generation settings")
    parser.add_argument("--output", help="Optional JSON/JSONL output path")
    args = parser.parse_args(argv)
    rows = _input_rows(args.text, args.input_file)
    outputs = predict_rows(args.checkpoint, args.model_type, rows, args.config)
    if args.text is not None:
        rendered = json.dumps(outputs[0], ensure_ascii=False, indent=2)
    else:
        rendered = "\n".join(json.dumps(output, ensure_ascii=False) for output in outputs)
        if rendered:
            rendered += "\n"
    if args.output:
        destination = Path(args.output).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + ("\n" if args.text is not None else ""), encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
