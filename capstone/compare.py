"""Compare compatible GAS and Paraphrase experiment artifacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any
from collections.abc import Mapping

from capstone.config import load_config, to_serializable_config


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _require_equal(left: Any, right: Any, name: str) -> None:
    if left != right:
        raise ValueError(f"Experiments are not comparable: {name} differs ({left!r} != {right!r})")


def _fingerprint_map(
    metadata: dict[str, Any],
    key: str,
    *,
    label: str,
    required_splits: tuple[str, ...] = (),
) -> dict[str, str]:
    value = metadata.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} metadata is missing {key}")
    result = {str(name): fingerprint for name, fingerprint in value.items()}
    missing = [name for name in required_splits if name not in result]
    if missing:
        raise ValueError(f"{label} {key} is missing: {', '.join(missing)}")
    for name, fingerprint in result.items():
        if not isinstance(fingerprint, str) or re.fullmatch(r"[0-9a-fA-F]{64}", fingerprint) is None:
            raise ValueError(f"{label} {key}[{name!r}] is not a SHA-256 fingerprint")
        result[name] = fingerprint.lower()
    if key == "source_fingerprints" and not result:
        raise ValueError(f"{label} metadata has no source_fingerprints")
    return result


def _evaluation_split_fingerprint(
    run: dict[str, Any],
    prepared_fingerprints: dict[str, str],
    *,
    label: str,
) -> str:
    split = run.get("split")
    if split != "test":
        raise ValueError(f"{label} comparison evaluation must use the test split")
    fingerprint = run.get("split_fingerprint")
    if not isinstance(fingerprint, str) or re.fullmatch(r"[0-9a-fA-F]{64}", fingerprint) is None:
        raise ValueError(f"{label} evaluation split fingerprint is missing or invalid")
    fingerprint = fingerprint.lower()
    if fingerprint != prepared_fingerprints["test"]:
        raise ValueError(
            f"{label} evaluation split fingerprint does not match prepared split fingerprint for test"
        )
    return fingerprint


def _aspect_inventory(metadata: dict[str, Any], *, label: str) -> list[str]:
    value = metadata.get("aspect_inventory")
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(aspect, str) or not aspect.strip() for aspect in value)
    ):
        raise ValueError(f"{label} metadata is missing a valid non-empty aspect_inventory")
    return sorted(set(value))


def compare_experiments(
    gas_experiment: str | Path,
    paraphrase_experiment: str | Path,
    output_dir: str | Path = "artifacts/comparisons",
) -> Path:
    gas_root = Path(gas_experiment).expanduser().resolve()
    para_root = Path(paraphrase_experiment).expanduser().resolve()
    gas_config = load_config(gas_root / "config.yaml")
    para_config = load_config(para_root / "config.yaml")
    gas_metadata = _read_json(gas_root / "metadata.json")
    para_metadata = _read_json(para_root / "metadata.json")
    gas_run = _read_json(gas_root / "evaluation" / "run_metadata.json")
    para_run = _read_json(para_root / "evaluation" / "run_metadata.json")
    gas_metrics = _read_json(gas_root / "evaluation" / "metrics.json")
    para_metrics = _read_json(para_root / "evaluation" / "metrics.json")

    gas_aspects = _aspect_inventory(gas_metadata, label="GAS")
    para_aspects = _aspect_inventory(para_metadata, label="Paraphrase")
    _require_equal(gas_aspects, para_aspects, "aspect_inventory")

    if gas_config["model"]["type"] != "gas" or para_config["model"]["type"] != "paraphrase":
        raise ValueError("Inputs must be one GAS and one Paraphrase experiment, in that order")
    gas_split_fingerprints = _fingerprint_map(
        gas_metadata,
        "split_fingerprints",
        label="GAS",
        required_splits=("train", "dev", "test"),
    )
    para_split_fingerprints = _fingerprint_map(
        para_metadata,
        "split_fingerprints",
        label="Paraphrase",
        required_splits=("train", "dev", "test"),
    )
    _require_equal(
        gas_split_fingerprints,
        para_split_fingerprints,
        "split_fingerprints (full train/dev/test split fingerprint mapping)",
    )
    gas_source_fingerprints = _fingerprint_map(
        gas_metadata,
        "source_fingerprints",
        label="GAS",
    )
    para_source_fingerprints = _fingerprint_map(
        para_metadata,
        "source_fingerprints",
        label="Paraphrase",
    )
    _require_equal(
        gas_source_fingerprints,
        para_source_fingerprints,
        "source_fingerprints",
    )
    gas_evaluation_fingerprint = _evaluation_split_fingerprint(
        gas_run,
        gas_split_fingerprints,
        label="GAS",
    )
    para_evaluation_fingerprint = _evaluation_split_fingerprint(
        para_run,
        para_split_fingerprints,
        label="Paraphrase",
    )
    _require_equal(
        gas_evaluation_fingerprint,
        para_evaluation_fingerprint,
        "evaluation split fingerprint",
    )
    required_pairs = (
        (gas_metadata["domain"], para_metadata["domain"], "domain"),
        (gas_metadata["split_mode"], para_metadata["split_mode"], "split_mode"),
        (gas_metadata["dataset_version"], para_metadata["dataset_version"], "dataset_version"),
        (gas_metadata["sentiment_mode"], para_metadata["sentiment_mode"], "sentiment_mode"),
        (gas_metadata["seed"], para_metadata["seed"], "seed"),
        (gas_metadata.get("cleaning_configuration", {}), para_metadata.get("cleaning_configuration", {}), "cleaning_configuration"),
        (gas_run["split"], para_run["split"], "evaluation split"),
        (gas_run["sample_count"], para_run["sample_count"], "evaluation sample count"),
        (gas_metadata["train_samples"], para_metadata["train_samples"], "training sample count"),
        (gas_metadata["validation_samples"], para_metadata["validation_samples"], "validation sample count"),
    )
    for left, right, name in required_pairs:
        _require_equal(left, right, name)

    # Compare the evaluated artifacts against the experiment snapshots as well
    # as against each other. This prevents stale evaluations from being
    # reported under a newer or otherwise mismatched config.
    for metadata, run, config, label in (
        (gas_metadata, gas_run, gas_config, "GAS"),
        (para_metadata, para_run, para_config, "Paraphrase"),
    ):
        for left, right, name in (
            (metadata.get("model_type"), config["model"]["type"], "experiment model type"),
            (metadata.get("model_backbone"), config["model"]["model_name"], "experiment backbone"),
            (
                metadata.get("tokenizer", metadata.get("tokenizer_name")),
                config["model"]["tokenizer_name"],
                "experiment tokenizer",
            ),
        ):
            _require_equal(left, right, f"{label} {name}")
        checks = (
            (run.get("model_type"), config["model"]["type"], "model type"),
            (run.get("model_backbone"), config["model"]["model_name"], "model backbone"),
            (run.get("tokenizer"), config["model"]["tokenizer_name"], "tokenizer"),
            (run.get("domain"), metadata["domain"], "domain"),
            (run.get("split_mode"), metadata["split_mode"], "split mode"),
            (run.get("dataset_version"), metadata["dataset_version"], "dataset version"),
            (run.get("sentiment_mode"), metadata["sentiment_mode"], "sentiment mode"),
            (run.get("seed"), metadata["seed"], "seed"),
        )
        for left, right, name in checks:
            _require_equal(left, right, f"{label} evaluation {name}")
        _require_equal(
            _aspect_inventory(run, label=f"{label} evaluation"),
            gas_aspects if label == "GAS" else para_aspects,
            f"{label} evaluation aspect_inventory",
        )

        run_split_fingerprints = _fingerprint_map(
            run,
            "split_fingerprints",
            label=f"{label} evaluation",
            required_splits=("train", "dev", "test"),
        )
        _require_equal(
            run_split_fingerprints,
            gas_split_fingerprints if label == "GAS" else para_split_fingerprints,
            f"{label} evaluation split_fingerprints",
        )
        dataset_identity = run.get("dataset_identity")
        if not isinstance(dataset_identity, Mapping):
            raise ValueError(f"{label} evaluation metadata is missing dataset_identity")
        expected_identity = {
            "domain": metadata["domain"],
            "dataset_version": metadata["dataset_version"],
            "split_mode": metadata["split_mode"],
            "sentiment_mode": metadata["sentiment_mode"],
            "seed": metadata["seed"],
            "cleaning_configuration": metadata.get("cleaning_configuration", {}),
            "split_fingerprints": gas_split_fingerprints if label == "GAS" else para_split_fingerprints,
            "source_fingerprints": gas_source_fingerprints if label == "GAS" else para_source_fingerprints,
        }
        _require_equal(dict(dataset_identity), expected_identity, f"{label} evaluation dataset_identity")

        evaluation_settings = run.get("evaluation_settings")
        if not isinstance(evaluation_settings, Mapping):
            raise ValueError(f"{label} evaluation metadata is missing evaluation_settings")
        _require_equal(
            evaluation_settings.get("sequence"),
            config.get("sequence"),
            f"{label} evaluation sequence settings",
        )
        _require_equal(
            evaluation_settings.get("generation"),
            config.get("generation"),
            f"{label} evaluation generation settings",
        )

    for section in ("training", "sequence", "generation"):
        _require_equal(
            gas_config.get(section),
            para_config.get(section),
            f"{section} configuration",
        )

    domain = gas_metadata["domain"]
    result = {
        "comparison": f"{domain}_gas_vs_paraphrase",
        "compatibility": {
            "domain": domain,
            "split_mode": gas_metadata["split_mode"],
            "dataset_version": gas_metadata["dataset_version"],
            "cleaning_configuration": gas_metadata.get("cleaning_configuration", {}),
            "sentiment_mode": gas_metadata["sentiment_mode"],
            "evaluation_split": gas_run["split"],
            "split_fingerprint": gas_evaluation_fingerprint,
            "split_fingerprints": gas_split_fingerprints,
            "source_fingerprints": gas_source_fingerprints,
            "seed": gas_metadata["seed"],
        },
        "sample_counts": {
            "training": gas_metadata["train_samples"],
            "validation": gas_metadata["validation_samples"],
            "test": gas_run["sample_count"] if gas_run["split"] == "test" else None,
            "evaluation": gas_run["sample_count"],
        },
        "models": [
            {
                "model": "GAS-T5",
                "model_type": "gas",
                "model_backbone": gas_config["model"]["model_name"],
                "precision": gas_metrics["precision"],
                "recall": gas_metrics["recall"],
                "exact_match_f1": gas_metrics["f1"],
                "invalid_generations": gas_run.get("invalid_generation_count", 0),
                "seed": gas_metadata["seed"],
                "configuration": to_serializable_config(gas_config),
                "checkpoint": gas_run["checkpoint"],
            },
            {
                "model": "Paraphrase-T5",
                "model_type": "paraphrase",
                "model_backbone": para_config["model"]["model_name"],
                "precision": para_metrics["precision"],
                "recall": para_metrics["recall"],
                "exact_match_f1": para_metrics["f1"],
                "invalid_generations": para_run.get("invalid_generation_count", 0),
                "seed": para_metadata["seed"],
                "configuration": to_serializable_config(para_config),
                "checkpoint": para_run["checkpoint"],
            },
        ],
    }
    destination_dir = Path(output_dir).expanduser()
    if not destination_dir.is_absolute():
        destination_dir = Path.cwd() / destination_dir
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{domain}_gas_vs_paraphrase.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare compatible OATS GAS and Paraphrase experiments")
    parser.add_argument("--gas-experiment", required=True, help="GAS experiment artifact directory")
    parser.add_argument("--paraphrase-experiment", required=True, help="Paraphrase experiment artifact directory")
    parser.add_argument("--output-dir", default="artifacts/comparisons")
    args = parser.parse_args(argv)
    output = compare_experiments(args.gas_experiment, args.paraphrase_experiment, args.output_dir)
    print(json.dumps({"comparison": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
