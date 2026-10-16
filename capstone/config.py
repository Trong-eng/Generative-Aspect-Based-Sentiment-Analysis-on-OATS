"""Configuration loading and validation shared by the command-line tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any


DOMAINS = {"amazon_ff", "coursera", "hotels"}
SPLIT_MODES = {"original_split", "clean_split", "provided_split"}
SENTIMENT_MODES = {"raw_4_class", "paper_compatible"}
MODEL_TYPES = {"gas", "paraphrase"}


def _require_yaml() -> Any:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("PyYAML is required to load experiment configs; install requirements.txt") from exc
    return yaml


def _project_root(config_path: Path) -> Path:
    parent = config_path.parent
    return parent.parent if parent.name == "configs" else parent


def load_config(path: str | Path) -> dict[str, Any]:
    """Read YAML and resolve configured paths relative to the project root."""
    config_path = Path(path).expanduser().resolve()
    yaml = _require_yaml()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")

    paths = config.setdefault("paths", {})
    if not isinstance(paths, dict):
        raise ValueError("The 'paths' configuration must be a mapping")
    root = _project_root(config_path)
    for name, value in tuple(paths.items()):
        if value is None:
            continue
        candidate = Path(str(value)).expanduser()
        paths[name] = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    cleaning = config.get("cleaning")
    if isinstance(cleaning, dict):
        manifest_path = cleaning.get("unannotated_reviews_manifest")
        if manifest_path is not None:
            candidate = Path(str(manifest_path)).expanduser()
            resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
            cleaning["unannotated_reviews_manifest"] = str(resolved)
    config["_config_path"] = config_path
    config["_project_root"] = root
    errors = validate_config(config)
    if errors:
        raise ValueError("Invalid experiment configuration:\n- " + "\n- ".join(errors))
    return config


def validate_config(config: dict[str, Any]) -> list[str]:
    """Return actionable validation errors without mutating the config."""
    errors: list[str] = []
    experiment = config.get("experiment")
    model = config.get("model")
    dataset = config.get("dataset")
    paths = config.get("paths")
    for name, value in (("experiment", experiment), ("model", model), ("dataset", dataset), ("paths", paths)):
        if not isinstance(value, dict):
            errors.append(f"{name} must be a mapping")
    if errors:
        return errors

    assert isinstance(experiment, dict)
    assert isinstance(model, dict)
    assert isinstance(dataset, dict)
    assert isinstance(paths, dict)

    if not str(experiment.get("name", "")).strip():
        errors.append("experiment.name is required")
    if experiment.get("domain") not in DOMAINS:
        errors.append(f"experiment.domain must be one of {sorted(DOMAINS)}")
    if experiment.get("split_mode") not in SPLIT_MODES:
        errors.append(f"experiment.split_mode must be one of {sorted(SPLIT_MODES)}")
    try:
        if int(experiment.get("seed")) < 0:
            errors.append("experiment.seed must be a non-negative integer")
    except (TypeError, ValueError):
        errors.append("experiment.seed must be a non-negative integer")
    if model.get("type") not in MODEL_TYPES:
        errors.append(f"model.type must be one of {sorted(MODEL_TYPES)}")
    for key in ("model_name", "tokenizer_name"):
        if not str(model.get(key, "")).strip():
            errors.append(f"model.{key} is required")
    if dataset.get("sentiment_mode") not in SENTIMENT_MODES:
        errors.append(f"dataset.sentiment_mode must be one of {sorted(SENTIMENT_MODES)}")
    if not str(dataset.get("version", "")).strip():
        errors.append("dataset.version is required")
    for section in ("training", "sequence", "generation"):
        if not isinstance(config.get(section), dict):
            errors.append(f"{section} must be a mapping")
    for section in ("cleaning", "evaluation", "inference", "logging"):
        if section in config and not isinstance(config[section], dict):
            errors.append(f"{section} must be a mapping when provided")
    cleaning = config.get("cleaning", {})
    if isinstance(cleaning, dict):
        remove_unannotated = cleaning.get("remove_unannotated_reviews", False)
        if not isinstance(remove_unannotated, bool):
            errors.append("cleaning.remove_unannotated_reviews must be a boolean")
        manifest = cleaning.get("unannotated_reviews_manifest")
        if manifest is not None and not str(manifest).strip():
            errors.append("cleaning.unannotated_reviews_manifest must be a non-empty path or null")
        if remove_unannotated and manifest is None:
            errors.append(
                "cleaning.unannotated_reviews_manifest is required when "
                "unannotated review removal is enabled"
            )
        if manifest is not None and not remove_unannotated:
            errors.append(
                "cleaning.remove_unannotated_reviews must be true when a manifest is configured"
            )
        if experiment.get("split_mode") == "provided_split" and (
            remove_unannotated or manifest is not None
        ):
            errors.append("provided_split cannot use review-level annotation manifests")
    training = config.get("training")
    sequence = config.get("sequence")
    generation = config.get("generation")
    evaluation = config.get("evaluation", {})
    if isinstance(training, dict):
        for key in ("epochs", "batch_size", "gradient_accumulation_steps"):
            try:
                if int(training.get(key, 0)) < 1:
                    errors.append(f"training.{key} must be a positive integer")
            except (TypeError, ValueError):
                errors.append(f"training.{key} must be a positive integer")
        for key in ("learning_rate",):
            try:
                if float(training.get(key, 0)) <= 0:
                    errors.append(f"training.{key} must be positive")
            except (TypeError, ValueError):
                errors.append(f"training.{key} must be positive")
        for key in ("weight_decay",):
            if key in training:
                try:
                    if float(training[key]) < 0:
                        errors.append(f"training.{key} must be non-negative")
                except (TypeError, ValueError):
                    errors.append(f"training.{key} must be a number")
        if "warmup_ratio" in training:
            try:
                if not 0 <= float(training["warmup_ratio"]) <= 1:
                    errors.append("training.warmup_ratio must be between 0 and 1")
            except (TypeError, ValueError):
                errors.append("training.warmup_ratio must be between 0 and 1")
    if isinstance(sequence, dict):
        for key in ("max_input_length", "max_output_length"):
            try:
                if int(sequence.get(key, 0)) < 1:
                    errors.append(f"sequence.{key} must be a positive integer")
            except (TypeError, ValueError):
                errors.append(f"sequence.{key} must be a positive integer")
    if isinstance(generation, dict):
        for key in ("num_beams", "max_new_tokens"):
            try:
                if int(generation.get(key, 0)) < 1:
                    errors.append(f"generation.{key} must be a positive integer")
            except (TypeError, ValueError):
                errors.append(f"generation.{key} must be a positive integer")
        if "early_stopping" not in generation:
            errors.append("generation.early_stopping is required")
        elif not isinstance(generation["early_stopping"], bool):
            errors.append("generation.early_stopping must be a boolean")
    if isinstance(evaluation, dict) and "batch_size" in evaluation:
        try:
            if int(evaluation["batch_size"]) < 1:
                errors.append("evaluation.batch_size must be a positive integer")
        except (TypeError, ValueError):
            errors.append("evaluation.batch_size must be a positive integer")
    if isinstance(config.get("inference", {}), dict):
        inference = config.get("inference", {})
        if "batch_size" in inference:
            try:
                if int(inference["batch_size"]) < 1:
                    errors.append("inference.batch_size must be a positive integer")
            except (TypeError, ValueError):
                errors.append("inference.batch_size must be a positive integer")
    for key in ("raw_data", "processed_data", "output_dir"):
        if key not in paths or paths[key] is None or not str(paths[key]).strip():
            errors.append(f"paths.{key} is required")
    return errors


def to_serializable_config(config: dict[str, Any]) -> dict[str, Any]:
    """Remove internal keys and convert resolved Path values to strings."""
    result: dict[str, Any] = {}
    for key, value in config.items():
        if key.startswith("_"):
            continue
        if key == "paths" and isinstance(value, dict):
            result[key] = {path_key: str(path_value) if isinstance(path_value, Path) else path_value
                           for path_key, path_value in value.items()}
        else:
            result[key] = value
    return result
