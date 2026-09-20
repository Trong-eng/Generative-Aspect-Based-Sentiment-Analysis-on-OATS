from pathlib import Path

from capstone.config import load_config, validate_config
from capstone.models.base import resolve_checkpoint


def test_load_config_resolves_relative_paths_against_repository_root(tmp_path: Path):
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        "experiment:\n  name: demo\n  domain: hotels\n  split_mode: clean_split\n  seed: 42\n"
        "model:\n  type: paraphrase\n  model_name: t5-base\n  tokenizer_name: t5-base\n"
        "training:\n  epochs: 1\n  batch_size: 1\n  gradient_accumulation_steps: 1\n  learning_rate: 0.001\n  optimizer: AdamW\n  scheduler: linear\n"
        "sequence:\n  max_input_length: 128\n  max_output_length: 128\n"
        "generation:\n  num_beams: 1\n  max_new_tokens: 128\n  early_stopping: false\n"
        "dataset:\n  version: oats_xml_v1\n  sentiment_mode: paper_compatible\n"
        "cleaning:\n  remove_unannotated_reviews: true\n"
        "  unannotated_reviews_manifest: annotation_status/hotels.json\n"
        "paths:\n  raw_data: OATS-ABSA-Dataset/data\n  processed_data: data/processed\n"
        "  output_dir: artifacts/experiments/demo\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config["paths"]["raw_data"] == (tmp_path / "OATS-ABSA-Dataset/data").resolve()
    assert config["experiment"]["seed"] == 42
    assert config["cleaning"]["unannotated_reviews_manifest"] == str(
        (tmp_path / "annotation_status/hotels.json").resolve()
    )


def test_validate_config_rejects_unsupported_domain_and_split_mode():
    config = {
        "experiment": {"name": "bad", "domain": "other", "split_mode": "sentence_split", "seed": 0},
        "model": {"type": "gas", "model_name": "t5-base", "tokenizer_name": "t5-base"},
        "dataset": {"version": "oats_xml_v1", "sentiment_mode": "paper_compatible"},
        "training": {},
        "sequence": {},
        "generation": {},
        "paths": {},
    }

    errors = validate_config(config)

    assert "domain" in errors[0]
    assert "split_mode" in errors[1]


def test_config_requires_manifest_when_unannotated_review_removal_is_enabled():
    config = {
        "experiment": {"name": "demo", "domain": "hotels", "split_mode": "clean_split", "seed": 42},
        "model": {"type": "gas", "model_name": "t5-base", "tokenizer_name": "t5-base"},
        "dataset": {"version": "oats_xml_v1", "sentiment_mode": "paper_compatible"},
        "training": {"epochs": 1, "batch_size": 1, "gradient_accumulation_steps": 1, "learning_rate": 0.001},
        "sequence": {"max_input_length": 128, "max_output_length": 128},
        "generation": {"num_beams": 1, "max_new_tokens": 128},
        "cleaning": {"remove_unannotated_reviews": True, "unannotated_reviews_manifest": None},
        "paths": {"raw_data": "raw", "processed_data": "processed", "output_dir": "output"},
    }

    errors = validate_config(config)

    assert any("manifest is required" in error for error in errors)


def test_validate_config_rejects_non_boolean_early_stopping():
    config = {
        "experiment": {"name": "demo", "domain": "hotels", "split_mode": "clean_split", "seed": 42},
        "model": {"type": "gas", "model_name": "t5-base", "tokenizer_name": "t5-base"},
        "dataset": {"version": "oats_xml_v1", "sentiment_mode": "paper_compatible"},
        "training": {
            "epochs": 1,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 0.001,
            "optimizer": "AdamW",
            "scheduler": "linear",
        },
        "sequence": {"max_input_length": 128, "max_output_length": 128},
        "generation": {"num_beams": 1, "max_new_tokens": 16, "early_stopping": "false"},
        "paths": {"raw_data": "raw", "processed_data": "processed", "output_dir": "output"},
    }

    errors = validate_config(config)

    assert "generation.early_stopping must be a boolean" in errors


def test_resolve_checkpoint_accepts_experiment_directory(tmp_path: Path):
    experiment = tmp_path / "experiment"
    checkpoint = experiment / "checkpoints" / "checkpoint-epoch-002"
    checkpoint.mkdir(parents=True)
    (experiment / "best_checkpoint.json").write_text(
        '{"path": "checkpoints/checkpoint-epoch-002"}',
        encoding="utf-8",
    )

    assert resolve_checkpoint(experiment) == checkpoint
