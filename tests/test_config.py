from pathlib import Path

import pytest

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


def test_repository_ships_a_complete_unique_three_domain_config_matrix():
    project_root = Path(__file__).resolve().parents[1]
    domains = ("amazon_ff", "coursera", "hotels")
    model_types = ("gas", "paraphrase")
    expected_names = {
        f"{model_type}_t5_{domain}.yaml"
        for domain in domains
        for model_type in model_types
    }
    config_paths = sorted((project_root / "configs").glob("*_t5_*.yaml"))

    assert {path.name for path in config_paths} == expected_names

    configs = {path.name: load_config(path) for path in config_paths}
    experiment_names = set()
    output_directories = set()
    for domain in domains:
        paired = []
        for model_type in model_types:
            filename = f"{model_type}_t5_{domain}.yaml"
            config = configs[filename]
            expected_experiment = f"{model_type}_t5_{domain}_seed42"
            assert config["experiment"] == {
                "name": expected_experiment,
                "domain": domain,
                "split_mode": "provided_split",
                "seed": 42,
            }
            assert config["model"]["type"] == model_type
            assert config["dataset"]["sentiment_mode"] == "paper_compatible"
            assert config["dataset"]["version"] == "oats_provided_txt_v1"
            assert Path(config["paths"]["processed_data"]).name == "processed"
            assert Path(config["paths"]["output_dir"]).name == expected_experiment
            experiment_names.add(config["experiment"]["name"])
            output_directories.add(config["paths"]["output_dir"])
            paired.append(config)

        for section in ("training", "sequence", "generation", "cleaning", "evaluation"):
            assert paired[0][section] == paired[1][section]

    assert len(experiment_names) == 6
    assert len(output_directories) == 6


@pytest.mark.parametrize("domain", ["amazon_ff", "coursera", "hotels"])
def test_validate_config_accepts_every_supported_domain(domain):
    config = {
        "experiment": {"name": f"demo-{domain}", "domain": domain, "split_mode": "clean_split", "seed": 42},
        "model": {"type": "gas", "model_name": "t5-base", "tokenizer_name": "t5-base"},
        "dataset": {"version": "oats_xml_v1", "sentiment_mode": "paper_compatible"},
        "training": {
            "epochs": 1,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 0.001,
        },
        "sequence": {"max_input_length": 16, "max_output_length": 16},
        "generation": {"num_beams": 1, "max_new_tokens": 8, "early_stopping": False},
        "paths": {"raw_data": "raw", "processed_data": "processed", "output_dir": "output"},
    }

    assert validate_config(config) == []


def test_repository_keeps_a_clean_split_config_outside_the_reported_matrix():
    """The six reported configs use provided_split, where review-level leakage cannot be
    checked. Keep at least one clean_split config loadable so the primary protocol and its
    leakage guard stay exercised, with its own processed root and output directory."""
    project_root = Path(__file__).resolve().parents[1]
    clean_split_paths = sorted((project_root / "configs" / "clean_split").glob("*.yaml"))

    assert clean_split_paths, "expected at least one clean_split config"

    matrix_output_dirs = {
        load_config(path)["paths"]["output_dir"]
        for path in (project_root / "configs").glob("*_t5_*.yaml")
    }
    for path in clean_split_paths:
        config = load_config(path)
        assert config["experiment"]["split_mode"] == "clean_split"
        assert config["dataset"]["version"] == "oats_xml_v1"
        assert config["cleaning"]["version"] == "review_level_clean_v1"
        assert config["paths"]["output_dir"] not in matrix_output_dirs
        assert Path(config["paths"]["processed_data"]).name != "processed"


def test_paths_resolve_against_the_project_root_for_nested_config_directories(tmp_path):
    """Configs in a subdirectory of configs/ (e.g. configs/clean_split/) must resolve
    relative paths against the project root, not against their own folder."""
    project_root = Path(__file__).resolve().parents[1]
    nested = project_root / "configs" / "clean_split" / "paraphrase_t5_hotels_clean_split.yaml"

    config = load_config(nested)

    assert config["_project_root"] == project_root
    assert config["paths"]["processed_data"] == project_root / "data" / "processed_clean_split"
    assert config["paths"]["raw_data"] == project_root / "OATS-ABSA-Dataset" / "data"


def test_config_outside_a_configs_directory_resolves_against_its_own_folder(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    source = (project_root / "configs" / "gas_t5_hotels.yaml").read_text(encoding="utf-8")
    relocated = tmp_path / "experiment.yaml"
    relocated.write_text(source, encoding="utf-8")

    config = load_config(relocated)

    assert config["_project_root"] == tmp_path
    assert config["paths"]["processed_data"] == tmp_path / "data" / "processed"
