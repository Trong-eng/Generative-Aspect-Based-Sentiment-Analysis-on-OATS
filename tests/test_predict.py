import json
from pathlib import Path

import pytest

from capstone.predict import _known_aspects


def test_known_aspects_uses_experiment_ontology_instead_of_training_examples(tmp_path: Path):
    experiment = tmp_path / "experiment"
    checkpoint = experiment / "checkpoints" / "checkpoint-epoch-001"
    checkpoint.mkdir(parents=True)
    (experiment / "metadata.json").write_text(
        json.dumps({"aspect_inventory": ["hotel general", "rooms cleanliness"]}),
        encoding="utf-8",
    )

    assert _known_aspects(None, checkpoint) == {"hotel general", "rooms cleanliness"}


def test_known_aspects_rejects_an_empty_declared_inventory(tmp_path: Path):
    experiment = tmp_path / "experiment"
    checkpoint = experiment / "checkpoints" / "checkpoint-epoch-001"
    checkpoint.mkdir(parents=True)
    (experiment / "metadata.json").write_text(
        json.dumps({"aspect_inventory": []}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="aspect_inventory"):
        _known_aspects(None, checkpoint)


def test_known_aspects_falls_back_to_prepared_metadata(tmp_path: Path):
    processed = tmp_path / "processed"
    domain_dir = processed / "hotels"
    domain_dir.mkdir(parents=True)
    (domain_dir / "metadata.json").write_text(
        json.dumps({"aspect_inventory": ["hotel general"]}),
        encoding="utf-8",
    )

    config = {
        "experiment": {"domain": "hotels"},
        "paths": {"processed_data": processed},
    }
    assert _known_aspects(config, tmp_path / "standalone-checkpoint") == {"hotel general"}


def test_known_aspects_rejects_missing_inventory_in_experiment_metadata(tmp_path: Path):
    experiment = tmp_path / "experiment"
    checkpoint = experiment / "checkpoints" / "checkpoint-epoch-001"
    checkpoint.mkdir(parents=True)
    (experiment / "metadata.json").write_text(json.dumps({"domain": "hotels"}), encoding="utf-8")

    with pytest.raises(ValueError, match="aspect_inventory"):
        _known_aspects(None, checkpoint)


def test_known_aspects_requires_an_inventory_for_standalone_checkpoint(tmp_path: Path):
    with pytest.raises(ValueError, match="aspect inventory"):
        _known_aspects(None, tmp_path / "standalone-checkpoint")
