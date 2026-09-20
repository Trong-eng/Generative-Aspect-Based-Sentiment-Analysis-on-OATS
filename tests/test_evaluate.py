import hashlib
import json
from pathlib import Path

import pytest

from capstone.evaluate import run_evaluation
from capstone.models.base import ParsedGeneration, Seq2SeqGenerator


def _config_text(
    processed: Path,
    output: Path,
    *,
    model_name="t5-base",
    tokenizer_name="t5-base",
    max_input_length=128,
    num_beams=1,
):
    return (
        "experiment:\n  name: eval-test\n  domain: hotels\n  split_mode: clean_split\n  seed: 7\n"
        f"model:\n  type: gas\n  model_name: {model_name}\n  tokenizer_name: {tokenizer_name}\n"
        "training:\n  epochs: 1\n  batch_size: 1\n  gradient_accumulation_steps: 1\n"
        "  learning_rate: 0.001\n  weight_decay: 0.0\n  warmup_ratio: 0.0\n"
        "  optimizer: AdamW\n  scheduler: linear\n"
        f"sequence:\n  max_input_length: {max_input_length}\n  max_output_length: 128\n"
        f"generation:\n  num_beams: {num_beams}\n  max_new_tokens: 16\n  early_stopping: false\n"
        "dataset:\n  version: oats_xml_v1\n  sentiment_mode: paper_compatible\n"
        "cleaning:\n  version: review_level_clean_v1\n  remove_unannotated_reviews: false\n"
        "  invalid_aspect_policy: drop_and_log\n  conflict_policy: preserve\n"
        "logging:\n  level: DEBUG\n"
        f"paths:\n  raw_data: {processed.parent / 'raw'}\n"
        f"  processed_data: {processed}\n  output_dir: {output}\n"
    )


def _evaluation_tree(
    root: Path,
    *,
    prepared_fingerprint=None,
    experiment_split_fingerprints=None,
    aspect_inventory=None,
    experiment_aspect_inventory=None,
):
    experiment = root / "experiment"
    checkpoint = experiment / "checkpoints" / "checkpoint-epoch-001"
    checkpoint.mkdir(parents=True)
    (experiment / "best_checkpoint.json").write_text(
        json.dumps({"path": "checkpoints/checkpoint-epoch-001"}), encoding="utf-8"
    )
    processed = root / "processed"
    prepared = processed / "hotels"
    prepared.mkdir(parents=True)
    split_text = json.dumps(
        {"review_id": "r1", "sentence_id": "s1", "text": "No annotations.", "quadruples": []}
    ) + "\n"
    for split_name in ("train", "dev", "test"):
        (prepared / f"{split_name}.jsonl").write_text(split_text, encoding="utf-8")
    split_fingerprint = hashlib.sha256((prepared / "test.jsonl").read_bytes()).hexdigest()
    raw_source = root / "raw.xml"
    raw_source.write_bytes(b"fixture source identity")
    source_fingerprint = hashlib.sha256(raw_source.read_bytes()).hexdigest()
    metadata = {
        "domain": "hotels",
        "split_mode": "clean_split",
        "seed": 7,
        "sentiment_mode": "paper_compatible",
        "dataset_version": "oats_xml_v1",
        "cleaning_configuration": {
            "version": "review_level_clean_v1",
            "remove_unannotated_reviews": False,
            "invalid_aspect_policy": "drop_and_log",
            "conflict_policy": "preserve",
        },
        "split_fingerprints": {
            "train": split_fingerprint,
            "dev": split_fingerprint,
            "test": prepared_fingerprint or split_fingerprint,
        },
        "aspect_inventory": ["food quality"] if aspect_inventory is None else aspect_inventory,
        "source_paths": {"raw_xml": str(raw_source)},
        "source_fingerprints": {"raw_xml": source_fingerprint},
    }
    (prepared / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    experiment_metadata = {
        "domain": "hotels",
        "split_mode": "clean_split",
        "seed": 7,
        "dataset_version": "oats_xml_v1",
        "sentiment_mode": "paper_compatible",
        "model_type": "gas",
        "model_backbone": "t5-base",
        "tokenizer": "t5-base",
        "cleaning_configuration": metadata["cleaning_configuration"],
        "split_fingerprints": experiment_split_fingerprints or metadata["split_fingerprints"],
        "source_fingerprints": metadata["source_fingerprints"],
        "aspect_inventory": (
            metadata["aspect_inventory"]
            if experiment_aspect_inventory is None
            else experiment_aspect_inventory
        ),
    }
    (experiment / "metadata.json").write_text(json.dumps(experiment_metadata), encoding="utf-8")
    config_path = experiment / "config.yaml"
    config_path.write_text(_config_text(processed, experiment / "out"), encoding="utf-8")
    return experiment, config_path, split_fingerprint


def test_evaluation_uses_experiment_config_and_writes_under_selected_experiment(tmp_path, monkeypatch):
    experiment, config_path, fingerprint = _evaluation_tree(tmp_path)
    alternate_config = tmp_path / "requested.yaml"
    alternate_config.write_text(
        _config_text(tmp_path / "processed", tmp_path / "requested-output"),
        encoding="utf-8",
    )

    observed = {}

    class StubModel:
        def generate(self, texts, **kwargs):
            observed["early_stopping"] = kwargs["early_stopping"]
            return ["[]" for _ in texts]

    class StubLogger:
        def info(self, *_args):
            return None

    def configure_logging(_path, *, level):
        observed["logging_level"] = level
        return StubLogger()

    monkeypatch.setattr(
        Seq2SeqGenerator,
        "from_checkpoint",
        classmethod(lambda cls, checkpoint: StubModel()),
    )
    monkeypatch.setattr(
        "capstone.evaluate.parse_gas_generation",
        lambda raw, known_aspects=None: ParsedGeneration(),
    )
    monkeypatch.setattr("capstone.evaluate.configure_logging", configure_logging)

    output_dir = run_evaluation(alternate_config, experiment, split="test")

    assert output_dir == experiment / "evaluation"
    run_metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert run_metadata["split_fingerprint"] == fingerprint
    assert run_metadata["experiment_dir"] == str(experiment.resolve())
    assert run_metadata["checkpoint"] == "checkpoints/checkpoint-epoch-001"
    assert run_metadata["evaluation_settings"]["generation"]["num_beams"] == 1
    assert observed == {"early_stopping": False, "logging_level": "DEBUG"}


def test_evaluation_rejects_requested_config_that_differs_from_checkpoint_provenance(
    tmp_path, monkeypatch
):
    experiment, _, _ = _evaluation_tree(tmp_path)
    requested = tmp_path / "requested.yaml"
    processed = tmp_path / "processed"
    requested.write_text(
        _config_text(processed, experiment / "out", tokenizer_name="other-tokenizer"),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        Seq2SeqGenerator,
        "from_checkpoint",
        classmethod(lambda cls, checkpoint: pytest.fail("model load must not run")),
    )

    with pytest.raises(ValueError, match="tokenizer_name"):
        run_evaluation(requested, experiment, split="test")
    assert not (experiment / "evaluation").exists()


def test_evaluation_rejects_checkpoint_with_different_aspect_inventory(tmp_path, monkeypatch):
    experiment, config_path, _ = _evaluation_tree(
        tmp_path,
        aspect_inventory=["food quality"],
        experiment_aspect_inventory=["hotel service"],
    )
    monkeypatch.setattr(
        Seq2SeqGenerator,
        "from_checkpoint",
        classmethod(lambda cls, checkpoint: pytest.fail("model load must not run")),
    )

    with pytest.raises(ValueError, match="aspect inventory"):
        run_evaluation(config_path, experiment, split="test")


def test_evaluation_rejects_requested_generation_settings_that_differ_from_checkpoint(
    tmp_path, monkeypatch
):
    experiment, _, _ = _evaluation_tree(tmp_path)
    requested = tmp_path / "requested.yaml"
    requested.write_text(
        _config_text(tmp_path / "processed", experiment / "out", num_beams=4),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        Seq2SeqGenerator,
        "from_checkpoint",
        classmethod(lambda cls, checkpoint: pytest.fail("model load must not run")),
    )

    with pytest.raises(ValueError, match="generation"):
        run_evaluation(requested, experiment, split="test")
    assert not (experiment / "evaluation").exists()


def test_evaluation_rejects_prepared_split_fingerprint_mismatch(tmp_path):
    experiment, config_path, _ = _evaluation_tree(tmp_path, prepared_fingerprint="0" * 64)

    with pytest.raises(ValueError, match="fingerprint"):
        run_evaluation(config_path, experiment, split="test")


def test_evaluation_rejects_prepared_data_that_differs_from_checkpoint_provenance(tmp_path):
    experiment, config_path, _ = _evaluation_tree(
        tmp_path,
        experiment_split_fingerprints={
            "train": "1" * 64,
            "dev": "1" * 64,
            "test": "1" * 64,
        },
    )

    with pytest.raises(ValueError, match="checkpoint experiment provenance"):
        run_evaluation(config_path, experiment, split="test")


@pytest.mark.parametrize("inventory", [[], [" "]])
def test_evaluation_fails_closed_without_a_nonempty_aspect_inventory(tmp_path, inventory, monkeypatch):
    experiment, config_path, _ = _evaluation_tree(tmp_path, aspect_inventory=inventory)
    monkeypatch.setattr(
        Seq2SeqGenerator,
        "from_checkpoint",
        classmethod(lambda cls, checkpoint: pytest.fail("model load must not run")),
    )

    with pytest.raises(ValueError, match="aspect_inventory"):
        run_evaluation(config_path, experiment, split="test")
