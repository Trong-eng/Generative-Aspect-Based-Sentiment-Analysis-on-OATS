import json
from pathlib import Path

import pytest

from capstone.compare import compare_experiments


def _write_experiment(
    root: Path,
    model_type: str,
    model_name: str,
    *,
    split_mode="clean_split",
    split_fingerprint="a" * 64,
    prepared_split_fingerprint=None,
    prepared_split_fingerprints=None,
    source_fingerprints=None,
    include_split_fingerprint=True,
    include_source_fingerprints=True,
    run_split="test",
    run_num_beams=1,
    run_aspect_inventory=None,
    domain="hotels",
):
    root.mkdir(parents=True)
    (root / "config.yaml").write_text(
        "experiment:\n"
        f"  name: {root.name}\n  domain: {domain}\n  split_mode: {split_mode}\n  seed: 42\n"
        f"model:\n  type: {model_type}\n  model_name: {model_name}\n  tokenizer_name: {model_name}\n"
        "training:\n  epochs: 20\n  batch_size: 16\n  gradient_accumulation_steps: 1\n"
        "  learning_rate: 0.0003\n  weight_decay: 0.0\n  warmup_ratio: 0.0\n  optimizer: AdamW\n  scheduler: linear\n"
        "sequence:\n  max_input_length: 128\n  max_output_length: 128\n"
        "generation:\n  num_beams: 1\n  max_new_tokens: 128\n  early_stopping: false\n"
        "dataset:\n  sentiment_mode: paper_compatible\n  version: oats_xml_v1\n"
        "cleaning:\n  version: review_level_clean_v1\n  remove_unannotated_reviews: false\n"
        "  invalid_aspect_policy: drop_and_log\n  conflict_policy: preserve\n"
        "paths:\n  raw_data: raw\n  processed_data: processed\n  output_dir: out\n",
        encoding="utf-8",
    )
    experiment_metadata = {
        "domain": domain, "split_mode": split_mode, "dataset_version": "oats_xml_v1",
        "sentiment_mode": "paper_compatible", "seed": 42,
        "model_type": model_type, "model_backbone": model_name, "tokenizer": model_name,
        "cleaning_configuration": {
            "version": "review_level_clean_v1",
            "remove_unannotated_reviews": False,
            "invalid_aspect_policy": "drop_and_log",
            "conflict_policy": "preserve",
        },
        "train_samples": 8, "validation_samples": 1,
        "aspect_inventory": ["food quality"],
    }
    experiment_metadata["split_fingerprints"] = prepared_split_fingerprints or {
        "train": split_fingerprint,
        "dev": split_fingerprint,
        "test": prepared_split_fingerprint or split_fingerprint,
    }
    if include_source_fingerprints:
        experiment_metadata["source_fingerprints"] = (
            source_fingerprints if source_fingerprints is not None else {"raw_xml": "c" * 64}
        )
    (root / "metadata.json").write_text(json.dumps(experiment_metadata), encoding="utf-8")
    evaluation = root / "evaluation"
    evaluation.mkdir()
    (evaluation / "metrics.json").write_text(json.dumps({
        "precision": 0.5, "recall": 0.25, "f1": 1 / 3,
    }), encoding="utf-8")
    run_metadata = {
        "split": run_split, "sample_count": 1, "invalid_generation_count": 0,
        "checkpoint": str(root / "checkpoints" / "best"),
        "model_type": model_type,
        "model_backbone": model_name,
        "tokenizer": model_name,
        "domain": domain,
        "split_mode": split_mode,
        "dataset_version": "oats_xml_v1",
        "sentiment_mode": "paper_compatible",
        "seed": 42,
        "split_fingerprints": experiment_metadata["split_fingerprints"],
        "evaluation_settings": {
            "sequence": {"max_input_length": 128, "max_output_length": 128},
            "generation": {
                "num_beams": run_num_beams,
                "max_new_tokens": 128,
                "early_stopping": False,
            },
        },
        "dataset_identity": {
            "domain": domain,
            "dataset_version": "oats_xml_v1",
            "split_mode": split_mode,
            "sentiment_mode": "paper_compatible",
            "seed": 42,
            "cleaning_configuration": experiment_metadata["cleaning_configuration"],
            "source_fingerprints": experiment_metadata.get("source_fingerprints", {"raw_xml": "c" * 64}),
            "split_fingerprints": experiment_metadata["split_fingerprints"],
        },
        "aspect_inventory": (
            experiment_metadata["aspect_inventory"]
            if run_aspect_inventory is None
            else run_aspect_inventory
        ),
    }
    if include_split_fingerprint:
        run_metadata["split_fingerprint"] = split_fingerprint
    (evaluation / "run_metadata.json").write_text(json.dumps(run_metadata), encoding="utf-8")


def test_compare_writes_report_only_for_compatible_gas_and_paraphrase(tmp_path: Path):
    gas = tmp_path / "gas"
    paraphrase = tmp_path / "paraphrase"
    _write_experiment(gas, "gas", "t5-base")
    _write_experiment(paraphrase, "paraphrase", "t5-base")

    output = compare_experiments(gas, paraphrase, tmp_path / "comparisons")

    assert output.name == "hotels_gas_vs_paraphrase.json"
    result = json.loads(output.read_text(encoding="utf-8"))
    assert [row["model_type"] for row in result["models"]] == ["gas", "paraphrase"]


@pytest.mark.parametrize("domain", ["amazon_ff", "coursera", "hotels"])
def test_compare_names_output_for_each_supported_domain(tmp_path: Path, domain: str):
    gas = tmp_path / f"gas-{domain}"
    paraphrase = tmp_path / f"paraphrase-{domain}"
    _write_experiment(gas, "gas", "t5-base", domain=domain)
    _write_experiment(paraphrase, "paraphrase", "t5-base", domain=domain)

    output = compare_experiments(gas, paraphrase, tmp_path / "comparisons" / "seed42")

    assert output.name == f"{domain}_gas_vs_paraphrase.json"


def test_compare_rejects_different_split_modes(tmp_path: Path):
    gas = tmp_path / "gas"
    paraphrase = tmp_path / "paraphrase"
    _write_experiment(gas, "gas", "t5-base", split_mode="clean_split")
    _write_experiment(paraphrase, "paraphrase", "t5-base", split_mode="original_split")

    with pytest.raises(ValueError, match="split_mode"):
        compare_experiments(gas, paraphrase, tmp_path / "comparisons")


def test_compare_rejects_mismatched_split_fingerprints(tmp_path: Path):
    gas = tmp_path / "gas"
    paraphrase = tmp_path / "paraphrase"
    _write_experiment(gas, "gas", "t5-base", split_fingerprint="a" * 64)
    _write_experiment(paraphrase, "paraphrase", "t5-base", split_fingerprint="b" * 64)

    with pytest.raises(ValueError, match="split_fingerprints"):
        compare_experiments(gas, paraphrase, tmp_path / "comparisons")


def test_compare_rejects_missing_split_fingerprint(tmp_path: Path):
    gas = tmp_path / "gas"
    paraphrase = tmp_path / "paraphrase"
    _write_experiment(gas, "gas", "t5-base", include_split_fingerprint=False)
    _write_experiment(paraphrase, "paraphrase", "t5-base")

    with pytest.raises(ValueError, match="split fingerprint"):
        compare_experiments(gas, paraphrase, tmp_path / "comparisons")


def test_compare_rejects_run_fingerprint_that_disagrees_with_experiment_provenance(tmp_path: Path):
    gas = tmp_path / "gas"
    paraphrase = tmp_path / "paraphrase"
    _write_experiment(
        gas,
        "gas",
        "t5-base",
        split_fingerprint="a" * 64,
        prepared_split_fingerprints={"train": "a" * 64, "dev": "a" * 64, "test": "b" * 64},
    )
    _write_experiment(
        paraphrase,
        "paraphrase",
        "t5-base",
        split_fingerprint="b" * 64,
        prepared_split_fingerprints={"train": "a" * 64, "dev": "a" * 64, "test": "b" * 64},
    )

    with pytest.raises(ValueError, match="prepared split fingerprint"):
        compare_experiments(gas, paraphrase, tmp_path / "comparisons")


def test_compare_rejects_different_source_fingerprints(tmp_path: Path):
    gas = tmp_path / "gas"
    paraphrase = tmp_path / "paraphrase"
    _write_experiment(gas, "gas", "t5-base", source_fingerprints={"raw_xml": "a" * 64})
    _write_experiment(paraphrase, "paraphrase", "t5-base", source_fingerprints={"raw_xml": "b" * 64})

    with pytest.raises(ValueError, match="source_fingerprints"):
        compare_experiments(gas, paraphrase, tmp_path / "comparisons")


def test_compare_rejects_missing_source_fingerprints(tmp_path: Path):
    gas = tmp_path / "gas"
    paraphrase = tmp_path / "paraphrase"
    _write_experiment(gas, "gas", "t5-base", include_source_fingerprints=False)
    _write_experiment(paraphrase, "paraphrase", "t5-base")

    with pytest.raises(ValueError, match="source_fingerprints"):
        compare_experiments(gas, paraphrase, tmp_path / "comparisons")


def test_compare_requires_all_split_fingerprints_to_match(tmp_path: Path):
    gas = tmp_path / "gas"
    paraphrase = tmp_path / "paraphrase"
    _write_experiment(
        gas,
        "gas",
        "t5-base",
        prepared_split_fingerprints={"train": "a" * 64, "dev": "b" * 64, "test": "c" * 64},
    )
    _write_experiment(
        paraphrase,
        "paraphrase",
        "t5-base",
        prepared_split_fingerprints={"train": "a" * 64, "dev": "d" * 64, "test": "c" * 64},
    )

    with pytest.raises(ValueError, match="split_fingerprints"):
        compare_experiments(gas, paraphrase, tmp_path / "comparisons")


def test_compare_requires_test_evaluation_and_test_fingerprint(tmp_path: Path):
    gas = tmp_path / "gas"
    paraphrase = tmp_path / "paraphrase"
    _write_experiment(gas, "gas", "t5-base", run_split="dev")
    _write_experiment(paraphrase, "paraphrase", "t5-base", run_split="dev")

    with pytest.raises(ValueError, match="test split"):
        compare_experiments(gas, paraphrase, tmp_path / "comparisons")


def test_compare_rejects_evaluation_settings_that_differ_from_experiment_config(tmp_path: Path):
    gas = tmp_path / "gas"
    paraphrase = tmp_path / "paraphrase"
    _write_experiment(gas, "gas", "t5-base", run_num_beams=4)
    _write_experiment(paraphrase, "paraphrase", "t5-base")

    with pytest.raises(ValueError, match="evaluation generation settings"):
        compare_experiments(gas, paraphrase, tmp_path / "comparisons")
    assert not (tmp_path / "comparisons").exists()


def test_compare_rejects_evaluation_using_a_different_aspect_inventory(tmp_path: Path):
    gas = tmp_path / "gas"
    paraphrase = tmp_path / "paraphrase"
    _write_experiment(gas, "gas", "t5-base", run_aspect_inventory=["hotel service"])
    _write_experiment(paraphrase, "paraphrase", "t5-base")

    with pytest.raises(ValueError, match="evaluation aspect_inventory"):
        compare_experiments(gas, paraphrase, tmp_path / "comparisons")
    assert not (tmp_path / "comparisons").exists()
