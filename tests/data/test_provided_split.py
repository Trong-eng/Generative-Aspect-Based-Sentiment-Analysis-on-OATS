import hashlib
import json
from pathlib import Path

import pytest

from capstone.audit_data import audit_processed, build_parser, run
from capstone.prepare_data import prepare_domain
from capstone.data.serialization import read_jsonl, validate_prepared_metadata


TRAIN_TEXT = (
    "Same text.####[['food', 'food quality', 'positive', 'amazing']]\n"
    "No label.####[]\n"
    "\n"
    "Same text.####[['service', 'service general', 'negative', 'slow']]\n"
    "Bad source label.####[['NULL', 'POLARITY#POSITIVE', 'positive', 'great']]\n"
)


def _write_split_sources(root: Path) -> Path:
    domain_dir = root / "amazon_ff" / "quads"
    domain_dir.mkdir(parents=True)
    (domain_dir / "train.txt").write_text(TRAIN_TEXT, encoding="utf-8")
    (domain_dir / "dev.txt").write_text("Dev text.####[]\n", encoding="utf-8")
    (domain_dir / "test.txt").write_text("Test text.####[]\n", encoding="utf-8")
    return domain_dir


def test_prepare_provided_split_uses_txt_rows_without_xml_or_review_ids(tmp_path):
    raw_root = tmp_path / "raw"
    source_dir = _write_split_sources(raw_root)
    processed_root = tmp_path / "processed"

    outputs = prepare_domain(
        "amazon_ff",
        split_mode="provided_split",
        seed=42,
        sentiment_mode="paper_compatible",
        raw_root=raw_root,
        output_root=processed_root,
    )

    train_payloads = [
        json.loads(line)
        for line in outputs["train"].read_text(encoding="utf-8").splitlines()
    ]
    train_records = read_jsonl(outputs["train"])
    assert [record.source_line_number for record in train_records] == [1, 2, 4, 5]
    assert [record.text for record in train_records] == [
        "Same text.",
        "No label.",
        "Same text.",
        "Bad source label.",
    ]
    assert [record.quadruples[0].sentiment for record in (train_records[0], train_records[2])] == [
        "positive",
        "negative",
    ]
    assert train_records[1].quadruples == []
    assert train_records[3].quadruples == []
    assert all("review_id" not in row and "sentence_id" not in row for row in train_payloads)

    metadata = json.loads(outputs["metadata"].read_text(encoding="utf-8"))
    assert metadata["split_mode"] == "provided_split"
    assert metadata["seed"] is None
    assert metadata["review_level_leakage_checked"] is False
    assert metadata["review_level_leakage_check"]["status"] == "not_verifiable"
    assert any(
        issue["reason"] == "invalid_aspect_category"
        and issue["source_line_number"] == 5
        for issue in metadata["issues"]
    )
    assert set(metadata["source_fingerprints"]) == {
        "provided_train_txt",
        "provided_dev_txt",
        "provided_test_txt",
    }
    for split_name in ("train", "dev", "test"):
        source_path = source_dir / f"{split_name}.txt"
        source_key = f"provided_{split_name}_txt"
        assert metadata["source_paths"][source_key] == str(source_path)
        assert metadata["source_fingerprints"][source_key] == hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest()

    validated = validate_prepared_metadata(
        processed_root / "amazon_ff",
        domain="amazon_ff",
        split_mode="provided_split",
        sentiment_mode="paper_compatible",
        dataset_version="oats_provided_txt_v1",
        cleaning_configuration=metadata["cleaning_configuration"],
    )
    assert validated["source_fingerprints"] == metadata["source_fingerprints"]
    source_path = source_dir / "train.txt"
    source_path.write_text(
        source_path.read_text(encoding="utf-8") + "changed\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source SHA-256"):
        validate_prepared_metadata(
            processed_root / "amazon_ff",
            domain="amazon_ff",
            split_mode="provided_split",
            sentiment_mode="paper_compatible",
            dataset_version="oats_provided_txt_v1",
            cleaning_configuration=metadata["cleaning_configuration"],
        )


def test_provided_split_raw_and_processed_audits_do_not_claim_review_leakage(tmp_path, monkeypatch):
    raw_root = tmp_path / "raw"
    _write_split_sources(raw_root)
    output_dir = tmp_path / "raw-audit"
    args = build_parser().parse_args(
        [
            "--domain",
            "amazon_ff",
            "--source-mode",
            "provided_split",
            "--raw-root",
            str(raw_root),
            "--output-dir",
            str(output_dir),
        ]
    )
    monkeypatch.setattr(
        "capstone.audit_data.parse_oats_xml",
        lambda _path: (_ for _ in ()).throw(AssertionError("TXT audit must not parse XML")),
    )

    raw_result = run(args)["amazon_ff"]

    assert raw_result["splits"]["train"]["number_of_source_rows"] == 4
    assert raw_result["splits"]["train"]["number_of_quadruples"] == 3
    assert raw_result["splits"]["train"]["invalid_category_count"] == 1
    assert raw_result["review_level_leakage_check"]["status"] == "not_verifiable"
    assert "no_review_leakage" not in raw_result
    assert (output_dir / "amazon_ff.json").is_file()

    prepare_domain(
        "amazon_ff",
        split_mode="provided_split",
        raw_root=raw_root,
        output_root=tmp_path / "processed",
    )
    processed_result = audit_processed(tmp_path / "processed" / "amazon_ff")

    assert processed_result["splits"]["train"]["number_of_source_rows"] == 4
    assert processed_result["splits"]["train"]["number_of_quadruples"] == 2
    assert processed_result["review_level_leakage_check"]["status"] == "not_verifiable"
    assert "review_overlap" not in processed_result
    assert "no_review_leakage" not in processed_result


def test_provided_split_metadata_keeps_source_paths_portable_and_counts_issues(tmp_path, monkeypatch):
    """Processed data must be usable from another machine, and per-split issue counters
    must reflect the rows cleaning dropped (POLARITY#* in the fixture)."""
    raw_root = tmp_path / "raw"
    _write_split_sources(raw_root)
    processed_root = tmp_path / "processed"
    monkeypatch.chdir(tmp_path)

    prepare_domain(
        "amazon_ff",
        split_mode="provided_split",
        seed=None,
        sentiment_mode="paper_compatible",
        raw_root=Path("raw"),
        output_root=processed_root,
    )
    metadata = json.loads((processed_root / "amazon_ff" / "metadata.json").read_text(encoding="utf-8"))

    for name, source_path in metadata["source_paths"].items():
        assert not Path(source_path).is_absolute(), f"{name} must stay relative: {source_path}"
    train_stats = metadata["split_statistics"]["train"]
    assert train_stats["invalid_category_count"] == 1
    assert train_stats["issue_counts"].get("invalid_aspect_category") == 1
    assert metadata["split_statistics"]["dev"]["invalid_category_count"] == 0

    # The recorded relative paths still validate from the same working directory.
    validate_prepared_metadata(
        processed_root / "amazon_ff",
        domain="amazon_ff",
        split_mode="provided_split",
        sentiment_mode="paper_compatible",
        dataset_version="oats_provided_txt_v1",
        cleaning_configuration=metadata["cleaning_configuration"],
    )


def test_audit_data_defaults_to_xml_source_mode():
    """The CLI default must not silently switch an existing XML audit command to TXT."""
    assert build_parser().parse_args(["--domain", "amazon_ff"]).source_mode == "xml"
