import json
import hashlib

import pytest

from capstone.data.audit import audit_reviews
from capstone.data.serialization import (
    read_jsonl,
    validate_prepared_metadata,
    write_prepared_splits,
    write_jsonl,
)
from capstone.types import Quadruple, ReviewRecord, SentenceRecord


def test_audit_counts_empty_labels_implicit_fields_and_distributions():
    review = ReviewRecord(
        "r1",
        [
            SentenceRecord(
                "r1",
                "r1:0",
                "Great course.",
                [Quadruple("NULL", "course quality", "Great", "positive")],
            ),
            SentenceRecord("r1", "r1:1", "No label.", []),
        ],
    )

    stats = audit_reviews([review], split="train")

    assert stats["split"] == "train"
    assert stats["number_of_reviews"] == 1
    assert stats["number_of_sentences"] == 2
    assert stats["sentences_with_quadruples"] == 1
    assert stats["empty_label_sentences"] == 1
    assert stats["number_of_quadruples"] == 1
    assert stats["implicit_target_count"] == 1
    assert stats["explicit_opinion_count"] == 1
    assert stats["aspect_distribution"] == {"course quality": 1}
    assert stats["sentiment_distribution"] == {"positive": 1}
    assert stats["zero_label_review_status_counts"] == {
        "confirmed_unannotated": 0,
        "legitimate_zero_opinion": 0,
        "unknown": 0,
    }
    assert stats["zero_label_sentence_status_counts"]["legitimate_zero_opinion"] == 1


def test_audit_distinguishes_confirmed_legitimate_and_unknown_zero_label_reviews():
    reviews = [
        ReviewRecord("confirmed", [SentenceRecord("confirmed", "c:0", "C.", [])]),
        ReviewRecord("legitimate", [SentenceRecord("legitimate", "l:0", "L.", [])]),
        ReviewRecord("unknown", [SentenceRecord("unknown", "u:0", "U.", [])]),
    ]

    stats = audit_reviews(
        reviews,
        confirmed_unannotated_review_ids={"confirmed"},
        legitimate_zero_opinion_review_ids={"legitimate"},
    )

    assert stats["zero_label_review_status_counts"] == {
        "confirmed_unannotated": 1,
        "legitimate_zero_opinion": 1,
        "unknown": 1,
    }


def test_jsonl_round_trip_and_preparation_metadata(tmp_path):
    review = ReviewRecord(
        "r1",
        [SentenceRecord("r1", "r1:0", "Great course.", [])],
    )
    records = list(read_jsonl(write_jsonl(tmp_path / "sample.jsonl", review.sentences)))

    assert records == review.sentences

    paths = write_prepared_splits(
        tmp_path / "prepared",
        {"train": [review], "dev": [], "test": []},
        {
            "domain": "coursera",
            "split_mode": "clean_split",
            "aspect_inventory": ["course quality"],
            "source_paths": {},
            "source_fingerprints": {"raw_xml": "a" * 64},
        },
    )

    assert {name for name in paths} == {"train", "dev", "test", "metadata"}
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert metadata["domain"] == "coursera"
    assert metadata["aspect_inventory"] == ["course quality"]
    assert metadata["source_fingerprints"] == {"raw_xml": "a" * 64}
    for split_name in ("train", "dev", "test"):
        actual = hashlib.sha256(paths[split_name].read_bytes()).hexdigest()
        assert metadata["split_fingerprints"][split_name] == actual
    row = next(read_jsonl(paths["train"]))
    assert row.review_id == "r1"


def test_prepared_metadata_must_match_experiment_configuration(tmp_path):
    source_path = tmp_path / "source.xml"
    source_path.write_text("immutable source identity", encoding="utf-8")
    source_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    metadata = {
        "domain": "hotels",
        "split_mode": "clean_split",
        "sentiment_mode": "paper_compatible",
        "dataset_version": "oats_xml_v1",
        "seed": 42,
        "cleaning_configuration": {
            "version": "review_level_clean_v1",
            "remove_unannotated_reviews": False,
            "unannotated_reviews_manifest": None,
            "invalid_aspect_policy": "drop_and_log",
            "conflict_policy": "preserve",
        },
        "raw_xml": str(source_path),
        "source_paths": {"raw_xml": str(source_path)},
        "source_fingerprints": {"raw_xml": source_digest},
    }
    prepared = tmp_path / "prepared"
    write_prepared_splits(prepared, {"train": [], "dev": [], "test": []}, metadata)
    metadata = json.loads((prepared / "metadata.json").read_text(encoding="utf-8"))

    loaded = validate_prepared_metadata(
        prepared,
        domain="hotels",
        split_mode="clean_split",
        sentiment_mode="paper_compatible",
        dataset_version="oats_xml_v1",
        seed=42,
        cleaning_configuration=metadata["cleaning_configuration"],
    )

    assert loaded == metadata
    with pytest.raises(ValueError, match="seed"):
        validate_prepared_metadata(
            prepared,
            domain="hotels",
            split_mode="clean_split",
            sentiment_mode="paper_compatible",
            dataset_version="oats_xml_v1",
            seed=43,
        )

    (prepared / "test.jsonl").write_text('{"stale": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        validate_prepared_metadata(
            prepared,
            domain="hotels",
            split_mode="clean_split",
            sentiment_mode="paper_compatible",
            dataset_version="oats_xml_v1",
            seed=42,
            cleaning_configuration=metadata["cleaning_configuration"],
        )

    (prepared / "test.jsonl").write_text("", encoding="utf-8")
    source_path.write_text("changed source identity", encoding="utf-8")
    with pytest.raises(ValueError, match="source SHA-256"):
        validate_prepared_metadata(
            prepared,
            domain="hotels",
            split_mode="clean_split",
            sentiment_mode="paper_compatible",
            dataset_version="oats_xml_v1",
            seed=42,
            cleaning_configuration=metadata["cleaning_configuration"],
        )


def test_prepared_metadata_requires_a_verifiable_path_for_every_source_hash(tmp_path):
    prepared = tmp_path / "prepared"
    write_prepared_splits(
        prepared,
        {"train": [], "dev": [], "test": []},
        {
            "domain": "hotels",
            "split_mode": "clean_split",
            "sentiment_mode": "paper_compatible",
            "dataset_version": "oats_xml_v1",
            "seed": 42,
            "source_paths": {},
            "source_fingerprints": {"raw_xml": "a" * 64},
        },
    )

    with pytest.raises(ValueError, match="source_paths: missing path"):
        validate_prepared_metadata(
            prepared,
            domain="hotels",
            split_mode="clean_split",
            sentiment_mode="paper_compatible",
            dataset_version="oats_xml_v1",
            seed=42,
        )
