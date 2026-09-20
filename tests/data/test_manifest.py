import json

import pytest

from capstone.data.manifest import load_unannotated_review_manifest


def test_manifest_requires_domain_provenance_and_ids_and_fingerprints_source(tmp_path):
    manifest_path = tmp_path / "unannotated.json"
    manifest_path.write_text(
        json.dumps(
            {
                "domain": "hotels",
                "provenance": "Manually checked against annotation records.",
                "confirmed_unannotated_review_ids": ["review-1", "review-2"],
                "legitimate_zero_opinion_review_ids": ["review-3"],
            }
        ),
        encoding="utf-8",
    )

    manifest = load_unannotated_review_manifest(manifest_path, expected_domain="hotels")

    assert manifest.confirmed_review_ids == frozenset({"review-1", "review-2"})
    assert manifest.legitimate_zero_opinion_review_ids == frozenset({"review-3"})
    assert manifest.provenance == "Manually checked against annotation records."
    assert len(manifest.sha256) == 64
    metadata = manifest.to_metadata(
        source_review_ids={"review-1", "review-3"},
        excluded_review_ids={"review-1"},
    )
    assert metadata["matched_review_ids"] == ["review-1", "review-3"]
    assert metadata["unmatched_review_ids"] == ["review-2"]
    assert metadata["excluded_review_ids"] == ["review-1"]
    with pytest.raises(ValueError, match="absent from the selected source dataset"):
        manifest.validate_source_review_ids(["review-1", "review-3"])


def test_manifest_rejects_ids_marked_as_both_confirmed_and_legitimate(tmp_path):
    manifest_path = tmp_path / "unannotated.json"
    manifest_path.write_text(
        json.dumps(
            {
                "domain": "hotels",
                "provenance": "Manual review.",
                "confirmed_unannotated_review_ids": ["r1"],
                "legitimate_zero_opinion_review_ids": ["r1"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="both confirmed and legitimate"):
        load_unannotated_review_manifest(manifest_path, expected_domain="hotels")


def test_manifest_rejects_domain_mismatch_or_missing_provenance(tmp_path):
    manifest_path = tmp_path / "unannotated.json"
    manifest_path.write_text(
        json.dumps(
            {
                "domain": "coursera",
                "provenance": "Manual review.",
                "confirmed_unannotated_review_ids": ["r1"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="domain"):
        load_unannotated_review_manifest(manifest_path, expected_domain="hotels")

    manifest_path.write_text(
        json.dumps({"domain": "hotels", "confirmed_unannotated_review_ids": ["r1"]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="provenance"):
        load_unannotated_review_manifest(manifest_path, expected_domain="hotels")
