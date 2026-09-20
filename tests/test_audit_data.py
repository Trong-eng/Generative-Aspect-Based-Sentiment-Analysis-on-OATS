import json
from pathlib import Path

import pytest

from capstone.audit_data import audit_domain, build_parser, run
from capstone.data.parser import ParseResult
from capstone.data.paths import DomainPaths
from capstone.types import Quadruple, ReviewRecord, SentenceRecord


def test_original_split_audit_uses_statuses_from_original_xml(tmp_path: Path, monkeypatch):
    domain_dir = tmp_path / "hotels" / "quads"
    corrected_xml = domain_dir / "Hotels_corr.xml"
    original_xml = domain_dir / "Hotels.xml"
    split_files = {name: domain_dir / f"{name}.txt" for name in ("train", "dev", "test")}
    corrected_paths = DomainPaths("hotels", domain_dir, corrected_xml, split_files)
    original_paths = DomainPaths("hotels", domain_dir, original_xml, split_files)

    corrected_review = ReviewRecord(
        "corrected-only",
        [SentenceRecord("corrected-only", "corrected-only:0", "A labeled review.", [
            Quadruple("service", "hotel service", "helpful", "positive")
        ])],
    )
    original_review = ReviewRecord(
        "original-only",
        [SentenceRecord("original-only", "original-only:0", "An empty-label review.", [])],
    )

    def resolve(_domain, _raw_root=None, *, prefer_corrected_hotels=True):
        return corrected_paths if prefer_corrected_hotels else original_paths

    def parse(path):
        return ParseResult([corrected_review] if Path(path) == corrected_xml else [original_review])

    monkeypatch.setattr("capstone.audit_data.resolve_domain_paths", resolve)
    monkeypatch.setattr("capstone.audit_data.parse_oats_xml", parse)
    monkeypatch.setattr(
        "capstone.audit_data.load_original_splits",
        lambda _reviews, _split_files, sentiment_mode: {
            "train": [original_review],
            "dev": [],
            "test": [],
        },
    )
    manifest_path = tmp_path / "statuses.json"
    manifest_path.write_text(
        json.dumps({
            "domain": "hotels",
            "provenance": "manual annotation review",
            "confirmed_unannotated_review_ids": ["original-only"],
        }),
        encoding="utf-8",
    )

    result = audit_domain("hotels", unannotated_manifest_path=manifest_path)

    assert result["splits"]["train"]["zero_label_review_status_counts"] == {
        "confirmed_unannotated": 1,
        "legitimate_zero_opinion": 0,
        "unknown": 0,
    }


def test_raw_audit_requires_a_domain_or_all_domains():
    args = build_parser().parse_args([])

    with pytest.raises(ValueError, match="--domain or --all-domains"):
        run(args)
