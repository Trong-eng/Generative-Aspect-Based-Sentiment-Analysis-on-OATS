import json
from pathlib import Path

import pytest

from capstone.data.parser import ParseResult
from capstone.data.paths import DomainPaths
from capstone.prepare_data import build_parser, prepare_domain, run
from capstone.types import Quadruple, ReviewRecord, SentenceRecord


def test_prepare_cli_accepts_config_and_optional_manifest():
    args = build_parser().parse_args(
        ["--config", "configs/gas_t5_hotels.yaml", "--unannotated-manifest", "status.json"]
    )

    assert args.config == Path("configs/gas_t5_hotels.yaml")
    assert args.unannotated_manifest == Path("status.json")


def test_prepare_cli_requires_an_explicit_domain_when_no_config_is_supplied():
    args = build_parser().parse_args([])

    with pytest.raises(ValueError, match="domain is required"):
        run(args)


def test_prepare_run_forwards_config_and_cli_manifest_override(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    manifest_path = tmp_path / "status.json"
    observed = {}
    config = {
        "experiment": {"domain": "hotels", "split_mode": "clean_split", "seed": 17},
        "dataset": {"sentiment_mode": "raw_4_class"},
        "paths": {"raw_data": tmp_path / "raw", "processed_data": tmp_path / "processed"},
        "cleaning": {"remove_unannotated_reviews": False, "unannotated_reviews_manifest": None},
    }
    monkeypatch.setattr("capstone.prepare_data.load_config", lambda path: config)
    monkeypatch.setattr(
        "capstone.prepare_data.prepare_domain",
        lambda domain, **kwargs: observed.update(domain=domain, **kwargs),
    )
    args = build_parser().parse_args(
        ["--config", str(config_path), "--unannotated-manifest", str(manifest_path)]
    )

    run(args)

    assert observed["domain"] == "hotels"
    assert observed["split_mode"] == "clean_split"
    assert observed["seed"] == 17
    assert observed["sentiment_mode"] == "raw_4_class"
    assert observed["raw_root"] == tmp_path / "raw"
    assert observed["output_root"] == tmp_path / "processed"
    assert observed["unannotated_manifest_path"] == manifest_path


def test_preparation_persists_all_retained_aspects_statuses_and_content_hashes(tmp_path, monkeypatch):
    raw_domain = tmp_path / "raw" / "hotels" / "quads"
    raw_domain.mkdir(parents=True)
    xml_path = raw_domain / "Hotels_corr.xml"
    xml_path.write_bytes(b"fixture xml identity")
    split_paths = {name: raw_domain / f"{name}.txt" for name in ("train", "dev", "test")}
    paths = DomainPaths("hotels", raw_domain, xml_path, split_paths)

    reviews = [
        ReviewRecord("labeled-a", [SentenceRecord("labeled-a", "a:0", "A.", [
            Quadruple("room", "hotel rooms", "clean", "positive")
        ])]),
        ReviewRecord("labeled-b", [SentenceRecord("labeled-b", "b:0", "B.", [
            Quadruple("staff", "hotel service", "kind", "positive")
        ])]),
        ReviewRecord("unknown-zero", [SentenceRecord("unknown-zero", "u:0", "U.", [])]),
        ReviewRecord("confirmed-zero", [SentenceRecord("confirmed-zero", "c:0", "C.", [])]),
    ]
    monkeypatch.setattr(
        "capstone.prepare_data.resolve_domain_paths",
        lambda domain, raw_root=None, **kwargs: paths,
    )
    monkeypatch.setattr(
        "capstone.prepare_data.parse_oats_xml",
        lambda source: ParseResult(reviews, []),
    )
    manifest_path = tmp_path / "hotels-status.json"
    manifest_path.write_text(
        json.dumps(
            {
                "domain": "hotels",
                "provenance": "Reviewed against the annotation export.",
                "confirmed_unannotated_review_ids": ["confirmed-zero"],
                "legitimate_zero_opinion_review_ids": [],
            }
        ),
        encoding="utf-8",
    )

    outputs = prepare_domain(
        "hotels",
        split_mode="clean_split",
        seed=42,
        output_root=tmp_path / "processed",
        unannotated_manifest_path=manifest_path,
    )

    metadata = json.loads(outputs["metadata"].read_text(encoding="utf-8"))
    assert metadata["aspect_inventory"] == ["hotel rooms", "hotel service"]
    assert metadata["annotation_status_policy"]["status_by_review_id"] == {
        "confirmed-zero": "confirmed_unannotated",
        "labeled-a": "annotated",
        "labeled-b": "annotated",
        "unknown-zero": "unknown",
    }
    excluded = metadata["excluded_reviews"][0]
    assert excluded["annotation_status"] == "confirmed_unannotated"
    assert excluded["evidence"] == "Reviewed against the annotation export."
    assert metadata["source_fingerprints"].keys() == {"raw_xml", "annotation_manifest"}
    assert all(len(value) == 64 for value in metadata["source_fingerprints"].values())
    assert set(metadata["split_fingerprints"]) == {"train", "dev", "test"}
    assert "confirmed-zero" not in {
        review_id
        for split_review_ids in metadata["review_ids_by_split"].values()
        for review_id in split_review_ids
    }


def test_preparation_rejects_manifest_ids_missing_from_selected_xml(tmp_path, monkeypatch):
    raw_domain = tmp_path / "raw" / "hotels" / "quads"
    raw_domain.mkdir(parents=True)
    xml_path = raw_domain / "Hotels_corr.xml"
    xml_path.write_bytes(b"fixture xml identity")
    paths = DomainPaths(
        "hotels",
        raw_domain,
        xml_path,
        {name: raw_domain / f"{name}.txt" for name in ("train", "dev", "test")},
    )
    review = ReviewRecord("known-review", [SentenceRecord("known-review", "s1", "Known.", [])])
    monkeypatch.setattr("capstone.prepare_data.resolve_domain_paths", lambda *args, **kwargs: paths)
    monkeypatch.setattr(
        "capstone.prepare_data.parse_oats_xml",
        lambda source: ParseResult([review], []),
    )
    manifest_path = tmp_path / "stale-status.json"
    manifest_path.write_text(
        json.dumps({
            "domain": "hotels",
            "provenance": "Manual review.",
            "confirmed_unannotated_review_ids": ["stale-review"],
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="absent from the selected source dataset"):
        prepare_domain(
            "hotels",
            split_mode="clean_split",
            output_root=tmp_path / "processed",
            unannotated_manifest_path=manifest_path,
        )
