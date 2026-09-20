"""Audit raw OATS XML and the repository's original sentence splits."""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from capstone.data.audit import audit_reviews
from capstone.data.cleaning import clean_dataset
from capstone.data.manifest import load_unannotated_review_manifest
from capstone.data.parser import DataIssue, load_original_splits, parse_oats_xml
from capstone.data.paths import DOMAINS, resolve_domain_paths
from capstone.data.serialization import read_jsonl
from capstone.types import ReviewRecord

LOGGER = logging.getLogger("capstone.audit_data")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=DOMAINS, default="hotels")
    parser.add_argument("--all-domains", action="store_true")
    parser.add_argument(
        "--processed",
        type=Path,
        default=None,
        help="inspect processed train/dev/test JSONL files in this directory",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help="directory containing amazon_ff, coursera, and hotels folders",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/data_audit"))
    parser.add_argument("--sentiment-mode", choices=("raw_4_class", "paper_compatible"), default="raw_4_class")
    parser.add_argument(
        "--unannotated-manifest",
        type=Path,
        default=None,
        help="optional evidence manifest for classifying zero-label reviews",
    )
    return parser


def _issues_for_reviews(issues: Sequence[DataIssue], review_ids: set[str]) -> list[DataIssue]:
    return [issue for issue in issues if issue.review_id is None or issue.review_id in review_ids]


def audit_processed(directory: str | Path) -> dict[str, object]:
    """Inspect prepared split JSONL counts and report any review ID overlap."""

    root = Path(directory)
    metadata: dict[str, object] = {}
    metadata_path = root / "metadata.json"
    if metadata_path.is_file():
        try:
            loaded_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"could not read prepared metadata at {metadata_path}: {error}") from error
        if isinstance(loaded_metadata, dict):
            metadata = loaded_metadata
    status_policy = metadata.get("annotation_status_policy", {})
    status_by_id = (
        status_policy.get("status_by_review_id", {})
        if isinstance(status_policy, dict)
        else {}
    )
    if not isinstance(status_by_id, dict):
        status_by_id = {}
    splits: dict[str, list[ReviewRecord]] = {}
    for split_name in ("train", "dev", "test"):
        records = read_jsonl(root / f"{split_name}.jsonl")
        grouped: dict[str, list] = defaultdict(list)
        for record in records:
            grouped[record.review_id].append(record)
        splits[split_name] = [
            ReviewRecord(review_id, sentences)
            for review_id, sentences in grouped.items()
        ]

    ids = {
        split_name: {review.review_id for review in reviews}
        for split_name, reviews in splits.items()
    }
    overlap = {
        "train_dev": sorted(ids["train"] & ids["dev"]),
        "train_test": sorted(ids["train"] & ids["test"]),
        "dev_test": sorted(ids["dev"] & ids["test"]),
    }
    return {
        "processed_directory": str(root),
        "splits": {
            split_name: audit_reviews(
                reviews,
                split=split_name,
                review_status_by_id={
                    review.review_id: status_by_id[review.review_id]
                    for review in reviews
                    if review.review_id in status_by_id
                },
            )
            for split_name, reviews in splits.items()
        },
        "review_overlap": overlap,
        "no_review_leakage": not any(overlap.values()),
        "aspect_inventory": metadata.get("aspect_inventory", []),
        "split_fingerprints": metadata.get("split_fingerprints", {}),
        "source_fingerprints": metadata.get("source_fingerprints", {}),
        "zero_label_status_policy": status_policy,
    }


def audit_domain(
    domain: str,
    *,
    raw_root: Path | None = None,
    sentiment_mode: str = "raw_4_class",
    unannotated_manifest_path: Path | None = None,
) -> dict[str, object]:
    paths = resolve_domain_paths(domain, raw_root)
    original_split_paths = resolve_domain_paths(
        domain,
        raw_root,
        prefer_corrected_hotels=False,
    )
    parsed = parse_oats_xml(paths.xml_path)
    manifest = (
        load_unannotated_review_manifest(unannotated_manifest_path, expected_domain=domain)
        if unannotated_manifest_path is not None
        else None
    )
    cleaned = clean_dataset(
        parsed.reviews,
        sentiment_mode=sentiment_mode,
        source_issues=parsed.issues,
        confirmed_unannotated_review_ids=(manifest.confirmed_review_ids if manifest else ()),
        legitimate_zero_opinion_review_ids=(manifest.legitimate_zero_opinion_review_ids if manifest else ()),
        manifest_provenance=manifest.provenance if manifest else None,
        manifest_path=str(manifest.path) if manifest else None,
        manifest_sha256=manifest.sha256 if manifest else None,
    )
    split_source = (
        parsed
        if original_split_paths.xml_path == paths.xml_path
        else parse_oats_xml(original_split_paths.xml_path)
    )
    split_cleaned = (
        cleaned
        if split_source is parsed
        else clean_dataset(
            split_source.reviews,
            sentiment_mode=sentiment_mode,
            source_issues=split_source.issues,
            confirmed_unannotated_review_ids=(manifest.confirmed_review_ids if manifest else ()),
            legitimate_zero_opinion_review_ids=(
                manifest.legitimate_zero_opinion_review_ids if manifest else ()
            ),
            manifest_provenance=manifest.provenance if manifest else None,
            manifest_path=str(manifest.path) if manifest else None,
            manifest_sha256=manifest.sha256 if manifest else None,
        )
    )
    original_splits = load_original_splits(
        split_source.reviews,
        paths.split_files,
        sentiment_mode=sentiment_mode,
    )
    split_stats: dict[str, object] = {}
    for split_name, split_reviews in original_splits.items():
        review_ids = {review.review_id for review in split_reviews}
        split_stats[split_name] = audit_reviews(
            split_reviews,
            issues=_issues_for_reviews(split_cleaned.issues, review_ids),
            split=split_name,
            review_status_by_id={
                review_id: split_cleaned.annotation_status_by_review_id[review_id]
                for review_id in review_ids
                if review_id in split_cleaned.annotation_status_by_review_id
            },
        )
    return {
        "domain": domain,
        "xml_source": str(paths.xml_path),
        "original_split_xml_source": str(original_split_paths.xml_path),
        "sentiment_mode": sentiment_mode,
        "overall": audit_reviews(
            parsed.reviews,
            issues=cleaned.issues,
            review_status_by_id=cleaned.annotation_status_by_review_id,
        ),
        "splits": split_stats,
        "excluded_reviews": cleaned.excluded_reviews,
        "annotation_status_manifest": manifest.to_metadata(
            source_review_ids=(review.review_id for review in parsed.reviews),
            excluded_review_ids=(record["review_id"] for record in cleaned.excluded_reviews),
        ) if manifest else None,
        "issues": [issue.to_dict() for issue in cleaned.issues],
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.processed is not None:
        if args.unannotated_manifest is not None:
            raise ValueError("--unannotated-manifest only applies to raw-domain audits")
        if args.all_domains:
            raise ValueError("--processed cannot be combined with --all-domains")
        result = audit_processed(args.processed)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        destination = args.output_dir / "processed.json"
        with destination.open("w", encoding="utf-8", newline="\n") as output:
            json.dump(result, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
        summary_path = args.output_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8", newline="\n") as output:
            json.dump(result["splits"], output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
        LOGGER.info("wrote processed-data inspection to %s", destination)
        return {"processed": result}

    if args.all_domains and args.unannotated_manifest is not None:
        raise ValueError("--unannotated-manifest requires auditing one domain")
    domains = list(DOMAINS) if args.all_domains else [args.domain]
    results = {
        domain: audit_domain(
            domain,
            raw_root=args.raw_root,
            sentiment_mode=args.sentiment_mode,
            unannotated_manifest_path=args.unannotated_manifest,
        )
        for domain in domains
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for domain, result in results.items():
        destination = args.output_dir / f"{domain}.json"
        with destination.open("w", encoding="utf-8", newline="\n") as output:
            json.dump(result, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
        LOGGER.info("wrote %s", destination)

    summary = {
        domain: result["overall"]
        for domain, result in results.items()
    }
    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
    LOGGER.info("wrote %s", summary_path)
    return results


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    arguments = build_parser().parse_args(argv)
    run(arguments)


if __name__ == "__main__":
    main()
