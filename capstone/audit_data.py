"""Audit supplied OATS TXT splits by default, or XML with an explicit source mode."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

from capstone.data.audit import audit_reviews, audit_sentence_rows
from capstone.data.cleaning import clean_dataset, clean_split_rows
from capstone.data.manifest import load_unannotated_review_manifest
from capstone.data.parser import (
    DataIssue,
    load_original_splits,
    parse_oats_xml,
    parse_original_split_file,
)
from capstone.data.paths import DOMAINS, resolve_domain_paths
from capstone.data.serialization import read_jsonl, sha256_file
from capstone.types import ReviewRecord, SentenceRecord

LOGGER = logging.getLogger("capstone.audit_data")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=DOMAINS, default=None)
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
        "--source-mode",
        choices=("provided_split", "xml"),
        default="xml",
        help="audit the XML corpus by default; use provided_split to audit the supplied TXT splits",
    )
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
    """Inspect prepared JSONL counts and report review overlap only when IDs exist."""

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
    rows_by_split: dict[str, list[SentenceRecord]] = {}
    for split_name in ("train", "dev", "test"):
        rows_by_split[split_name] = read_jsonl(root / f"{split_name}.jsonl")

    row_mode = metadata.get("split_mode") == "provided_split" or any(
        row.review_id is None and row.source_line_number is not None
        for rows in rows_by_split.values()
        for row in rows
    )
    if row_mode:
        if any(
            row.review_id is not None or row.sentence_id is not None
            for rows in rows_by_split.values()
            for row in rows
        ):
            raise ValueError("provided split JSONL must not include review_id or sentence_id")
        for split_name, rows in rows_by_split.items():
            line_numbers = [row.source_line_number for row in rows]
            if len(line_numbers) != len(set(line_numbers)):
                raise ValueError(f"{split_name} JSONL contains duplicate source_line_number values")
        return {
            "processed_directory": str(root),
            "splits": {
                split_name: audit_sentence_rows(rows, split=split_name)
                for split_name, rows in rows_by_split.items()
            },
            "review_level_leakage_check": metadata.get(
                "review_level_leakage_check",
                {
                    "status": "not_verifiable",
                    "reason": "Processed TXT rows do not include review IDs.",
                },
            ),
            "aspect_inventory": metadata.get("aspect_inventory", []),
            "split_fingerprints": metadata.get("split_fingerprints", {}),
            "source_fingerprints": metadata.get("source_fingerprints", {}),
        }

    splits: dict[str, list[ReviewRecord]] = {}
    for split_name, records in rows_by_split.items():
        grouped: dict[str, list] = defaultdict(list)
        for record in records:
            if record.review_id is None or record.sentence_id is None:
                raise ValueError("review-linked processed JSONL must include review_id and sentence_id")
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


def audit_provided_split_domain(
    domain: str,
    *,
    raw_root: Path | None = None,
    sentiment_mode: str = "raw_4_class",
) -> dict[str, object]:
    """Audit TXT source rows and their cleaned training form without opening XML."""
    paths = resolve_domain_paths(domain, raw_root, prefer_corrected_hotels=False)
    source_paths = {
        f"provided_{split_name}_txt": str(path.resolve())
        for split_name, path in paths.split_files.items()
    }
    source_fingerprints = {
        name: sha256_file(path)
        for name, path in source_paths.items()
    }
    raw_splits: dict[str, list[SentenceRecord]] = {}
    cleaned_splits: dict[str, list[SentenceRecord]] = {}
    issues_by_split: dict[str, list[DataIssue]] = {}
    issues: list[DataIssue] = []
    for split_name, path in paths.split_files.items():
        rows = [
            SentenceRecord(
                review_id=None,
                sentence_id=None,
                text=row.text,
                quadruples=row.quadruples,
                source_line_number=row.line_number,
            )
            for row in parse_original_split_file(path)
        ]
        cleaned = clean_split_rows(
            rows,
            sentiment_mode=sentiment_mode,
            source_path=str(path.resolve()),
        )
        raw_splits[split_name] = rows
        cleaned_splits[split_name] = cleaned.rows
        issues_by_split[split_name] = cleaned.issues
        issues.extend(cleaned.issues)

    raw_rows = [row for rows in raw_splits.values() for row in rows]
    cleaned_rows = [row for rows in cleaned_splits.values() for row in rows]
    return {
        "domain": domain,
        "source_type": "provided_txt_splits",
        "sentiment_mode": sentiment_mode,
        "source_paths": source_paths,
        "source_fingerprints": source_fingerprints,
        "overall": audit_sentence_rows(raw_rows, issues=issues),
        "splits": {
            split_name: audit_sentence_rows(
                rows,
                issues=issues_by_split[split_name],
                split=split_name,
            )
            for split_name, rows in raw_splits.items()
        },
        "cleaned_splits": {
            split_name: audit_sentence_rows(rows, split=split_name)
            for split_name, rows in cleaned_splits.items()
        },
        "issue_counts": dict(sorted(Counter(issue.reason for issue in issues).items())),
        "issues": [issue.to_dict() for issue in issues],
        "review_level_leakage_check": {
            "status": "not_verifiable",
            "reason": "TXT source rows do not include review IDs; review-level overlap cannot be checked.",
            "source_split_attestation": "User confirms the provided train/dev/test split assignment is correct.",
        },
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
    if args.source_mode == "provided_split" and args.unannotated_manifest is not None:
        raise ValueError("--unannotated-manifest cannot be used for TXT splits without review IDs")
    if not args.all_domains and args.domain is None:
        raise ValueError("raw audit requires --domain or --all-domains")
    domains = list(DOMAINS) if args.all_domains else [args.domain]
    if args.source_mode == "provided_split":
        results = {
            domain: audit_provided_split_domain(
                domain,
                raw_root=args.raw_root,
                sentiment_mode=args.sentiment_mode,
            )
            for domain in domains
        }
    else:
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
