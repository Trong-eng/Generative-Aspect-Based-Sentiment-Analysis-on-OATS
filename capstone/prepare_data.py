"""Prepare review-linked OATS sentence records for the ASQP pipeline."""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from capstone.data.audit import audit_reviews
from capstone.data.cleaning import clean_dataset
from capstone.data.inventory import collect_aspect_inventory
from capstone.data.manifest import load_unannotated_review_manifest
from capstone.data.parser import DataIssue, load_original_splits, parse_oats_xml
from capstone.data.paths import DOMAINS, resolve_domain_paths
from capstone.data.serialization import sha256_file, write_prepared_splits
from capstone.data.splitting import assert_no_review_leakage, split_reviews
from capstone.config import load_config


DATASET_VERSION = "oats_xml_v1"

LOGGER = logging.getLogger("capstone.prepare_data")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="optional experiment YAML config")
    parser.add_argument("--domain", choices=DOMAINS, default=None)
    parser.add_argument(
        "--split-mode",
        choices=("original_split", "clean_split"),
        default=None,
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--sentiment-mode",
        choices=("raw_4_class", "paper_compatible"),
        default=None,
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help="directory containing amazon_ff, coursera, and hotels folders",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--unannotated-manifest",
        type=Path,
        default=None,
        help="explicit JSON evidence manifest; only listed confirmed-unannotated review IDs are removed",
    )
    return parser


def _issue_summary(issues: Sequence[DataIssue]) -> dict[str, int]:
    return dict(sorted(Counter(issue.reason for issue in issues).items()))


def prepare_domain(
    domain: str,
    *,
    split_mode: str = "clean_split",
    seed: int = 42,
    sentiment_mode: str = "paper_compatible",
    raw_root: Path | None = None,
    output_root: Path = Path("data/processed"),
    unannotated_manifest_path: Path | None = None,
) -> dict[str, Path]:
    if split_mode not in {"original_split", "clean_split"}:
        raise ValueError("split_mode must be 'original_split' or 'clean_split'")
    paths = resolve_domain_paths(
        domain,
        raw_root,
        prefer_corrected_hotels=(split_mode == "clean_split"),
    )
    parsed = parse_oats_xml(paths.xml_path)
    manifest = (
        load_unannotated_review_manifest(unannotated_manifest_path, expected_domain=domain)
        if unannotated_manifest_path is not None
        else None
    )
    if manifest is not None:
        manifest.validate_source_review_ids(review.review_id for review in parsed.reviews)
    confirmed_ids = manifest.confirmed_review_ids if manifest else frozenset()
    legitimate_ids = manifest.legitimate_zero_opinion_review_ids if manifest else frozenset()
    manifest_args: dict[str, Any] = {
        "confirmed_unannotated_review_ids": confirmed_ids,
        "legitimate_zero_opinion_review_ids": legitimate_ids,
        "manifest_provenance": manifest.provenance if manifest else None,
        "manifest_path": str(manifest.path) if manifest else None,
        "manifest_sha256": manifest.sha256 if manifest else None,
    }

    if split_mode == "clean_split":
        cleaned = clean_dataset(
            parsed.reviews,
            sentiment_mode=sentiment_mode,
            source_issues=parsed.issues,
            **manifest_args,
        )
        annotation_status_by_review_id = cleaned.annotation_status_by_review_id
        all_retained_reviews = cleaned.reviews
        splits = split_reviews(cleaned.reviews, seed=seed)
        assert_no_review_leakage(splits)
        issues = cleaned.issues
        excluded_reviews = cleaned.excluded_reviews
    else:
        original_splits = load_original_splits(
            parsed.reviews,
            paths.split_files,
            sentiment_mode=sentiment_mode,
        )
        issues = list(parsed.issues)
        splits = {}
        annotation_status_by_review_id: dict[str, str] = {}
        all_retained_reviews = []
        excluded_reviews = []
        for split_name, reviews_in_split in original_splits.items():
            review_ids = {review.review_id for review in reviews_in_split}
            relevant_source_issues = [
                issue
                for issue in parsed.issues
                if issue.review_id is None or issue.review_id in review_ids
            ]
            cleaned_split = clean_dataset(
                reviews_in_split,
                sentiment_mode=sentiment_mode,
                source_issues=relevant_source_issues,
                **manifest_args,
            )
            splits[split_name] = cleaned_split.reviews
            annotation_status_by_review_id.update(cleaned_split.annotation_status_by_review_id)
            all_retained_reviews.extend(cleaned_split.reviews)
            excluded_reviews.extend(cleaned_split.excluded_reviews)
            issues.extend(cleaned_split.issues[len(relevant_source_issues) :])

    split_stats = {
        split_name: audit_reviews(
            reviews,
            split=split_name,
            review_status_by_id={
                review_id: annotation_status_by_review_id[review_id]
                for review_id in {review.review_id for review in reviews}
                if review_id in annotation_status_by_review_id
            },
        )
        for split_name, reviews in splits.items()
    }
    zero_statuses = ("confirmed_unannotated", "legitimate_zero_opinion", "unknown")
    zero_status_counts = {
        status: sum(value == status for value in annotation_status_by_review_id.values())
        for status in zero_statuses
    }
    source_paths = {"raw_xml": str(paths.xml_path)}
    source_fingerprints = {"raw_xml": sha256_file(paths.xml_path)}
    if split_mode == "original_split":
        for split_name, split_path in paths.split_files.items():
            fingerprint_name = f"original_{split_name}_txt"
            source_paths[fingerprint_name] = str(split_path)
            source_fingerprints[fingerprint_name] = sha256_file(split_path)
    if manifest is not None:
        source_paths["annotation_manifest"] = str(manifest.path)
        source_fingerprints["annotation_manifest"] = manifest.sha256
    manifest_metadata = (
        manifest.to_metadata(
            source_review_ids=(review.review_id for review in parsed.reviews),
            excluded_review_ids=(
                excluded["review_id"] for excluded in excluded_reviews
            ),
        )
        if manifest is not None
        else None
    )
    metadata = {
        "domain": domain,
        "split_mode": split_mode,
        "seed": seed if split_mode == "clean_split" else None,
        "sentiment_mode": sentiment_mode,
        "dataset_version": DATASET_VERSION,
        "cleaning_configuration": {
            "version": "review_level_clean_v1",
            "remove_unannotated_reviews": manifest is not None,
            "unannotated_reviews_manifest": str(manifest.path) if manifest else None,
            "invalid_aspect_policy": "drop_and_log",
            "conflict_policy": "preserve",
        },
        "annotation_status_policy": {
            "zero_quadruple_reviews_without_manifest": "unknown",
            "unknown_reviews": "retain",
            "only_confirmed_unannotated_manifest_ids_are_excluded": True,
            "manifest": manifest_metadata,
            "status_counts": zero_status_counts,
            "status_by_review_id": dict(sorted(annotation_status_by_review_id.items())),
        },
        "raw_xml": str(paths.xml_path),
        "original_split_files": {
            name: str(path) for name, path in paths.split_files.items()
        },
        "aspect_inventory": collect_aspect_inventory(all_retained_reviews),
        "fingerprint_algorithm": "sha256",
        "source_paths": source_paths,
        "source_fingerprints": source_fingerprints,
        "split_statistics": split_stats,
        "review_ids_by_split": {
            name: sorted({review.review_id for review in reviews})
            for name, reviews in splits.items()
        },
        "excluded_reviews": excluded_reviews,
        "issue_counts": _issue_summary(issues),
        "issues": [issue.to_dict() for issue in issues],
        "review_level_leakage_checked": split_mode == "clean_split",
    }
    destination = output_root / domain
    raw_data_root = paths.domain_dir.parent.parent.resolve()
    resolved_destination = destination.resolve()
    try:
        resolved_destination.relative_to(raw_data_root)
    except ValueError:
        pass
    else:
        raise ValueError(
            f"processed output directory {resolved_destination} overlaps raw OATS data "
            f"at {raw_data_root}"
        )
    output_paths = write_prepared_splits(destination, splits, metadata)
    LOGGER.info("prepared output paths for %s: %s", domain, output_paths)
    return output_paths


def run(args: argparse.Namespace) -> dict[str, Path]:
    config = load_config(args.config) if args.config is not None else {}
    experiment = config.get("experiment", {})
    dataset = config.get("dataset", {})
    configured_paths = config.get("paths", {})
    cleaning = config.get("cleaning", {})

    def choose(cli_value: Any, config_value: Any, default: Any) -> Any:
        return cli_value if cli_value is not None else config_value if config_value is not None else default

    manifest_value = args.unannotated_manifest
    if manifest_value is None:
        manifest_value = cleaning.get("unannotated_reviews_manifest")
    remove_unannotated = bool(cleaning.get("remove_unannotated_reviews", False))
    if args.unannotated_manifest is not None:
        remove_unannotated = True
    if remove_unannotated and manifest_value is None:
        raise ValueError("unannotated review removal requires --unannotated-manifest or a configured manifest")
    if manifest_value is not None and not remove_unannotated:
        raise ValueError("a manifest path requires cleaning.remove_unannotated_reviews=true")

    domain = choose(args.domain, experiment.get("domain"), None)
    if domain is None:
        raise ValueError("domain is required: pass --domain or provide experiment.domain in --config")

    return prepare_domain(
        domain,
        split_mode=choose(args.split_mode, experiment.get("split_mode"), "clean_split"),
        seed=choose(args.seed, experiment.get("seed"), 42),
        sentiment_mode=choose(args.sentiment_mode, dataset.get("sentiment_mode"), "paper_compatible"),
        raw_root=choose(args.raw_root, configured_paths.get("raw_data"), None),
        output_root=choose(args.output_root, configured_paths.get("processed_data"), Path("data/processed")),
        unannotated_manifest_path=Path(manifest_value) if manifest_value is not None else None,
    )


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    arguments = build_parser().parse_args(argv)
    run(arguments)


if __name__ == "__main__":
    main()
