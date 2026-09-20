"""JSONL and metadata serialization for prepared ASQP splits."""

from __future__ import annotations

import json
import hashlib
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from capstone.types import ReviewRecord, SentenceRecord


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file's bytes without interpreting it."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _sentences(records: Iterable[ReviewRecord | SentenceRecord]) -> Iterable[SentenceRecord]:
    for record in records:
        if isinstance(record, ReviewRecord):
            yield from record.sentences
        elif isinstance(record, SentenceRecord):
            yield record
        else:
            raise TypeError(f"cannot serialize record of type {type(record).__name__}")


def write_jsonl(
    path: str | Path,
    records: Iterable[ReviewRecord | SentenceRecord],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as output:
        for sentence in _sentences(records):
            output.write(json.dumps(sentence.to_dict(), ensure_ascii=False) + "\n")
    return destination


def read_jsonl(path: str | Path) -> list[SentenceRecord]:
    source = Path(path)
    records: list[SentenceRecord] = []
    with source.open("r", encoding="utf-8") as content:
        for line_number, line in enumerate(content, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("record must be a JSON object")
                records.append(SentenceRecord.from_dict(value))
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"{source}:{line_number}: invalid sentence JSONL: {error}") from error
    return records


def validate_prepared_metadata(
    directory: str | Path,
    *,
    domain: str,
    split_mode: str,
    sentiment_mode: str,
    dataset_version: str,
    seed: int | None = None,
    cleaning_configuration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail before training/evaluation if prepared data and config disagree."""
    metadata_path = Path(directory) / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read prepared-data metadata at {metadata_path}: {error}") from error
    if not isinstance(metadata, dict):
        raise ValueError(f"Prepared-data metadata must be a JSON object: {metadata_path}")

    expected: dict[str, Any] = {
        "domain": domain,
        "split_mode": split_mode,
        "sentiment_mode": sentiment_mode,
        "dataset_version": dataset_version,
    }
    if split_mode == "clean_split":
        expected["seed"] = seed
    mismatches = [
        f"{key}: prepared={metadata.get(key)!r}, config={value!r}"
        for key, value in expected.items()
        if metadata.get(key) != value
    ]
    if cleaning_configuration is not None:
        actual_cleaning = metadata.get("cleaning_configuration")
        if not isinstance(actual_cleaning, Mapping):
            mismatches.append("cleaning_configuration: missing from prepared-data metadata")
        else:
            for key, value in cleaning_configuration.items():
                if actual_cleaning.get(key) != value:
                    mismatches.append(
                        f"cleaning_configuration.{key}: "
                        f"prepared={actual_cleaning.get(key)!r}, config={value!r}"
                    )
    prepared_fingerprints = metadata.get("split_fingerprints")
    if not isinstance(prepared_fingerprints, Mapping):
        mismatches.append("split_fingerprints: missing from prepared-data metadata")
    else:
        for split_name in ("train", "dev", "test"):
            expected_digest = prepared_fingerprints.get(split_name)
            split_path = Path(directory) / f"{split_name}.jsonl"
            if not isinstance(expected_digest, str) or not expected_digest:
                mismatches.append(
                    f"split_fingerprints.{split_name}: missing SHA-256 digest"
                )
                continue
            try:
                actual_digest = sha256_file(split_path)
            except OSError as error:
                mismatches.append(
                    f"split_fingerprints.{split_name}: cannot read {split_path}: {error}"
                )
                continue
            if actual_digest != expected_digest:
                mismatches.append(
                    f"split_fingerprints.{split_name}: SHA-256 fingerprint mismatch"
                )
    source_fingerprints = metadata.get("source_fingerprints")
    if not isinstance(source_fingerprints, Mapping) or not source_fingerprints:
        mismatches.append("source_fingerprints: missing or empty in prepared-data metadata")
    else:
        source_paths = metadata.get("source_paths")
        if not isinstance(source_paths, Mapping):
            mismatches.append("source_paths: missing or invalid in prepared-data metadata")
            source_paths = {}
        fingerprint_names = {str(name) for name in source_fingerprints}
        path_names = {str(name) for name in source_paths}
        missing_paths = sorted(fingerprint_names - path_names)
        missing_fingerprints = sorted(path_names - fingerprint_names)
        if missing_paths:
            mismatches.append(
                "source_paths: missing path(s) for source_fingerprints "
                + ", ".join(missing_paths)
            )
        if missing_fingerprints:
            mismatches.append(
                "source_fingerprints: missing digest(s) for source_paths "
                + ", ".join(missing_fingerprints)
            )
        for source_name, expected_digest in source_fingerprints.items():
            if not isinstance(source_name, str) or not source_name:
                mismatches.append("source_fingerprints: source names must be non-empty strings")
                continue
            if not isinstance(expected_digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_digest):
                mismatches.append(
                    f"source_fingerprints.{source_name}: expected a 64-character SHA-256 digest"
                )
                continue
            source_path = source_paths.get(source_name)
            if source_path is None:
                continue
            if not isinstance(source_path, str) or not source_path.strip():
                mismatches.append(f"source_paths.{source_name}: expected a non-empty path")
                continue
            try:
                actual_source_digest = sha256_file(source_path)
            except (OSError, TypeError, ValueError) as error:
                mismatches.append(
                    f"source_fingerprints.{source_name}: cannot read source {source_path}: {error}"
                )
                continue
            if actual_source_digest != expected_digest:
                mismatches.append(
                    f"source_fingerprints.{source_name}: source SHA-256 fingerprint mismatch"
                )
    if mismatches:
        raise ValueError(
            "Prepared data does not match the experiment configuration:\n- "
            + "\n- ".join(mismatches)
            + f"\nRe-prepare the dataset into {Path(directory)} or update the config."
        )
    return metadata


def write_prepared_splits(
    output_dir: str | Path,
    splits: Mapping[str, Sequence[ReviewRecord]],
    metadata: Mapping[str, Any],
) -> dict[str, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split_name in ("train", "dev", "test"):
        if split_name not in splits:
            raise ValueError(f"prepared data is missing split {split_name!r}")
        paths[split_name] = write_jsonl(destination / f"{split_name}.jsonl", splits[split_name])

    serialized_metadata = dict(metadata)
    serialized_metadata.setdefault("fingerprint_algorithm", "sha256")
    serialized_metadata["split_fingerprints"] = {
        split_name: sha256_file(paths[split_name])
        for split_name in ("train", "dev", "test")
    }
    metadata_path = destination / "metadata.json"
    with metadata_path.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(serialized_metadata, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
    paths["metadata"] = metadata_path
    return paths
