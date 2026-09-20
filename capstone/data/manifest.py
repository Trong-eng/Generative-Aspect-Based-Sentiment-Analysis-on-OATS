"""Explicit review annotation-status manifests with byte-level provenance."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from capstone.data.serialization import sha256_file


@dataclass(frozen=True)
class UnannotatedReviewManifest:
    path: Path
    domain: str
    provenance: str
    confirmed_review_ids: frozenset[str]
    legitimate_zero_opinion_review_ids: frozenset[str]
    sha256: str

    def validate_source_review_ids(self, source_review_ids: Iterable[str]) -> None:
        """Reject manifest decisions that do not identify a review in this source."""

        available_ids = {str(review_id) for review_id in source_review_ids}
        listed_ids = self.confirmed_review_ids | self.legitimate_zero_opinion_review_ids
        unmatched = sorted(listed_ids - available_ids)
        if unmatched:
            raise ValueError(
                f"{self.path}: manifest review IDs are absent from the selected source dataset: "
                f"{unmatched}"
            )

    def to_metadata(
        self,
        *,
        source_review_ids: Iterable[str],
        excluded_review_ids: Iterable[str],
    ) -> dict[str, object]:
        source_ids = set(source_review_ids)
        excluded_ids = set(excluded_review_ids)
        listed_ids = self.confirmed_review_ids | self.legitimate_zero_opinion_review_ids
        return {
            "path": str(self.path),
            "domain": self.domain,
            "provenance": self.provenance,
            "sha256": self.sha256,
            "confirmed_unannotated_review_ids": sorted(self.confirmed_review_ids),
            "legitimate_zero_opinion_review_ids": sorted(
                self.legitimate_zero_opinion_review_ids
            ),
            "matched_review_ids": sorted(listed_ids & source_ids),
            "unmatched_review_ids": sorted(listed_ids - source_ids),
            "excluded_review_ids": sorted(excluded_ids),
        }


def _read_ids(payload: dict[str, object], key: str, path: Path) -> frozenset[str]:
    value = payload.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"{path}: '{key}' must be a list of review IDs")
    normalized: list[str] = []
    for review_id in value:
        if isinstance(review_id, bool) or not isinstance(review_id, (str, int)):
            raise ValueError(f"{path}: '{key}' values must be strings or integers")
        normalized_id = str(review_id).strip()
        if not normalized_id:
            raise ValueError(f"{path}: '{key}' cannot contain an empty review ID")
        normalized.append(normalized_id)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{path}: '{key}' contains duplicate review IDs")
    return frozenset(normalized)


def load_unannotated_review_manifest(
    path: str | Path,
    *,
    expected_domain: str,
) -> UnannotatedReviewManifest:
    """Load a JSON manifest that explicitly classifies empty-label reviews.

    Required fields are ``domain``, non-empty ``provenance``, and
    ``confirmed_unannotated_review_ids``. The optional
    ``legitimate_zero_opinion_review_ids`` list records confirmed legitimate
    zero-opinion reviews that must be retained.
    """

    source = Path(path).expanduser().resolve()
    try:
        raw_bytes = source.read_bytes()
        payload = json.loads(raw_bytes.decode("utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read annotation-status manifest at {source}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{source}: manifest root must be a JSON object")

    domain = payload.get("domain")
    if not isinstance(domain, str) or not domain.strip():
        raise ValueError(f"{source}: manifest requires a non-empty 'domain'")
    domain = domain.strip().lower()
    if domain != expected_domain:
        raise ValueError(
            f"{source}: manifest domain {domain!r} does not match requested domain "
            f"{expected_domain!r}"
        )
    provenance = payload.get("provenance")
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError(f"{source}: manifest requires non-empty provenance")

    if "confirmed_unannotated_review_ids" not in payload:
        raise ValueError(
            f"{source}: manifest requires 'confirmed_unannotated_review_ids' (use [] when none)"
        )
    confirmed_ids = _read_ids(payload, "confirmed_unannotated_review_ids", source)
    legitimate_ids = _read_ids(payload, "legitimate_zero_opinion_review_ids", source)
    overlap = confirmed_ids & legitimate_ids
    if overlap:
        raise ValueError(
            f"{source}: review IDs cannot be both confirmed and legitimate: {sorted(overlap)}"
        )

    return UnannotatedReviewManifest(
        path=source,
        domain=domain,
        provenance=provenance.strip(),
        confirmed_review_ids=confirmed_ids,
        legitimate_zero_opinion_review_ids=legitimate_ids,
        sha256=sha256_file(source),
    )
