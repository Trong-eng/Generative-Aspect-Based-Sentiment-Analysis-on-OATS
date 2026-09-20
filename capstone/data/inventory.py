"""Dataset-level label inventory helpers."""

from __future__ import annotations

from collections.abc import Iterable

from capstone.types import ReviewRecord


def collect_aspect_inventory(reviews: Iterable[ReviewRecord]) -> list[str]:
    """Return sorted distinct aspect labels across the supplied retained reviews."""

    aspects = {
        quadruple.aspect
        for review in reviews
        for sentence in review.sentences
        for quadruple in sentence.quadruples
    }
    return sorted(aspects)
