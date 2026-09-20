"""Deterministic review-level train/dev/test splitting."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

from capstone.types import ReviewRecord


def assert_no_review_leakage(splits: Mapping[str, Sequence[ReviewRecord]]) -> None:
    owners: dict[str, str] = {}
    for split_name in ("train", "dev", "test"):
        for review in splits.get(split_name, ()):
            previous = owners.get(review.review_id)
            if previous is not None and previous != split_name:
                raise ValueError(
                    "review-level leakage: review "
                    f"{review.review_id!r} appears in {previous!r} and {split_name!r}"
                )
            owners[review.review_id] = split_name


def split_reviews(
    reviews: Sequence[ReviewRecord],
    *,
    seed: int = 42,
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> dict[str, list[ReviewRecord]]:
    """Shuffle reviews once and divide them 80/10/10 by default."""

    ratios = (train_ratio, dev_ratio, test_ratio)
    if any(ratio < 0 for ratio in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("train/dev/test ratios must be non-negative and sum to 1")
    ids = [review.review_id for review in reviews]
    if len(ids) != len(set(ids)):
        raise ValueError("clean split requires one complete record per unique review_id")

    shuffled = list(reviews)
    random.Random(seed).shuffle(shuffled)
    train_end = int(len(shuffled) * train_ratio)
    dev_end = train_end + int(len(shuffled) * dev_ratio)
    splits = {
        "train": shuffled[:train_end],
        "dev": shuffled[train_end:dev_end],
        "test": shuffled[dev_end:],
    }
    assert_no_review_leakage(splits)
    return splits
