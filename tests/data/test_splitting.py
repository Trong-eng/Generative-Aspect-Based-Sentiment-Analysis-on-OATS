import pytest

from capstone.data.splitting import assert_no_review_leakage, split_reviews
from capstone.types import ReviewRecord, SentenceRecord


def _review(review_id):
    return ReviewRecord(
        review_id,
        [
            SentenceRecord(review_id, f"{review_id}:0", "First sentence.", []),
            SentenceRecord(review_id, f"{review_id}:1", "Second sentence.", []),
        ],
    )


def test_review_split_is_seeded_and_keeps_all_sentences_together():
    reviews = [_review(f"r{index}") for index in range(20)]

    first = split_reviews(reviews, seed=42)
    second = split_reviews(reviews, seed=42)

    assert {
        split: [review.review_id for review in values]
        for split, values in first.items()
    } == {
        split: [review.review_id for review in values]
        for split, values in second.items()
    }
    assert [len(first[name]) for name in ("train", "dev", "test")] == [16, 2, 2]
    assert all(
        sentence.review_id == review.review_id
        for values in first.values()
        for review in values
        for sentence in review.sentences
    )
    assert_no_review_leakage(first)


def test_leakage_assertion_rejects_overlapping_review_ids():
    with pytest.raises(ValueError, match="review-level leakage"):
        assert_no_review_leakage(
            {"train": [_review("shared")], "dev": [_review("shared")], "test": []}
        )
