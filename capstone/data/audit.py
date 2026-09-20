"""Machine-readable counts for raw or prepared OATS review records."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping

from capstone.data.parser import DataIssue
from capstone.types import ReviewRecord


ZERO_LABEL_STATUSES = (
    "confirmed_unannotated",
    "legitimate_zero_opinion",
    "unknown",
)


def classify_review_annotation_status(
    reviews: Iterable[ReviewRecord],
    *,
    confirmed_unannotated_review_ids: Iterable[str] = (),
    legitimate_zero_opinion_review_ids: Iterable[str] = (),
) -> dict[str, str]:
    """Classify each review without inferring that an empty label means unannotated."""

    review_list = list(reviews)
    confirmed_ids = {str(value) for value in confirmed_unannotated_review_ids}
    legitimate_ids = {str(value) for value in legitimate_zero_opinion_review_ids}
    overlap = confirmed_ids & legitimate_ids
    if overlap:
        raise ValueError(f"review IDs have conflicting zero-label statuses: {sorted(overlap)}")

    has_valid_quadruple: dict[str, bool] = {}
    for review in review_list:
        has_valid_quadruple[review.review_id] = has_valid_quadruple.get(review.review_id, False) or any(
            sentence.quadruples for sentence in review.sentences
        )

    statuses: dict[str, str] = {}
    for review_id, has_quadruple in has_valid_quadruple.items():
        if has_quadruple:
            statuses[review_id] = "annotated"
            continue
        if review_id in confirmed_ids:
            statuses[review_id] = "confirmed_unannotated"
        elif review_id in legitimate_ids:
            statuses[review_id] = "legitimate_zero_opinion"
        else:
            statuses[review_id] = "unknown"
    return statuses


def _is_null(value: str) -> bool:
    return value.strip().casefold() in {"null", "none"}


def audit_reviews(
    reviews: Iterable[ReviewRecord],
    *,
    issues: Iterable[DataIssue] = (),
    split: str | None = None,
    confirmed_unannotated_review_ids: Iterable[str] = (),
    legitimate_zero_opinion_review_ids: Iterable[str] = (),
    review_status_by_id: Mapping[str, str] | None = None,
) -> dict[str, object]:
    review_list = list(reviews)
    issue_list = list(issues)
    sentences = [sentence for review in review_list for sentence in review.sentences]
    quadruples = [quadruple for sentence in sentences for quadruple in sentence.quadruples]
    aspect_counts = Counter(quadruple.aspect for quadruple in quadruples)
    sentiment_counts = Counter(quadruple.sentiment for quadruple in quadruples)
    invalid_category_count = sum(
        1
        for quadruple in quadruples
        if quadruple.aspect.strip().casefold().replace("_", " ").startswith("polarity")
    )
    invalid_category_count = max(
        invalid_category_count,
        sum(issue.reason == "invalid_aspect_category" for issue in issue_list),
    )
    statuses = classify_review_annotation_status(
        review_list,
        confirmed_unannotated_review_ids=confirmed_unannotated_review_ids,
        legitimate_zero_opinion_review_ids=legitimate_zero_opinion_review_ids,
    )
    if review_status_by_id is not None:
        statuses.update(review_status_by_id)
    status_counts = Counter(statuses.values())
    status_counts = Counter({status: status_counts.get(status, 0) for status in ZERO_LABEL_STATUSES})
    sentence_status_counts = Counter({status: 0 for status in ZERO_LABEL_STATUSES})
    for review in review_list:
        has_any_quadruple = any(sentence.quadruples for sentence in review.sentences)
        for sentence in review.sentences:
            if sentence.quadruples:
                continue
            if has_any_quadruple or statuses.get(review.review_id) == "annotated":
                sentence_status_counts["legitimate_zero_opinion"] += 1
            else:
                state = statuses.get(review.review_id, "unknown")
                if state in sentence_status_counts:
                    sentence_status_counts[state] += 1

    result: dict[str, object] = {
        "number_of_reviews": len({review.review_id for review in review_list}),
        "number_of_sentences": len(sentences),
        "sentences_with_quadruples": sum(bool(sentence.quadruples) for sentence in sentences),
        "empty_label_sentences": sum(not sentence.quadruples for sentence in sentences),
        "number_of_quadruples": len(quadruples),
        "explicit_target_count": sum(not _is_null(quadruple.target) for quadruple in quadruples),
        "implicit_target_count": sum(_is_null(quadruple.target) for quadruple in quadruples),
        "explicit_opinion_count": sum(not _is_null(quadruple.opinion) for quadruple in quadruples),
        "implicit_opinion_count": sum(_is_null(quadruple.opinion) for quadruple in quadruples),
        "aspect_distribution": dict(sorted(aspect_counts.items())),
        "sentiment_distribution": dict(sorted(sentiment_counts.items())),
        "malformed_quadruple_count": sum(
            issue.reason == "malformed_quadruple" for issue in issue_list
        ),
        "invalid_category_count": invalid_category_count,
        "suspicious_annotation_count": sum(
            issue.reason == "suspicious_annotation" for issue in issue_list
        ),
        "malformed_sentence_count": sum(
            issue.reason == "malformed_sentence" for issue in issue_list
        ),
        "zero_label_review_status_counts": dict(status_counts),
        "zero_label_sentence_status_counts": dict(sentence_status_counts),
        "zero_label_review_status_by_id": dict(sorted(statuses.items())),
    }
    if split is not None:
        result["split"] = split
    return result
