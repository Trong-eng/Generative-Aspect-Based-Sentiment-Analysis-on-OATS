"""Validate annotations and remove only reviews known to lack annotation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from capstone.data.parser import DataIssue
from capstone.evaluation.normalization import normalize_quadruple
from capstone.types import Quadruple, ReviewRecord, SentenceRecord


VALID_SENTIMENTS = frozenset({"positive", "negative", "neutral", "conflict"})


@dataclass
class CleaningResult:
    reviews: list[ReviewRecord]
    excluded_reviews: list[dict[str, str]] = field(default_factory=list)
    issues: list[DataIssue] = field(default_factory=list)
    annotation_status_by_review_id: dict[str, str] = field(default_factory=dict)


@dataclass
class SplitRowsCleaningResult:
    rows: list[SentenceRecord]
    issues: list[DataIssue] = field(default_factory=list)


def _invalid_aspect(aspect: str) -> bool:
    normalized = " ".join(aspect.replace("#", " ").replace("_", " ").split()).casefold()
    return normalized == "polarity" or normalized.startswith("polarity ")


def _with_sentiment_mode(quadruple: Quadruple, sentiment_mode: str) -> Quadruple:
    if sentiment_mode not in {"raw_4_class", "paper_compatible"}:
        raise ValueError("sentiment_mode must be 'raw_4_class' or 'paper_compatible'")
    # Preserve conflict until an official ASQP source establishes a conversion.
    return quadruple


def _clean_sentence(
    sentence: SentenceRecord,
    *,
    sentiment_mode: str,
    issues: list[DataIssue],
    review_id: str | None,
    source_path: str | None = None,
) -> SentenceRecord:
    cleaned_quads: list[Quadruple] = []
    for quadruple in sentence.quadruples:
        if _invalid_aspect(quadruple.aspect):
            issues.append(
                DataIssue(
                    "invalid_aspect_category",
                    review_id,
                    sentence.sentence_id,
                    quadruple.to_dict(),
                    "aspect category uses the invalid POLARITY#* namespace",
                    source_path,
                    sentence.source_line_number,
                )
            )
            continue
        if quadruple.sentiment not in VALID_SENTIMENTS:
            issues.append(
                DataIssue(
                    "invalid_sentiment",
                    review_id,
                    sentence.sentence_id,
                    quadruple.to_dict(),
                    f"unsupported sentiment {quadruple.sentiment!r}",
                    source_path,
                    sentence.source_line_number,
                )
            )
            continue
        normalized = normalize_quadruple(quadruple)
        if normalized is None:
            issues.append(
                DataIssue(
                    "malformed_quadruple",
                    review_id,
                    sentence.sentence_id,
                    quadruple.to_dict(),
                    "quadruple fields could not be normalized",
                    source_path,
                    sentence.source_line_number,
                )
            )
            continue
        cleaned_quads.append(_with_sentiment_mode(normalized, sentiment_mode))
    return SentenceRecord(
        review_id=review_id,
        sentence_id=sentence.sentence_id,
        text=sentence.text,
        quadruples=cleaned_quads,
        source_line_number=sentence.source_line_number,
    )


def clean_split_rows(
    rows: Iterable[SentenceRecord],
    *,
    sentiment_mode: str = "raw_4_class",
    source_path: str | None = None,
) -> SplitRowsCleaningResult:
    """Apply the standard quadruple filters to independent TXT source rows."""
    if sentiment_mode not in {"raw_4_class", "paper_compatible"}:
        raise ValueError("sentiment_mode must be 'raw_4_class' or 'paper_compatible'")
    row_list = list(rows)
    issues: list[DataIssue] = []
    cleaned_rows: list[SentenceRecord] = []
    for row in row_list:
        if row.review_id is not None or row.sentence_id is not None:
            raise ValueError("provided split rows must not have review_id or sentence_id")
        if row.source_line_number is None:
            raise ValueError("provided split rows require source_line_number")
        cleaned_rows.append(
            _clean_sentence(
                row,
                sentiment_mode=sentiment_mode,
                issues=issues,
                review_id=None,
                source_path=source_path,
            )
        )
    return SplitRowsCleaningResult(cleaned_rows, issues)


def clean_dataset(
    reviews: Iterable[ReviewRecord],
    *,
    sentiment_mode: str = "raw_4_class",
    source_issues: Iterable[DataIssue] = (),
    confirmed_unannotated_review_ids: Iterable[str] = (),
    legitimate_zero_opinion_review_ids: Iterable[str] = (),
    manifest_provenance: str | None = None,
    manifest_path: str | None = None,
    manifest_sha256: str | None = None,
) -> CleaningResult:
    """Filter malformed labels and exclude only explicitly confirmed reviews.

    Zero-quadruple reviews are retained by default because their status is
    unknown. Only IDs in ``confirmed_unannotated_review_ids`` are excluded;
    ``legitimate_zero_opinion_review_ids`` are explicitly retained. Empty
    sentences in reviews with labels also remain in the output.
    """

    if sentiment_mode not in {"raw_4_class", "paper_compatible"}:
        raise ValueError("sentiment_mode must be 'raw_4_class' or 'paper_compatible'")

    inherited_issues = list(source_issues)
    issues = list(inherited_issues)
    confirmed_ids = {str(review_id) for review_id in confirmed_unannotated_review_ids}
    legitimate_ids = {str(review_id) for review_id in legitimate_zero_opinion_review_ids}
    overlap = confirmed_ids & legitimate_ids
    if overlap:
        raise ValueError(
            "review IDs cannot be both confirmed unannotated and legitimate zero-opinion: "
            f"{sorted(overlap)}"
        )

    retained: list[ReviewRecord] = []
    excluded: list[dict[str, str]] = []
    annotation_status: dict[str, str] = {}
    for review in reviews:
        cleaned_sentences: list[SentenceRecord] = []
        cleaned_quadruple_count = 0
        raw_quadruple_count = sum(len(sentence.quadruples) for sentence in review.sentences)
        for sentence in review.sentences:
            cleaned_sentence = _clean_sentence(
                sentence,
                sentiment_mode=sentiment_mode,
                issues=issues,
                review_id=review.review_id,
            )
            cleaned_quads = cleaned_sentence.quadruples
            cleaned_quadruple_count += len(cleaned_quads)
            cleaned_sentences.append(cleaned_sentence)

        explicitly_zero_label = (
            review.review_id in confirmed_ids or review.review_id in legitimate_ids
        )
        if explicitly_zero_label and cleaned_quadruple_count > 0:
            status = (
                "confirmed unannotated"
                if review.review_id in confirmed_ids
                else "legitimate zero-opinion"
            )
            raise ValueError(
                f"manifest marks review {review.review_id!r} as {status}, but it "
                "contains valid quadruples"
            )
        if explicitly_zero_label and raw_quadruple_count > 0:
            raise ValueError(
                f"manifest marks review {review.review_id!r} as zero-label, but it "
                "contains raw quadruple annotations that may have been rejected during cleaning"
            )
        review_issues = [issue for issue in issues if issue.review_id == review.review_id]
        if explicitly_zero_label and review_issues:
            reasons = sorted({issue.reason for issue in review_issues})
            raise ValueError(
                f"manifest marks review {review.review_id!r} as zero-label, but parser or "
                f"cleaning issues remain: {', '.join(reasons)}"
            )
        annotation_status[review.review_id] = (
            "annotated"
            if cleaned_quadruple_count > 0
            else "confirmed_unannotated"
            if review.review_id in confirmed_ids
            else "legitimate_zero_opinion"
            if review.review_id in legitimate_ids
            else "unknown"
        )
        if review.review_id in confirmed_ids:
            excluded.append(
                {
                    "review_id": review.review_id,
                    "reason": "unannotated_review",
                    "annotation_status": "confirmed_unannotated",
                    "evidence": manifest_provenance or "Explicitly listed in supplied annotation-status manifest.",
                    "provenance": manifest_provenance or "Explicitly listed in supplied annotation-status manifest.",
                    "provenance_type": "manifest",
                    "manifest_path": manifest_path or "",
                    "manifest_sha256": manifest_sha256 or "",
                }
            )
            continue

        retained.append(ReviewRecord(review.review_id, cleaned_sentences))

    return CleaningResult(retained, excluded, issues, annotation_status)
