import pytest

from capstone.data.cleaning import clean_dataset
from capstone.data.parser import DataIssue
from capstone.types import Quadruple, ReviewRecord, SentenceRecord


def _review(review_id, sentences):
    return ReviewRecord(
        review_id=review_id,
        sentences=[
            SentenceRecord(review_id, sentence_id, text, quads)
            for sentence_id, text, quads in sentences
        ],
    )


def test_cleaning_removes_invalid_polarity_category_and_logs_reason():
    source = _review(
        "r1",
        [
            (
                "s1",
                "Room was clean.",
                [
                    Quadruple("room", "polarity positive", "clean", "positive"),
                    Quadruple("room", "rooms cleanliness", "clean", "positive"),
                ],
            )
        ],
    )

    result = clean_dataset([source])

    assert len(result.reviews) == 1
    assert [q.aspect for q in result.reviews[0].sentences[0].quadruples] == [
        "rooms cleanliness"
    ]
    issue = next(issue for issue in result.issues if issue.reason == "invalid_aspect_category")
    assert issue.review_id == "r1"
    assert issue.sentence_id == "s1"
    assert issue.quadruple["aspect"] == "polarity positive"


def test_cleaning_retains_zero_label_reviews_by_default_and_keeps_empty_sentences():
    labeled = _review(
        "labeled",
        [
            ("labeled:0", "A labeled sentence.", [Quadruple("NULL", "hotel general", "great", "positive")]),
            ("labeled:1", "A legitimate empty sentence.", []),
        ],
    )
    unannotated = _review(
        "unannotated",
        [("unannotated:0", "No annotation in the review.", [])],
    )

    result = clean_dataset([labeled, unannotated])

    assert [review.review_id for review in result.reviews] == ["labeled", "unannotated"]
    assert len(result.reviews[0].sentences) == 2
    assert result.reviews[0].sentences[1].quadruples == []
    assert result.reviews[1].sentences[0].quadruples == []
    assert result.excluded_reviews == []


def test_cleaning_excludes_only_review_ids_confirmed_by_manifest():
    labeled = _review(
        "labeled",
        [("labeled:0", "A labeled sentence.", [Quadruple("NULL", "hotel general", "great", "positive")])],
    )
    confirmed = _review(
        "confirmed-empty",
        [("confirmed-empty:0", "No annotation in the review.", [])],
    )
    unknown = _review("unknown-empty", [("unknown-empty:0", "Unknown status.", [])])
    legitimate = _review(
        "legitimate-empty",
        [("legitimate-empty:0", "Verified zero-opinion review.", [])],
    )

    result = clean_dataset(
        [labeled, confirmed, unknown, legitimate],
        confirmed_unannotated_review_ids={"confirmed-empty"},
        legitimate_zero_opinion_review_ids={"legitimate-empty"},
        manifest_provenance="Cross-checked against annotation records.",
        manifest_path="annotation_status/hotels.json",
        manifest_sha256="f" * 64,
    )

    assert [review.review_id for review in result.reviews] == [
        "labeled",
        "unknown-empty",
        "legitimate-empty",
    ]
    assert result.excluded_reviews == [
        {
            "review_id": "confirmed-empty",
            "reason": "unannotated_review",
            "annotation_status": "confirmed_unannotated",
            "evidence": "Cross-checked against annotation records.",
            "provenance": "Cross-checked against annotation records.",
            "provenance_type": "manifest",
            "manifest_path": "annotation_status/hotels.json",
            "manifest_sha256": "f" * 64,
        }
    ]


def test_manifest_cannot_exclude_a_review_with_valid_labels():
    labeled = _review(
        "labeled",
        [("labeled:0", "A labeled sentence.", [Quadruple("NULL", "hotel general", "great", "positive")])],
    )

    with pytest.raises(ValueError, match="contains valid quadruples"):
        clean_dataset([labeled], confirmed_unannotated_review_ids={"labeled"})


def test_manifest_cannot_exclude_a_review_with_only_rejected_raw_annotations():
    invalid = _review(
        "invalid-only",
        [(
            "invalid-only:0",
            "An invalid annotation attempt.",
            [Quadruple("room", "POLARITY#POSITIVE", "clean", "positive")],
        )],
    )

    with pytest.raises(ValueError, match="raw quadruple annotations"):
        clean_dataset([invalid], confirmed_unannotated_review_ids={"invalid-only"})


def test_manifest_cannot_classify_a_review_with_malformed_parser_evidence():
    empty = _review("malformed-only", [("malformed-only:0", "An unparsed annotation.", [])])
    malformed = DataIssue(
        reason="malformed_quadruple",
        review_id="malformed-only",
        sentence_id="malformed-only:0",
        details="annotation XML could not be normalized",
    )

    with pytest.raises(ValueError, match="parser or cleaning issues"):
        clean_dataset(
            [empty],
            source_issues=[malformed],
            legitimate_zero_opinion_review_ids={"malformed-only"},
        )


def test_conflict_is_not_mapped_without_official_asqp_evidence():
    review = _review(
        "r1",
        [("s1", "Mixed feedback.", [Quadruple("course", "course quality", "NULL", "conflict")])],
    )

    raw = clean_dataset([review], sentiment_mode="raw_4_class")
    compatible = clean_dataset([review], sentiment_mode="paper_compatible")

    assert raw.reviews[0].sentences[0].quadruples[0].sentiment == "conflict"
    assert compatible.reviews[0].sentences[0].quadruples[0].sentiment == "conflict"


def test_malformed_source_zero_quadruple_review_is_retained_by_default():
    review = _review("r1", [("s1", "Malformed label.", [])])
    malformed = DataIssue(
        reason="malformed_quadruple",
        review_id="r1",
        sentence_id="s1",
        quadruple={"target": "room"},
    )

    result = clean_dataset([review], source_issues=[malformed])

    assert [review.review_id for review in result.reviews] == ["r1"]
    assert result.reviews[0].sentences[0].quadruples == []
    assert result.excluded_reviews == []
