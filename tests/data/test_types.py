from capstone.types import Quadruple, ReviewRecord, SentenceRecord


def test_quadruple_round_trip_preserves_asqp_fields():
    quad = Quadruple(
        target="NULL",
        aspect="course quality",
        opinion="NULL",
        sentiment="neutral",
    )

    assert Quadruple.from_dict(quad.to_dict()) == quad
    assert quad.to_dict() == {
        "target": "NULL",
        "aspect": "course quality",
        "opinion": "NULL",
        "sentiment": "neutral",
    }


def test_review_round_trip_retains_review_sentence_linkage_and_empty_labels():
    review = ReviewRecord(
        review_id="review-7",
        sentences=[
            SentenceRecord(
                review_id="review-7",
                sentence_id="review-7:0",
                text="Helpful course.",
                quadruples=[
                    Quadruple("course", "course quality", "Helpful", "positive")
                ],
            ),
            SentenceRecord(
                review_id="review-7",
                sentence_id="review-7:1",
                text="Thanks.",
                quadruples=[],
            ),
        ],
    )

    restored = ReviewRecord.from_dict(review.to_dict())

    assert restored == review
    assert all(sentence.review_id == restored.review_id for sentence in restored.sentences)
    assert restored.sentences[1].quadruples == []
