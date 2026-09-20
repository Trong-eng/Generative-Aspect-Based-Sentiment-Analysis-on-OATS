from capstone.data.inventory import collect_aspect_inventory
from capstone.types import Quadruple, ReviewRecord, SentenceRecord


def test_aspect_inventory_uses_all_supplied_retained_reviews_and_is_sorted():
    reviews = [
        ReviewRecord(
            "r1",
            [SentenceRecord("r1", "r1:0", "A.", [Quadruple("x", "z aspect", "y", "positive")])],
        ),
        ReviewRecord(
            "r2",
            [SentenceRecord("r2", "r2:0", "B.", [Quadruple("x", "a aspect", "y", "negative")])],
        ),
        ReviewRecord("r3", [SentenceRecord("r3", "r3:0", "C.", [])]),
    ]

    assert collect_aspect_inventory(reviews) == ["a aspect", "z aspect"]
