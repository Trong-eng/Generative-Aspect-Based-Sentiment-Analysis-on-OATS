from capstone.models.paraphrase import parse_paraphrase_generation, serialize_paraphrase
from capstone.types import Quadruple


def test_paraphrase_serialization_and_parser_support_multiple_quads():
    expected = [
        Quadruple("room", "rooms cleanliness", "spotless", "positive"),
        Quadruple("NULL", "hotel general", "recommend", "negative"),
    ]

    parsed = parse_paraphrase_generation(
        serialize_paraphrase(expected),
        known_aspects={"rooms cleanliness", "hotel general"},
    )

    assert parsed.quadruples == expected
    assert parsed.invalid_predictions == []


def test_paraphrase_serialization_maps_implicit_target_and_opinion_to_it():
    serialized = serialize_paraphrase(
        [Quadruple("NULL", "hotel general", "NULL", "positive")]
    )

    assert serialized == "hotel general is great because it is it"
    parsed = parse_paraphrase_generation(serialized)
    assert parsed.quadruples == [
        Quadruple("NULL", "hotel general", "NULL", "positive")
    ]


def test_paraphrase_parser_drops_unknown_aspect_as_invalid_without_crashing():
    parsed = parse_paraphrase_generation(
        "unknown aspect is great because room is spotless",
        known_aspects={"rooms cleanliness"},
    )

    assert parsed.quadruples == []
    assert parsed.invalid_predictions[0]["reason"] == "invalid_aspect_category"


def test_paraphrase_parser_handles_malformed_segment_and_continues():
    parsed = parse_paraphrase_generation(
        "rooms cleanliness is great because room is spotless [SSEP] not a paraphrase"
    )

    assert parsed.quadruples == [
        Quadruple("room", "rooms cleanliness", "spotless", "positive")
    ]
    assert parsed.invalid_predictions[0]["reason"] == "malformed_generation"


def test_paraphrase_parser_preserves_duplicate_predictions_for_false_positive_accounting():
    parsed = parse_paraphrase_generation(
        "rooms cleanliness is great because room is clean [SSEP] "
        "rooms cleanliness is great because room is clean"
    )

    assert len(parsed.quadruples) == 2


def test_paraphrase_parser_reports_unexpected_separator_and_continues():
    parsed = parse_paraphrase_generation(
        "rooms cleanliness is great because room is clean | "
        "hotel general is ok because hotel is quiet"
    )

    assert len(parsed.quadruples) == 2
    assert parsed.invalid_predictions == [{"raw": "|", "reason": "unexpected_separator"}]


def test_paraphrase_parser_reports_unknown_separator_token():
    parsed = parse_paraphrase_generation(
        "rooms cleanliness is great because room is clean <sep> "
        "hotel general is ok because hotel is quiet"
    )

    assert len(parsed.quadruples) == 2
    assert parsed.invalid_predictions == [{"raw": "<sep>", "reason": "unexpected_separator"}]
