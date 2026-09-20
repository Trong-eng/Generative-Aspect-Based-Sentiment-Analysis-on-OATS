from capstone.models.gas import parse_gas_generation, serialize_gas
from capstone.types import Quadruple


def test_gas_serialization_round_trips_multiple_quads_and_nulls():
    expected = [
        Quadruple("room", "rooms cleanliness", "spotless", "positive"),
        Quadruple("NULL", "hotel general", "NULL", "neutral"),
    ]

    parsed = parse_gas_generation(serialize_gas(expected))

    assert parsed.quadruples == expected
    assert parsed.invalid_predictions == []
    assert serialize_gas(expected).startswith('[["room","rooms cleanliness","spotless","positive"]')


def test_gas_empty_quadruple_list_round_trips_as_valid_empty_prediction():
    assert serialize_gas([]) == "[]"
    parsed = parse_gas_generation("[]")

    assert parsed.quadruples == []
    assert parsed.invalid_predictions == []


def test_gas_parser_keeps_valid_quads_when_generation_contains_malformed_item():
    generated = (
        '[{"target":"room","aspect":"rooms cleanliness",'
        '"opinion":"spotless","sentiment":"positive"}] [QSEP] '
        '{"target":"missing fields"}'
    )

    parsed = parse_gas_generation(generated)

    assert parsed.quadruples == [
        Quadruple("room", "rooms cleanliness", "spotless", "positive")
    ]
    assert parsed.invalid_predictions[0]["reason"] == "missing_field"


def test_gas_parser_treats_qsep_as_a_candidate_boundary():
    parsed = parse_gas_generation(
        '["room","rooms cleanliness","spotless","positive"] [QSEP] '
        '["hotel","hotel general","great","positive"]'
    )

    assert len(parsed.quadruples) == 2
    assert parsed.invalid_predictions == []


def test_gas_parser_returns_invalid_record_instead_of_raising_for_malformed_json():
    parsed = parse_gas_generation('{"target":"room",')

    assert parsed.quadruples == []
    assert parsed.invalid_predictions
    assert parsed.invalid_predictions[0]["reason"] == "malformed_generation"


def test_gas_parser_preserves_duplicate_predictions_for_false_positive_accounting():
    generation = (
        '[{"target":"room","aspect":"rooms cleanliness",'
        '"opinion":"clean","sentiment":"positive"},'
        '{"target":"room","aspect":"rooms cleanliness",'
        '"opinion":"clean","sentiment":"positive"}]'
    )

    parsed = parse_gas_generation(generation)

    assert len(parsed.quadruples) == 2
