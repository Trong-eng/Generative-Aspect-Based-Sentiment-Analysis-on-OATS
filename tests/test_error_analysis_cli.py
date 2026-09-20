import json
from pathlib import Path

from capstone.error_analysis import write_error_analysis


def test_error_analysis_cli_writes_structured_mismatch_cases(tmp_path: Path):
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps({
        "review_id": "r1",
        "sentence_id": "r1:0",
        "text": "The room was spotless.",
        "gold_quadruples": [{
            "target": "room", "aspect": "rooms cleanliness", "opinion": "spotless", "sentiment": "positive",
        }],
        "raw_generation": "generated",
        "parsed_quadruples": [],
        "normalized_quadruples": [],
        "invalid_predictions": [{"raw": "generated", "reason": "malformed_generation"}],
    }) + "\n", encoding="utf-8")

    output = write_error_analysis(predictions)

    errors = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert errors
    assert errors[0]["review_id"] == "r1"
    assert errors[0]["invalid_predictions"][0]["reason"] == "malformed_generation"
    assert "malformed_generation" in errors[0]["error_types"]
