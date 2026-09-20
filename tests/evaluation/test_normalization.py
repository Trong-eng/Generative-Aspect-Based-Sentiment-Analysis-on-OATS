from capstone.evaluation import (
    PredictionErrorCode,
    normalize_quadruple,
    normalize_quadruples,
    normalize_quadruples_with_issues,
)
from capstone.types import Quadruple


def test_normalization_trims_fields_canonicalizes_known_aspects_and_sentiment():
    item = {
        "target": " NULL ",
        "aspect": " food#quality ",
        "opinion": "  tasty ",
        "sentiment": " Positive ",
    }

    result = normalize_quadruple(item, known_aspects=["FOOD#QUALITY"])

    assert result == Quadruple("NULL", "food quality", "tasty", "positive")


def test_normalization_canonicalizes_null_aliases_and_aspect_categories():
    result = normalize_quadruple(
        {
            "target": " none ",
            "aspect": " FOOD#QUALITY ",
            "opinion": " Null ",
            "sentiment": " POS ",
        }
    )

    assert result == Quadruple("NULL", "food quality", "NULL", "positive")


def test_normalization_collapses_whitespace_and_preserves_category_underscores():
    result = normalize_quadruple(
        {
            "target": "  New   York ",
            "aspect": " ROOMS#DESIGN_FEATURES ",
            "opinion": " Very   comfortable ",
            "sentiment": " Positive ",
        }
    )

    assert result == Quadruple(
        "new york", "rooms design_features", "very comfortable", "positive"
    )


def test_normalize_quadruple_rejects_missing_or_non_string_fields():
    assert normalize_quadruple({"target": "food", "aspect": "FOOD#QUALITY"}) is None
    assert normalize_quadruple(
        {"target": "food", "aspect": "FOOD#QUALITY", "opinion": 7, "sentiment": "positive"}
    ) is None


def test_python_none_is_a_null_sentinel_for_text_fields():
    result = normalize_quadruple(
        {"target": None, "aspect": "FOOD#QUALITY", "opinion": None, "sentiment": "neutral"}
    )

    assert result == Quadruple("NULL", "food quality", "NULL", "neutral")


def test_null_none_and_empty_aspects_are_rejected():
    for aspect in ("NULL", "null", "None", None, "", "   "):
        assert normalize_quadruple(
            {"target": "food", "aspect": aspect, "opinion": "great", "sentiment": "positive"}
        ) is None


def test_invalid_sentiment_label_is_rejected():
    assert normalize_quadruple(
        {"target": "food", "aspect": "FOOD#QUALITY", "opinion": "great", "sentiment": "bullish"}
    ) is None
    assert normalize_quadruple(
        {"target": "food", "aspect": "FOOD#QUALITY", "opinion": "great", "sentiment": "None"}
    ) is None


def test_collection_normalizer_reports_unknown_fields_and_skips_invalid_items():
    normalized, issues = normalize_quadruples_with_issues(
        [
            {
                "target": "food",
                "aspect": "FOOD#QUALITY",
                "opinion": "great",
                "sentiment": "positive",
                "confidence": 0.9,
            },
            {"target": "food", "aspect": "FOOD#QUALITY"},
        ]
    )

    assert normalized == [Quadruple("food", "food quality", "great", "positive")]
    assert [issue.code for issue in issues] == [
        PredictionErrorCode.UNKNOWN_FIELD,
        PredictionErrorCode.INVALID,
        PredictionErrorCode.INVALID,
    ]


def test_json_generation_with_unrecognized_fields_is_classified_without_raising():
    normalized, issues = normalize_quadruples_with_issues(
        '{"response": "no structured output"}',
        parse_generation=True,
    )

    assert normalized == []
    assert PredictionErrorCode.UNKNOWN_FIELD in [issue.code for issue in issues]
    assert PredictionErrorCode.MISSING in [issue.code for issue in issues]


def test_public_collection_normalizer_handles_missing_and_malformed_inputs():
    assert normalize_quadruples(None) == []
    assert normalize_quadruples("not a collection") == []
