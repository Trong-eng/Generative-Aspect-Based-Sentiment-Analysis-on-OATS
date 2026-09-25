import pytest

from capstone.evaluation import evaluate_quadruples, evaluate_records
from capstone.types import Quadruple


def q(target, aspect, opinion, sentiment):
    return Quadruple(target=target, aspect=aspect, opinion=opinion, sentiment=sentiment)


def test_exact_metrics_are_multiset_safe():
    gold = [q("food", "FOOD#QUALITY", "great", "positive"), q("food", "FOOD#QUALITY", "great", "positive")]
    predicted = [q("food", "FOOD#QUALITY", "great", "positive")]

    metrics = evaluate_quadruples(gold, predicted)

    assert metrics == {
        "tp": 1,
        "fp": 0,
        "fn": 1,
        "precision": 1.0,
        "recall": 0.5,
        "f1": 2 / 3,
    }

    duplicate_prediction_metrics = evaluate_quadruples(
        [q("food", "FOOD#QUALITY", "great", "positive")],
        [
            q("food", "FOOD#QUALITY", "great", "positive"),
            q("food", "FOOD#QUALITY", "great", "positive"),
        ],
    )
    assert duplicate_prediction_metrics["tp"] == 1
    assert duplicate_prediction_metrics["fp"] == 1


def test_zero_denominators_produce_zero_scores():
    assert evaluate_quadruples([], []) == {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }


def test_record_diagnostics_include_components_strata_and_count_buckets():
    gold = [q("", "SERVICE#GENERAL", "slow", "negative")]
    predicted = [q("waiter", "SERVICE#GENERAL", "slow", "negative")]

    metrics, diagnostics, errors = evaluate_records(
        [
            {
                "review_id": "r1",
                "sentence_id": "s1",
                "text": "Service was slow.",
                "gold_quadruples": gold,
                "predicted_quadruples": predicted,
            },
            {"gold_quadruples": [], "predicted_quadruples": []},
        ]
    )

    assert metrics["tp"] == 0
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert set(diagnostics["components"]) == {"target", "aspect", "opinion", "sentiment"}
    assert set(diagnostics["explicit_implicit"]) == {"target", "opinion"}
    assert set(diagnostics["explicit_implicit"]["target"]) == {"explicit", "implicit"}
    assert set(diagnostics["explicit_implicit"]["opinion"]) == {"explicit", "implicit"}
    assert "service general" in diagnostics["per_aspect"]
    assert "negative" in diagnostics["per_sentiment"]
    assert set(diagnostics["by_quadruple_count"]) == {"0", "1", "2", "3+"}
    assert diagnostics["counts"]["examples"] == 2
    assert diagnostics["by_quadruple_count"]["0"]["example_accuracy"] == 1.0
    assert errors[0]["counts"] == {"tp": 0, "fp": 1, "fn": 1}
    assert errors[0]["wrong_field_mismatches"][0]["wrong_fields"] == ["target"]
    assert len(errors[0]["implicit_target_failures"]) == 1
    assert diagnostics["counts"]["example_accuracy"] == 0.5


def test_implicit_target_failures_include_unpaired_misses_and_spurious_predictions():
    implicit_target = q("", "SERVICE#GENERAL", "slow", "negative")
    expected_quadruple = {
        "target": "NULL",
        "aspect": "service general",
        "opinion": "slow",
        "sentiment": "negative",
    }

    _, _, errors = evaluate_records(
        [
            {"gold_quadruples": [implicit_target], "predicted_quadruples": []},
            {"gold_quadruples": [], "predicted_quadruples": [implicit_target]},
        ]
    )

    assert errors[0]["implicit_target_failures"] == [
        {"gold": expected_quadruple, "predicted": None, "reason": "missed_implicit"}
    ]
    assert errors[1]["implicit_target_failures"] == [
        {"gold": None, "predicted": expected_quadruple, "reason": "spurious_implicit"}
    ]


@pytest.mark.parametrize(
    ("gold", "predicted", "unpaired_reason"),
    [
        (
            [q("", "SERVICE#GENERAL", "slow", "negative")] * 2,
            [q("waiter", "SERVICE#GENERAL", "slow", "negative")],
            "missed_implicit",
        ),
        (
            [q("waiter", "SERVICE#GENERAL", "slow", "negative")],
            [q("", "SERVICE#GENERAL", "slow", "negative")] * 2,
            "spurious_implicit",
        ),
    ],
)
def test_implicit_target_mismatch_keeps_any_unpaired_duplicate(
    gold, predicted, unpaired_reason
):
    _, _, errors = evaluate_records(
        [{"gold_quadruples": gold, "predicted_quadruples": predicted}]
    )

    failures = errors[0]["implicit_target_failures"]
    assert sum("wrong_fields" in failure for failure in failures) == 1
    assert [failure.get("reason") for failure in failures if "reason" in failure] == [
        unpaired_reason
    ]


def test_record_accepts_attribute_based_input():
    class Case:
        review_id = "r1"
        sentence_id = "s1"
        text = "Great food."
        gold_quadruples = [q("food", "FOOD#QUALITY", "great", "positive")]
        predicted_quadruples = [q("food", "FOOD#QUALITY", "great", "positive")]

    metrics, _, errors = evaluate_records([Case()])

    assert metrics["f1"] == 1.0
    assert errors == []


def test_missing_and_malformed_generation_are_returned_as_structured_errors():
    metrics, diagnostics, errors = evaluate_records(
        [
            {"gold_quadruples": [q("food", "FOOD#QUALITY", "great", "positive")]},
            {
                "review_id": "r2",
                "sentence_id": "s2",
                "gold_quadruples": [],
                "raw_generation": "{not-json",
            },
        ]
    )

    assert metrics["fn"] == 1
    assert diagnostics["counts"]["invalid_predictions"] >= 2
    assert len(errors) == 2
    all_codes = {issue["code"] for case in errors for issue in case["issues"]}
    assert "missing" in all_codes
    assert "malformed" in all_codes
    assert "missing_generation" in {kind for case in errors for kind in case["error_types"]}
    assert "malformed_generation" in {kind for case in errors for kind in case["error_types"]}


def test_metrics_and_diagnostics_do_not_match_quadruples_across_records():
    same = q("food", "FOOD#QUALITY", "great", "positive")
    metrics, diagnostics, errors = evaluate_records(
        [
            {"gold_quadruples": [same], "predicted_quadruples": []},
            {"gold_quadruples": [], "predicted_quadruples": [same]},
        ]
    )

    assert metrics["tp"] == 0
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert diagnostics["components"]["target"]["tp"] == 0
    assert diagnostics["per_aspect"]["food quality"]["tp"] == 0
    assert diagnostics["per_sentiment"]["positive"]["tp"] == 0
    assert len(errors) == 2


def test_aspect_and_sentiment_accuracy_use_only_aligned_other_fields():
    _, diagnostics, _ = evaluate_records(
        [
            {
                "gold_quadruples": [q("food", "FOOD#QUALITY", "great", "positive")],
                "predicted_quadruples": [q("food", "SERVICE#GENERAL", "great", "positive")],
            },
            {
                "gold_quadruples": [q("food", "FOOD#QUALITY", "great", "positive")],
                "predicted_quadruples": [q("food", "FOOD#QUALITY", "great", "negative")],
            },
        ]
    )

    assert diagnostics["components"]["aspect"]["accuracy"] == 0.0
    assert diagnostics["components"]["sentiment"]["accuracy"] == 0.0
    assert diagnostics["component_accuracy"]["aspect"]["aligned"] == 1
    assert diagnostics["component_accuracy"]["sentiment"]["aligned"] == 1
    assert diagnostics["component_accuracy"]["diagnostic_only"] is True


def test_invalid_sentiment_and_unknown_generation_fields_are_classified():
    _, _, invalid = evaluate_records(
        [
            {
                "gold_quadruples": [],
                "predicted_quadruples": [
                    {"target": "food", "aspect": "FOOD#QUALITY", "opinion": "great", "sentiment": "bullish"}
                ],
            }
        ]
    )
    _, _, unknown = evaluate_records(
        [
            {
                "gold_quadruples": [],
                "raw_generation": {"response": "unrecognized", "detail": "kept"},
            }
        ]
    )

    assert "invalid_generation" in invalid[0]["error_types"]
    assert "unknown_generation_field" in unknown[0]["error_types"]
    assert unknown[0]["raw_generation"] == {"response": "unrecognized", "detail": "kept"}


def test_error_cases_classify_true_positives_missing_extras_and_implicit_opinion_failures():
    matched = q("staff", "SERVICE#GENERAL", "helpful", "positive")
    implicit_gold = q("NULL", "SERVICE#GENERAL", "NULL", "neutral")
    gold_missing = q("meal", "FOOD#QUALITY", "fresh", "positive")
    predicted_extra = q("room", "ROOMS#GENERAL", "quiet", "positive")
    wrong_implicit_opinion = q("NULL", "SERVICE#GENERAL", "rude", "neutral")
    metrics, _, errors = evaluate_records(
        [
            {
                "review_id": "r1",
                "sentence_id": "s1",
                "gold_quadruples": [matched, implicit_gold, gold_missing],
                "predicted_quadruples": [matched, wrong_implicit_opinion],
                "raw_generation": "preserve this output",
                "invalid_predictions": [{"code": "invalid", "message": "bad item"}],
            },
            {"gold_quadruples": [], "predicted_quadruples": [predicted_extra]},
        ]
    )

    first = errors[0]
    assert first["counts"] == {"tp": 1, "fp": 1, "fn": 2}
    assert first["missing_quadruples"]
    assert first["implicit_opinion_failures"]
    assert first["wrong_field_mismatches"][0]["wrong_fields"] == ["opinion"]
    assert first["raw_generation"] == "preserve this output"
    assert first["invalid_predictions"] == [{"code": "invalid", "message": "bad item"}]
    assert "invalid_generation" in first["error_types"]
    assert any(case["extra_quadruples"] for case in errors)
    assert metrics["tp"] == 1
    assert metrics["fn"] == 2
    assert metrics["fp"] == 2
