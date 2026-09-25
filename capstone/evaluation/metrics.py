"""Exact-match and diagnostic metrics for ASQP quadruple predictions."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from capstone.types import Quadruple

from .normalization import (
    PredictionErrorCode,
    PredictionIssue,
    normalize_quadruples,
    normalize_quadruples_with_issues,
)


_MISSING = object()
_COMPONENTS = ("target", "aspect", "opinion", "sentiment")
_COUNT_BUCKETS = ("0", "1", "2", "3+")


def _quadruple_key(quadruple: Quadruple) -> tuple[str, str, str, str]:
    return (
        quadruple.target,
        quadruple.aspect,
        quadruple.opinion,
        quadruple.sentiment,
    )


def _scores(tp: int, fp: int, fn: int) -> dict[str, int | float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _counts(gold: Iterable[Quadruple], predicted: Iterable[Quadruple]) -> tuple[int, int, int]:
    gold_counts = Counter(_quadruple_key(item) for item in gold)
    predicted_counts = Counter(_quadruple_key(item) for item in predicted)
    tp = sum((gold_counts & predicted_counts).values())
    return tp, sum(predicted_counts.values()) - tp, sum(gold_counts.values()) - tp


def evaluate_quadruples(
    gold: Iterable[Any] | None,
    predicted: Iterable[Any] | None,
) -> dict[str, int | float]:
    """Compute exact quadruple micro P/R/F1 with duplicate-safe matching."""
    normalized_gold = normalize_quadruples(gold)
    normalized_predicted = normalize_quadruples(predicted)
    return _scores(*_counts(normalized_gold, normalized_predicted))


def _get(record: Any, name: str, default: Any = _MISSING) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    try:
        return getattr(record, name)
    except (AttributeError, TypeError):
        return default


def _count_bucket(count: int) -> str:
    return str(count) if count < 3 else "3+"


def _is_implicit_target(target: str) -> bool:
    return target.strip().casefold() in {"", "null", "none", "implicit"}


def _is_implicit(value: str) -> bool:
    return value.strip().casefold() in {"", "null", "none", "implicit"}


def _add_counts(target: list[int], counts: tuple[int, int, int]) -> None:
    target[0] += counts[0]
    target[1] += counts[1]
    target[2] += counts[2]


def _metric_map(values: Mapping[str, list[int]]) -> dict[str, dict[str, int | float]]:
    return {key: _scores(*counts) for key, counts in sorted(values.items())}


def _quadruple_dict(quadruple: Quadruple) -> dict[str, str]:
    return {field: getattr(quadruple, field) for field in _COMPONENTS}


def _component_accuracy_counts(
    gold: Sequence[Quadruple],
    predicted: Sequence[Quadruple],
    component: str,
) -> tuple[int, int]:
    """Return correctness/aligned counts after matching the other three fields."""
    other_fields = tuple(field for field in _COMPONENTS if field != component)
    gold_groups: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    predicted_groups: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    for item in gold:
        key = tuple(getattr(item, field) for field in other_fields)
        gold_groups[key][getattr(item, component)] += 1
    for item in predicted:
        key = tuple(getattr(item, field) for field in other_fields)
        predicted_groups[key][getattr(item, component)] += 1

    correct = 0
    aligned = 0
    for key in set(gold_groups) | set(predicted_groups):
        gold_values = gold_groups[key]
        predicted_values = predicted_groups[key]
        aligned += min(sum(gold_values.values()), sum(predicted_values.values()))
        correct += sum((gold_values & predicted_values).values())
    return correct, aligned


def _issue_dict(issue: PredictionIssue) -> dict[str, Any]:
    return {
        "code": issue.code.value,
        "message": issue.message,
        "quadruple_index": issue.quadruple_index,
        "field": issue.field,
    }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _upstream_issues(value: Any) -> list[PredictionIssue]:
    if value is _MISSING or value is None:
        return []
    if isinstance(value, (list, tuple, Mapping)) and not value:
        return []
    values = value if isinstance(value, (list, tuple)) else [value]
    issues = []
    for index, item in enumerate(values):
        if isinstance(item, Mapping):
            reason = str(item.get("reason", ""))
            reason_codes = {
                "malformed_generation": PredictionErrorCode.MALFORMED,
                "unexpected_text": PredictionErrorCode.MALFORMED,
                "unexpected_separator": PredictionErrorCode.MALFORMED,
                "invalid_quadruple_shape": PredictionErrorCode.MALFORMED,
                "unknown_field": PredictionErrorCode.UNKNOWN_FIELD,
                "missing_field": PredictionErrorCode.INVALID,
                "non_string_field": PredictionErrorCode.INVALID,
                "invalid_aspect_category": PredictionErrorCode.INVALID,
                "unknown_sentiment": PredictionErrorCode.INVALID,
            }
            code_value = item.get(
                "code",
                reason_codes.get(reason, PredictionErrorCode.UPSTREAM_INVALID.value),
            )
            if isinstance(code_value, PredictionErrorCode):
                code = code_value
            else:
                try:
                    code = PredictionErrorCode(str(code_value))
                except ValueError:
                    code = PredictionErrorCode.UPSTREAM_INVALID
            message = str(
                item.get(
                    "message",
                    item.get(
                        "error",
                        f"Upstream parser issue ({reason}): {item.get('raw', '')}".strip(),
                    ),
                )
            )
            field = item.get("field")
            quadruple_index = item.get("quadruple_index", item.get("index", index))
            issues.append(
                PredictionIssue(
                    code,
                    message,
                    quadruple_index if isinstance(quadruple_index, int) else None,
                    str(field) if field is not None else None,
                )
            )
        else:
            issues.append(
                PredictionIssue(
                    PredictionErrorCode.UPSTREAM_INVALID,
                    str(item),
                    index,
                )
            )
    return issues


def _record_issue_case(
    record: Any,
    issues: Sequence[PredictionIssue],
    raw_generation: Any,
    invalid_predictions: Any = _MISSING,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "review_id": _json_safe(_get(record, "review_id", None)),
        "sentence_id": _json_safe(_get(record, "sentence_id", None)),
        "issues": [_issue_dict(issue) for issue in issues],
    }
    text = _get(record, "text", _MISSING)
    if text is not _MISSING:
        result["text"] = _json_safe(text)
    if raw_generation is not _MISSING:
        result["raw_generation"] = _json_safe(raw_generation)
    if invalid_predictions is not _MISSING:
        result["invalid_predictions"] = _json_safe(invalid_predictions)
    return result


def _generation_error_type(issue: PredictionIssue) -> str:
    if issue.code is PredictionErrorCode.MISSING:
        return "missing_generation"
    if issue.code is PredictionErrorCode.MALFORMED:
        return "malformed_generation"
    if issue.code is PredictionErrorCode.UNKNOWN_FIELD:
        return "unknown_generation_field"
    return "invalid_generation"


def _classify_quadruple_errors(
    gold: Sequence[Quadruple], predicted: Sequence[Quadruple]
) -> tuple[list[Quadruple], list[Quadruple], list[dict[str, Any]]]:
    predicted_remaining = Counter(_quadruple_key(item) for item in predicted)
    false_negatives: list[Quadruple] = []
    true_positives: list[Quadruple] = []
    for item in gold:
        key = _quadruple_key(item)
        if predicted_remaining[key]:
            predicted_remaining[key] -= 1
            true_positives.append(item)
        else:
            false_negatives.append(item)

    gold_remaining = Counter(_quadruple_key(item) for item in gold)
    false_positives: list[Quadruple] = []
    for item in predicted:
        key = _quadruple_key(item)
        if gold_remaining[key]:
            gold_remaining[key] -= 1
        else:
            false_positives.append(item)

    unmatched_gold = list(false_negatives)
    unmatched_predicted = list(false_positives)
    mismatches: list[dict[str, Any]] = []
    while unmatched_gold and unmatched_predicted:
        best: tuple[int, int, int, list[str]] | None = None
        for gold_index, gold_item in enumerate(unmatched_gold):
            for predicted_index, predicted_item in enumerate(unmatched_predicted):
                wrong_fields = [
                    field
                    for field in _COMPONENTS
                    if getattr(gold_item, field) != getattr(predicted_item, field)
                ]
                candidate = (len(wrong_fields), gold_index, predicted_index, wrong_fields)
                if best is None or candidate[:3] < best[:3]:
                    best = candidate
        assert best is not None
        _, gold_index, predicted_index, wrong_fields = best
        gold_item = unmatched_gold.pop(gold_index)
        predicted_item = unmatched_predicted.pop(predicted_index)
        mismatch: dict[str, Any] = {
            "gold": _quadruple_dict(gold_item),
            "predicted": _quadruple_dict(predicted_item),
            "wrong_fields": wrong_fields,
        }
        if _is_implicit(gold_item.target) != _is_implicit(predicted_item.target):
            mismatch["implicit_target_failure"] = True
        if _is_implicit(gold_item.opinion) != _is_implicit(predicted_item.opinion):
            mismatch["implicit_opinion_failure"] = True
        mismatches.append(mismatch)
    return true_positives, false_positives, {
        "false_negatives": false_negatives,
        "mismatches": mismatches,
    }


def _as_cases(cases: Any) -> list[Any]:
    if cases is None:
        return []
    if isinstance(cases, Mapping) or not isinstance(cases, Iterable):
        return [cases]
    if isinstance(cases, (str, bytes, bytearray)):
        return [cases]
    return list(cases)


def evaluate_records(
    cases: Iterable[Any] | Any,
    known_aspects: Iterable[str] | None = None,
) -> tuple[dict[str, int | float], dict[str, Any], list[dict[str, Any]]]:
    """Evaluate case records and return exact metrics, diagnostics, and errors.

    Each case may be a mapping or an object with ``gold_quadruples`` and
    ``predicted_quadruples``. ``raw_generation`` is decoded when the explicit
    prediction field is absent. Missing and malformed predictions are recorded
    as structured ``error_cases``; they are treated as an empty prediction list
    so gold items remain false negatives.
    """
    if known_aspects is not None and not isinstance(known_aspects, str):
        known_aspects = tuple(known_aspects)
    exact_counts = [0, 0, 0]
    component_counts: dict[str, list[int]] = {name: [0, 0, 0] for name in _COMPONENTS}
    component_accuracy_counts = {name: [0, 0] for name in ("aspect", "sentiment")}
    explicitness_counts: dict[str, dict[str, list[int]]] = {
        field: {group: [0, 0, 0] for group in ("explicit", "implicit")}
        for field in ("target", "opinion")
    }
    aspect_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    sentiment_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    bucket_counts: dict[str, list[int]] = {
        bucket: [0, 0, 0] for bucket in _COUNT_BUCKETS
    }
    bucket_example_counts = {bucket: 0 for bucket in _COUNT_BUCKETS}
    bucket_exact_examples = {bucket: 0 for bucket in _COUNT_BUCKETS}
    errors: list[dict[str, Any]] = []
    case_count = 0
    invalid_prediction_count = 0
    exact_match_examples = 0

    for case in _as_cases(cases):
        case_count += 1
        if case is None or isinstance(case, (str, bytes, bytearray, int, float, bool)):
            gold_raw = []
            prediction_raw = _MISSING
            raw_generation = _MISSING
            issues = [
                PredictionIssue(
                    PredictionErrorCode.MALFORMED,
                    "Evaluation case must be a mapping or object with case fields.",
                )
            ]
        else:
            gold_raw = _get(case, "gold_quadruples", _MISSING)
            if gold_raw is _MISSING:
                gold_raw = _get(case, "quadruples", [])
            prediction_raw = _get(case, "predicted_quadruples", _MISSING)
            raw_generation = _get(case, "raw_generation", _MISSING)
            issues = []
            invalid_predictions = _get(case, "invalid_predictions", _MISSING)

        gold = normalize_quadruples(gold_raw if gold_raw is not _MISSING else [])
        if prediction_raw is _MISSING:
            if raw_generation is _MISSING:
                predicted, prediction_issues = normalize_quadruples_with_issues(None, known_aspects)
            else:
                predicted, prediction_issues = normalize_quadruples_with_issues(
                    raw_generation,
                    known_aspects,
                    parse_generation=True,
                )
        else:
            predicted, prediction_issues = normalize_quadruples_with_issues(
                prediction_raw,
                known_aspects,
                parse_generation=isinstance(prediction_raw, str),
            )
        issues.extend(prediction_issues)
        if case is None or isinstance(case, (str, bytes, bytearray, int, float, bool)):
            invalid_predictions = _MISSING
        issues.extend(_upstream_issues(invalid_predictions))
        invalid_prediction_count += sum(
            1 for issue in issues if issue.code is not PredictionErrorCode.UNKNOWN_FIELD
        )

        case_counts = _counts(gold, predicted)
        _add_counts(exact_counts, case_counts)
        tp, fp, fn = case_counts
        if tp == len(gold) == len(predicted):
            exact_match_examples += 1

        bucket = _count_bucket(len(gold))
        _add_counts(bucket_counts[bucket], case_counts)
        bucket_example_counts[bucket] += 1
        if tp == len(gold) == len(predicted):
            bucket_exact_examples[bucket] += 1

        for component in _COMPONENTS:
            gold_values = Counter(getattr(item, component) for item in gold)
            predicted_values = Counter(getattr(item, component) for item in predicted)
            component_tp = sum((gold_values & predicted_values).values())
            _add_counts(
                component_counts[component],
                (
                    component_tp,
                    sum(predicted_values.values()) - component_tp,
                    sum(gold_values.values()) - component_tp,
                ),
            )

        for component in ("aspect", "sentiment"):
            correct, aligned = _component_accuracy_counts(gold, predicted, component)
            component_accuracy_counts[component][0] += correct
            component_accuracy_counts[component][1] += aligned

        for field in ("target", "opinion"):
            for group in ("explicit", "implicit"):
                gold_group = [
                    item
                    for item in gold
                    if _is_implicit(getattr(item, field)) is (group == "implicit")
                ]
                predicted_group = [
                    item
                    for item in predicted
                    if _is_implicit(getattr(item, field)) is (group == "implicit")
                ]
                _add_counts(
                    explicitness_counts[field][group],
                    _counts(gold_group, predicted_group),
                )

        for aspect in {item.aspect for item in gold} | {item.aspect for item in predicted}:
            _add_counts(
                aspect_counts[aspect],
                _counts(
                    [item for item in gold if item.aspect == aspect],
                    [item for item in predicted if item.aspect == aspect],
                ),
            )
        for sentiment in {item.sentiment for item in gold} | {
            item.sentiment for item in predicted
        }:
            _add_counts(
                sentiment_counts[sentiment],
                _counts(
                    [item for item in gold if item.sentiment == sentiment],
                    [item for item in predicted if item.sentiment == sentiment],
                ),
            )

        true_positives, false_positives, classified = _classify_quadruple_errors(
            gold, predicted
        )
        false_negatives = classified["false_negatives"]
        mismatches = classified["mismatches"]
        implicit_target_failures: list[dict[str, Any]] = []
        implicit_opinion_failures: list[dict[str, Any]] = []
        for mismatch in mismatches:
            if mismatch.get("implicit_target_failure"):
                implicit_target_failures.append(mismatch)
            if mismatch.get("implicit_opinion_failure"):
                implicit_opinion_failures.append(mismatch)
        implicit_target_mismatch_gold = Counter(
            tuple(mismatch["gold"][field] for field in _COMPONENTS)
            for mismatch in mismatches
            if mismatch.get("implicit_target_failure")
        )
        implicit_target_mismatch_predicted = Counter(
            tuple(mismatch["predicted"][field] for field in _COMPONENTS)
            for mismatch in mismatches
            if mismatch.get("implicit_target_failure")
        )
        for item in false_negatives:
            key = _quadruple_key(item)
            if implicit_target_mismatch_gold[key]:
                implicit_target_mismatch_gold[key] -= 1
            elif _is_implicit(item.target):
                implicit_target_failures.append(
                    {"gold": _quadruple_dict(item), "predicted": None, "reason": "missed_implicit"}
                )
            if _is_implicit(item.opinion):
                implicit_opinion_failures.append(
                    {"gold": _quadruple_dict(item), "predicted": None, "reason": "missed_implicit"}
                )
        for item in false_positives:
            key = _quadruple_key(item)
            if implicit_target_mismatch_predicted[key]:
                implicit_target_mismatch_predicted[key] -= 1
            elif _is_implicit(item.target):
                implicit_target_failures.append(
                    {"gold": None, "predicted": _quadruple_dict(item), "reason": "spurious_implicit"}
                )
            if _is_implicit(item.opinion):
                implicit_opinion_failures.append(
                    {"gold": None, "predicted": _quadruple_dict(item), "reason": "spurious_implicit"}
                )

        error_types: list[str] = []
        if mismatches:
            error_types.append("wrong_field_mismatch")
        if false_negatives:
            error_types.append("missing_quadruple")
        if false_positives:
            error_types.append("extra_quadruple")
        if implicit_target_failures:
            error_types.append("implicit_target_failure")
        if implicit_opinion_failures:
            error_types.append("implicit_opinion_failure")
        for issue in issues:
            error_type = _generation_error_type(issue)
            if error_type not in error_types:
                error_types.append(error_type)

        if error_types:
            error_case = _record_issue_case(
                case,
                issues,
                raw_generation,
                invalid_predictions,
            )
            error_case.update(
                {
                    "counts": {"tp": tp, "fp": fp, "fn": fn},
                    "true_positive_quadruples": [
                        _quadruple_dict(item) for item in true_positives
                    ],
                    "false_positive_quadruples": [
                        _quadruple_dict(item) for item in false_positives
                    ],
                    "false_negative_quadruples": [
                        _quadruple_dict(item) for item in false_negatives
                    ],
                    "wrong_field_mismatches": mismatches,
                    "missing_quadruples": [
                        _quadruple_dict(item) for item in false_negatives
                    ],
                    "extra_quadruples": [
                        _quadruple_dict(item) for item in false_positives
                    ],
                    "implicit_target_failures": implicit_target_failures,
                    "implicit_opinion_failures": implicit_opinion_failures,
                    "generation_errors": [_issue_dict(issue) for issue in issues],
                    "error_types": error_types,
                }
            )
            errors.append(error_case)

    metrics = _scores(*exact_counts)
    component_metrics = _metric_map(component_counts)
    component_accuracy: dict[str, Any] = {
        "definition": (
            "Correctness among a maximum one-to-one alignment of quadruples whose "
            "other three fields match; diagnostic only, not end-to-end accuracy."
        ),
        "diagnostic_only": True,
    }
    for component in ("aspect", "sentiment"):
        correct, aligned = component_accuracy_counts[component]
        accuracy = correct / aligned if aligned else 0.0
        component_metrics[component]["accuracy"] = accuracy
        component_accuracy[component] = {
            "correct": correct,
            "aligned": aligned,
            "accuracy": accuracy,
        }

    diagnostics: dict[str, Any] = {
        "components": component_metrics,
        "component_accuracy": component_accuracy,
        "explicit_implicit": {
            field: _metric_map(groups)
            for field, groups in explicitness_counts.items()
        },
        "per_aspect": _metric_map(aspect_counts),
        "per_sentiment": _metric_map(sentiment_counts),
        "by_quadruple_count": {
            bucket: {
                **_scores(*bucket_counts[bucket]),
                "example_count": bucket_example_counts[bucket],
                "example_accuracy": (
                    bucket_exact_examples[bucket] / bucket_example_counts[bucket]
                    if bucket_example_counts[bucket]
                    else 0.0
                ),
            }
            for bucket in _COUNT_BUCKETS
        },
        "counts": {
            "examples": case_count,
            "gold_quadruples": exact_counts[0] + exact_counts[2],
            "predicted_quadruples": exact_counts[0] + exact_counts[1],
            "invalid_predictions": invalid_prediction_count,
            "exact_match_examples": exact_match_examples,
            "example_accuracy": exact_match_examples / case_count if case_count else 0.0,
        },
    }
    diagnostics["example_accuracy"] = diagnostics["counts"]["example_accuracy"]

    return metrics, diagnostics, errors


def analyze_errors(
    cases: Iterable[Any] | Any,
    known_aspects: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Return only the structured error cases from an evaluation input."""
    return evaluate_records(cases, known_aspects)[2]
