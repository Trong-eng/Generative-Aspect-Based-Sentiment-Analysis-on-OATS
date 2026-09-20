"""Shared, conservative normalization for generated ASQP quadruples."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from capstone.types import Quadruple


FIELDS = ("target", "aspect", "opinion", "sentiment")
_NULL_VALUES = {"null", "none"}
_SENTIMENT_ALIASES = {
    "positive": "positive",
    "pos": "positive",
    "+": "positive",
    "negative": "negative",
    "neg": "negative",
    "-": "negative",
    "neutral": "neutral",
    "neu": "neutral",
    "conflict": "conflict",
    "mixed": "conflict",
}


class PredictionErrorCode(str, Enum):
    """Machine-readable classes for prediction parsing and validation errors."""

    MISSING = "missing"
    MALFORMED = "malformed"
    INVALID = "invalid"
    UNKNOWN_FIELD = "unknown_field"
    UPSTREAM_INVALID = "upstream_invalid"


@dataclass(frozen=True)
class PredictionIssue:
    """An issue found while normalizing one generated prediction."""

    code: PredictionErrorCode
    message: str
    quadruple_index: int | None = None
    field: str | None = None


def _collapse_whitespace(value: str) -> str:
    return " ".join(value.split())


def _is_null(value: str) -> bool:
    return _collapse_whitespace(value).casefold() in _NULL_VALUES


def _canonical_aspect(value: str) -> str:
    if _is_null(value):
        return "NULL"
    return _collapse_whitespace(value.replace("#", " ")).casefold()


def _canonical_sentiment(value: str) -> str | None:
    if _is_null(value):
        return "NULL"
    return _SENTIMENT_ALIASES.get(_collapse_whitespace(value).casefold())


def _known_aspect_map(known_aspects: Iterable[str] | None) -> dict[str, str] | None:
    if known_aspects is None:
        return None
    result: dict[str, str] = {}
    aspects: Iterable[str] = [known_aspects] if isinstance(known_aspects, str) else known_aspects
    for aspect in aspects:
        if isinstance(aspect, str):
            canonical = _canonical_aspect(aspect)
            result[canonical] = canonical
    return result


def _record_fields(value: Any) -> tuple[dict[str, Any] | None, list[PredictionIssue], list[str]]:
    """Return required fields, issues, and extra field names for a raw tuple."""
    if isinstance(value, Mapping):
        fields = {name: value[name] for name in FIELDS if name in value}
        missing = [name for name in FIELDS if name not in value]
        extras = [str(name) for name in value if name not in FIELDS]
        issues = [
            PredictionIssue(
                PredictionErrorCode.INVALID,
                f"Missing required field '{name}'.",
                field=name,
            )
            for name in missing
        ]
        return fields, issues, extras

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) != len(FIELDS):
            return (
                None,
                [
                    PredictionIssue(
                        PredictionErrorCode.MALFORMED,
                        "A positional quadruple must contain exactly four values.",
                    )
                ],
                [],
            )
        return dict(zip(FIELDS, value)), [], []

    if value is None or isinstance(value, (str, bytes, bytearray, int, float, bool)):
        return (
            None,
            [
                PredictionIssue(
                    PredictionErrorCode.MALFORMED,
                    "A quadruple must be a mapping, a four-value sequence, or an object with named fields.",
                )
            ],
            [],
        )

    fields: dict[str, Any] = {}
    missing = []
    for name in FIELDS:
        try:
            fields[name] = getattr(value, name)
        except (AttributeError, TypeError):
            missing.append(name)
    issues = [
        PredictionIssue(
            PredictionErrorCode.INVALID,
            f"Missing required field '{name}'.",
            field=name,
        )
        for name in missing
    ]
    return fields, issues, []


def _normalize_one(
    value: Any,
    known_aspects: dict[str, str] | None,
    index: int | None = None,
) -> tuple[Quadruple | None, list[PredictionIssue]]:
    fields, issues, extras = _record_fields(value)
    issues = [
        PredictionIssue(issue.code, issue.message, index, issue.field)
        for issue in issues
    ]
    issues.extend(
        PredictionIssue(
            PredictionErrorCode.UNKNOWN_FIELD,
            f"Unknown generated field '{name}'.",
            index,
            name,
        )
        for name in extras
    )
    if fields is None:
        return None, issues

    normalized: dict[str, str] = {}
    for name in FIELDS:
        if name not in fields:
            continue
        raw_value = fields[name]
        if raw_value is None and name != "sentiment":
            normalized[name] = "NULL"
            continue
        if not isinstance(raw_value, str):
            issues.append(
                PredictionIssue(
                    PredictionErrorCode.INVALID,
                    f"Field '{name}' must be a string.",
                    index,
                    name,
                )
            )
            continue
        if name in {"target", "opinion"}:
            collapsed = _collapse_whitespace(raw_value)
            normalized[name] = (
                "NULL"
                if not collapsed or _is_null(collapsed)
                else collapsed.casefold()
            )
        elif name == "aspect":
            normalized[name] = _canonical_aspect(raw_value)
        else:
            canonical_sentiment = _canonical_sentiment(raw_value)
            if canonical_sentiment is None or canonical_sentiment == "NULL":
                issues.append(
                    PredictionIssue(
                        PredictionErrorCode.INVALID,
                        f"Unknown sentiment label '{raw_value.strip()}'.",
                        index,
                        name,
                    )
                )
                continue
            normalized[name] = canonical_sentiment

    if "aspect" in normalized and normalized["aspect"] in {"", "NULL"}:
        issues.append(
            PredictionIssue(
                PredictionErrorCode.INVALID,
                "Aspect label is required and cannot be NULL or empty.",
                index,
                "aspect",
            )
        )

    if len(normalized) != len(FIELDS) or any(
        issue.code is PredictionErrorCode.INVALID for issue in issues
    ):
        return None, issues

    if known_aspects is not None:
        aspect_key = normalized["aspect"]
        canonical_aspect = known_aspects.get(aspect_key)
        if canonical_aspect is None:
            issues.append(
                PredictionIssue(
                    PredictionErrorCode.INVALID,
                    f"Unknown aspect '{normalized['aspect']}'.",
                    index,
                    "aspect",
                )
            )
            return None, issues
        normalized["aspect"] = canonical_aspect

    return Quadruple(**normalized), issues


def normalize_quadruple(
    value: Any,
    known_aspects: Iterable[str] | None = None,
) -> Quadruple | None:
    """Normalize a mapping/object/4-tuple into the project's Quadruple type.

    Repeated whitespace is collapsed and target, opinion, and aspect text is
    case-folded; aspect categories convert ``#`` to a space (underscores
    remain intact). Sentiment aliases map to the canonical
    lowercase four-class labels. ``NULL``, ``null``, ``None``, and Python
    ``None`` are normalized for text fields. If ``known_aspects`` is supplied,
    aspect labels are validated against that canonicalized set. Extra mapping
    keys are ignored by this convenience function; callers needing diagnostics
    should use :func:`normalize_quadruples_with_issues`.
    """
    aspect_map = _known_aspect_map(known_aspects)
    quadruple, _ = _normalize_one(value, aspect_map)
    return quadruple


def _looks_like_quadruple(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(name in value for name in FIELDS)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return len(value) == len(FIELDS) and all(isinstance(item, str) for item in value)
    return all(hasattr(value, name) for name in FIELDS)


def _decode_generation(value: Any) -> tuple[Any, list[PredictionIssue]]:
    """Decode JSON output and common generation wrappers without raising."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return None, [
                PredictionIssue(
                    PredictionErrorCode.MALFORMED,
                    "Generation output is not valid JSON.",
                )
            ]

    if isinstance(value, Mapping) and not any(name in value for name in FIELDS):
        for name in ("predicted_quadruples", "quadruples", "predictions", "output"):
            if name in value:
                metadata_fields = {"review_id", "sentence_id", "text", "id", "metadata"}
                extra_keys = [
                    str(key) for key in value if key != name and key not in metadata_fields
                ]
                issues = [
                    PredictionIssue(
                        PredictionErrorCode.UNKNOWN_FIELD,
                        f"Unknown generation field '{key}'.",
                        field=key,
                    )
                    for key in extra_keys
                ]
                decoded, nested_issues = _decode_generation(value[name])
                return decoded, issues + nested_issues
        metadata_fields = {"review_id", "sentence_id", "text", "id", "metadata"}
        unknown_fields = [str(key) for key in value if key not in metadata_fields]
        issues = [
            PredictionIssue(
                PredictionErrorCode.UNKNOWN_FIELD,
                f"Unknown generation field '{key}'.",
                field=key,
            )
            for key in unknown_fields
        ]
        issues.append(
            PredictionIssue(
                PredictionErrorCode.MISSING,
                "Generation output has no recognized quadruple field.",
            )
        )
        return None, issues
    return value, []


def normalize_quadruples_with_issues(
    values: Any,
    known_aspects: Iterable[str] | None = None,
    *,
    parse_generation: bool = False,
) -> tuple[list[Quadruple], list[PredictionIssue]]:
    """Normalize a collection and retain structured errors for invalid items."""
    pre_issues: list[PredictionIssue] = []
    if parse_generation:
        values, pre_issues = _decode_generation(values)

    if values is None:
        if any(
            issue.code in {PredictionErrorCode.MISSING, PredictionErrorCode.MALFORMED}
            for issue in pre_issues
        ):
            return [], pre_issues
        return [], pre_issues + [
            PredictionIssue(
                PredictionErrorCode.MISSING,
                "No prediction quadruples were supplied.",
            )
        ]

    if _looks_like_quadruple(values):
        items = [values]
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        items = list(values)
    elif isinstance(values, Iterable) and not isinstance(values, (str, bytes, bytearray, Mapping)):
        items = list(values)
    else:
        return [], pre_issues + [
            PredictionIssue(
                PredictionErrorCode.MALFORMED,
                "Prediction quadruples must be a sequence or a single quadruple.",
            )
        ]

    aspect_map = _known_aspect_map(known_aspects)
    normalized: list[Quadruple] = []
    issues = list(pre_issues)
    for index, item in enumerate(items):
        quadruple, item_issues = _normalize_one(item, aspect_map, index)
        issues.extend(item_issues)
        if quadruple is not None:
            normalized.append(quadruple)
    return normalized, issues


def normalize_quadruples(
    values: Any,
    known_aspects: Iterable[str] | None = None,
) -> list[Quadruple]:
    """Normalize a collection of generated quadruples, skipping invalid items."""
    normalized, _ = normalize_quadruples_with_issues(values, known_aspects)
    return normalized
