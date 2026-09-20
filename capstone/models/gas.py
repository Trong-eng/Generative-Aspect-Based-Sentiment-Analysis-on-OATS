"""Extraction-style GAS adaptation for OATS quadruple generation.

The GAS reference defines extraction-style generation of sentiment-element
tuples. This OATS ASQP adaptation serializes each complete quadruple as an
ordered four-string tuple inside a JSON array. The OATS paper does not publish
the exact GAS-T5 target syntax used for its table, so this is a source-aligned
adaptation, not an exact reproduction of that unpublished target format.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from capstone.evaluation.normalization import normalize_quadruple
from capstone.models.base import ParsedGeneration
from capstone.types import Quadruple


def serialize_gas(quadruples: Iterable[Quadruple]) -> str:
    payload = [
        [quad.target, quad.aspect, quad.opinion, quad.sentiment]
        for quad in quadruples
    ]
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _as_quadruple(value: Any) -> tuple[Quadruple | None, str | None]:
    if isinstance(value, dict):
        required = ("target", "aspect", "opinion", "sentiment")
        missing = [key for key in required if key not in value]
        if missing:
            return None, "missing_field"
        if any(key not in required for key in value):
            return None, "unknown_field"
        fields = [value[key] for key in required]
    elif isinstance(value, (list, tuple)) and len(value) == 4:
        fields = list(value)
    else:
        return None, "invalid_quadruple_shape"
    if any(not isinstance(item, str) for item in fields):
        return None, "non_string_field"
    return Quadruple(*fields), None


def _json_items(text: str) -> tuple[list[Any], list[dict[str, str]]]:
    decoder = json.JSONDecoder()
    items: list[Any] = []
    invalid: list[dict[str, str]] = []
    cursor = 0
    while cursor < len(text):
        while cursor < len(text) and (text[cursor].isspace() or text[cursor] in ",;|"):
            cursor += 1
        if cursor >= len(text):
            break
        # [QSEP] is a documented boundary between generated candidates. It is
        # structural text, not an invalid candidate to include in diagnostics.
        if text[cursor:cursor + len("[QSEP]")].casefold() == "[qsep]":
            cursor += len("[QSEP]")
            continue
        if text[cursor] in "[{":
            try:
                value, end = decoder.raw_decode(text, cursor)
            except json.JSONDecodeError:
                next_start = min(
                    [index for index in (text.find("{", cursor + 1), text.find("[", cursor + 1)) if index >= 0]
                    or [len(text)]
                )
                invalid.append({"raw": text[cursor:next_start].strip(), "reason": "malformed_generation"})
                cursor = next_start
                continue
            items.append(value)
            cursor = end
            continue
        next_start = min(
            [index for index in (text.find("{", cursor + 1), text.find("[", cursor + 1)) if index >= 0]
            or [len(text)]
        )
        invalid.append({"raw": text[cursor:next_start].strip(), "reason": "unexpected_text"})
        cursor = next_start
    return items, invalid


def _flatten_candidates(value: Any) -> list[Any]:
    if isinstance(value, list):
        if len(value) == 4 and all(isinstance(item, str) for item in value):
            return [value]
        return list(value)
    return [value]


def parse_gas_generation(text: str, known_aspects: set[str] | None = None) -> ParsedGeneration:
    cleaned = text.replace("<pad>", "").replace("</s>", "").strip()
    items, invalid = _json_items(cleaned)
    quadruples: list[Quadruple] = []
    for item in items:
        # An empty top-level list is the canonical serialization of a sentence
        # with no extracted quadruples, not a malformed candidate.
        if isinstance(item, list) and not item:
            continue
        for candidate in _flatten_candidates(item):
            quadruple, reason = _as_quadruple(candidate)
            if reason:
                invalid.append({"raw": json.dumps(candidate, ensure_ascii=False), "reason": reason})
                continue
            assert quadruple is not None
            normalized = normalize_quadruple(quadruple, known_aspects=known_aspects)
            if normalized is None:
                invalid.append({"raw": json.dumps(candidate, ensure_ascii=False), "reason": "invalid_aspect_category"})
                continue
            quadruples.append(normalized)
    if not items and cleaned and not invalid:
        invalid.append({"raw": cleaned, "reason": "malformed_generation"})
    return ParsedGeneration(quadruples=quadruples, invalid_predictions=invalid)
