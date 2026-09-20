"""Natural-language ASQP target serialization from the PARAPHRASE paper."""

from __future__ import annotations

import re
from typing import Iterable

from capstone.evaluation.normalization import normalize_quadruple
from capstone.models.base import ParsedGeneration
from capstone.types import Quadruple


SENTIMENT_TO_PHRASE = {
    "positive": "great",
    "negative": "bad",
    "neutral": "ok",
    # OATS has a fourth raw label; "mixed" is an explicit extension, not a paper setting.
    "conflict": "mixed",
}
PHRASE_TO_SENTIMENT = {value: key for key, value in SENTIMENT_TO_PHRASE.items()}
_UNKNOWN_SEPARATOR = re.compile(
    r"\[(?!(?:SSEP|QSEP)\])[A-Z][A-Z0-9_]*SEP\]|<[^>\r\n]*SEP[^>\r\n]*>|\|",
    flags=re.IGNORECASE,
)


def serialize_paraphrase(quadruples: Iterable[Quadruple]) -> str:
    segments: list[str] = []
    for quad in quadruples:
        normalized = normalize_quadruple(quad)
        if normalized is None:
            continue
        target = "it" if normalized.target == "NULL" else normalized.target
        opinion = "it" if normalized.opinion == "NULL" else normalized.opinion
        sentiment_phrase = SENTIMENT_TO_PHRASE[normalized.sentiment]
        segments.append(
            f"{normalized.aspect} is {sentiment_phrase} because {target} is {opinion}"
        )
    return " [SSEP] ".join(segments)


def _split_segments(text: str) -> list[str]:
    cleaned = text.replace("</s>", "").replace("<pad>", "")
    return [segment.strip(" \t\r\n;|") for segment in re.split(r"\[SSEP\]|\[QSEP\]|\n+", cleaned, flags=re.IGNORECASE)
            if segment.strip(" \t\r\n;|")]


def _parse_segment(segment: str) -> tuple[Quadruple | None, str | None, str | None]:
    if " because " not in segment.casefold():
        return None, "malformed_generation", None
    left, right = re.split(r"\s+because\s+", segment, maxsplit=1, flags=re.IGNORECASE)
    if " is " not in left.casefold() or " is " not in right.casefold():
        return None, "missing_field", None
    aspect, sentiment_phrase = re.split(r"\s+is\s+", left, maxsplit=1, flags=re.IGNORECASE)
    target, opinion = re.split(r"\s+is\s+", right, maxsplit=1, flags=re.IGNORECASE)
    aspect = aspect.strip()
    sentiment_phrase = sentiment_phrase.strip().casefold()
    target = target.strip()
    opinion = opinion.strip()
    sentiment = PHRASE_TO_SENTIMENT.get(sentiment_phrase)
    if sentiment is None:
        return None, "unknown_sentiment", aspect
    if not aspect or not target or not opinion:
        return None, "missing_field", aspect
    if target.casefold() == "it":
        target = "NULL"
    if opinion.casefold() == "it":
        opinion = "NULL"
    return Quadruple(target, aspect, opinion, sentiment), None, aspect


def parse_paraphrase_generation(
    text: str,
    known_aspects: set[str] | None = None,
) -> ParsedGeneration:
    quadruples: list[Quadruple] = []
    invalid: list[dict[str, str]] = []

    def replace_unexpected_separator(match: re.Match[str]) -> str:
        token = match.group(0)
        invalid.append({"raw": token, "reason": "unexpected_separator"})
        return " [SSEP] "

    parseable_text = _UNKNOWN_SEPARATOR.sub(replace_unexpected_separator, text)
    segments = _split_segments(parseable_text)
    for segment in segments:
        quadruple, reason, aspect = _parse_segment(segment)
        if reason:
            invalid.append({"raw": segment, "reason": reason})
            continue
        assert quadruple is not None
        normalized = normalize_quadruple(quadruple, known_aspects=known_aspects)
        if normalized is None:
            invalid.append({"raw": segment, "reason": "invalid_aspect_category"})
            continue
        quadruples.append(normalized)
    if text.strip() and not segments and not invalid:
        invalid.append({"raw": text, "reason": "malformed_generation"})
    return ParsedGeneration(quadruples=quadruples, invalid_predictions=invalid)
