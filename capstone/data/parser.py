"""Read OATS XML and its sentence-level ASQP text split files."""

from __future__ import annotations

import ast
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from capstone.evaluation.normalization import normalize_quadruple
from capstone.types import Quadruple, ReviewRecord, SentenceRecord


class DataFormatError(ValueError):
    """Raised when a raw OATS file cannot be interpreted without guessing."""


@dataclass(frozen=True)
class DataIssue:
    reason: str
    review_id: str | None = None
    sentence_id: str | None = None
    quadruple: dict[str, object] | None = None
    details: str | None = None
    source_path: str | None = None
    source_line_number: int | None = None

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {"reason": self.reason}
        if self.review_id is not None:
            value["review_id"] = self.review_id
        if self.sentence_id is not None:
            value["sentence_id"] = self.sentence_id
        if self.quadruple is not None:
            value["quadruple"] = self.quadruple
        if self.details is not None:
            value["details"] = self.details
        if self.source_path is not None:
            value["source_path"] = self.source_path
        if self.source_line_number is not None:
            value["source_line_number"] = self.source_line_number
        return value


@dataclass
class ParseResult:
    reviews: list[ReviewRecord]
    issues: list[DataIssue] = field(default_factory=list)


@dataclass(frozen=True)
class OriginalSplitRow:
    text: str
    quadruples: list[Quadruple]
    line_number: int


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _direct_children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in list(element) if _local_name(child.tag) == name]


def _descendants(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in element.iter() if child is not element and _local_name(child.tag) == name]


def _normalized_aspect(value: str) -> str:
    return value.replace("#", " ").strip().lower()


def _is_null(value: str) -> bool:
    return value.strip().casefold() in {"null", "none"}


def _check_offsets(
    opinion: ET.Element,
    text: str,
    review_id: str,
    sentence_id: str,
    issues: list[DataIssue],
) -> None:
    """Record suspicious character spans while keeping the annotation intact."""

    attributes = opinion.attrib
    for value_key, start_key, end_key in (
        ("target", "target_from", "target_to"),
        ("opinion_phrase", "opinion_phrase_from", "opinion_phrase_to"),
    ):
        value = attributes.get(value_key)
        if value is None or _is_null(value):
            continue
        if start_key not in attributes or end_key not in attributes:
            continue
        try:
            start = int(attributes[start_key])
            end = int(attributes[end_key])
        except ValueError:
            issues.append(
                DataIssue(
                    "suspicious_annotation",
                    review_id,
                    sentence_id,
                    dict(attributes),
                    f"non-integer offsets for {value_key}",
                )
            )
            continue
        if start < 0 or end < start or end >= len(text):
            issues.append(
                DataIssue(
                    "suspicious_annotation",
                    review_id,
                    sentence_id,
                    dict(attributes),
                    f"out-of-range offsets for {value_key}",
                )
            )
            continue
        observed = text[start : end + 1]
        if " ".join(observed.split()).casefold() != " ".join(value.split()).casefold():
            issues.append(
                DataIssue(
                    "suspicious_annotation",
                    review_id,
                    sentence_id,
                    dict(attributes),
                    f"text span disagrees with {value_key}",
                )
            )


def _xml_quadruple(
    opinion: ET.Element,
    *,
    review_id: str,
    sentence_id: str,
    text: str,
    issues: list[DataIssue],
) -> Quadruple | None:
    attributes = opinion.attrib
    required = ("category", "polarity", "target", "opinion_phrase")
    missing = [name for name in required if not attributes.get(name, "").strip()]
    if missing:
        issues.append(
            DataIssue(
                "malformed_quadruple",
                review_id,
                sentence_id,
                dict(attributes),
                f"missing required XML attributes: {', '.join(missing)}",
            )
        )
        return None

    _check_offsets(opinion, text, review_id, sentence_id, issues)
    candidate = {
        "target": attributes["target"],
        "aspect": _normalized_aspect(attributes["category"]),
        "opinion": attributes["opinion_phrase"],
        "sentiment": attributes["polarity"],
    }
    quadruple = normalize_quadruple(candidate)
    if quadruple is None:
        issues.append(
            DataIssue(
                "malformed_quadruple",
                review_id,
                sentence_id,
                candidate,
                "quadruple fields could not be normalized",
            )
        )
        return None
    return quadruple


def parse_oats_xml(xml_path: str | Path) -> ParseResult:
    """Parse reviews, sentences, and quads without discarding empty sentences."""

    path = Path(xml_path)
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as error:
        raise DataFormatError(f"could not parse OATS XML at {path}: {error}") from error

    review_nodes = (
        [root]
        if _local_name(root.tag) == "Review"
        else [node for node in root.iter() if _local_name(node.tag) == "Review"]
    )
    reviews: list[ReviewRecord] = []
    issues: list[DataIssue] = []
    for review_index, review_node in enumerate(review_nodes):
        review_id = review_node.get("rid") or review_node.get("id")
        if not review_id:
            review_id = f"missing-review-{review_index}"
            issues.append(
                DataIssue(
                    "malformed_review",
                    review_id,
                    details="review element has no rid or id attribute",
                )
            )

        sentence_nodes = _descendants(review_node, "sentence")
        sentences: list[SentenceRecord] = []
        for sentence_index, sentence_node in enumerate(sentence_nodes):
            sentence_id = sentence_node.get("sid") or f"{review_id}:{sentence_index}"
            text_nodes = _direct_children(sentence_node, "text")
            if not text_nodes:
                text_nodes = _descendants(sentence_node, "text")[:1]
            text = "" if not text_nodes or text_nodes[0].text is None else text_nodes[0].text
            if not text.strip():
                issues.append(
                    DataIssue(
                        "malformed_sentence",
                        review_id,
                        sentence_id,
                        details="sentence has missing or empty text",
                    )
                )

            opinion_nodes = _descendants(sentence_node, "Opinion")
            quadruples = [
                quadruple
                for opinion in opinion_nodes
                if (quadruple := _xml_quadruple(
                    opinion,
                    review_id=review_id,
                    sentence_id=sentence_id,
                    text=text,
                    issues=issues,
                ))
                is not None
            ]
            sentences.append(
                SentenceRecord(review_id, sentence_id, text, quadruples)
            )
        reviews.append(ReviewRecord(review_id, sentences))

    return ParseResult(reviews, issues)


def _parse_split_quadruple(value: object, path: Path, line_number: int) -> Quadruple:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise DataFormatError(
            f"{path}:{line_number}: each ASQP label must have four values "
            "[target, aspect, sentiment, opinion]"
        )
    if not all(isinstance(field, str) for field in value):
        raise DataFormatError(f"{path}:{line_number}: ASQP fields must be strings")
    target, aspect, sentiment, opinion = value
    candidate = {
        "target": target,
        "aspect": _normalized_aspect(aspect),
        "opinion": opinion,
        "sentiment": sentiment,
    }
    quadruple = normalize_quadruple(candidate)
    if quadruple is None:
        raise DataFormatError(f"{path}:{line_number}: invalid ASQP quadruple fields")
    return quadruple


def parse_original_split_file(path: str | Path) -> list[OriginalSplitRow]:
    """Read original ``sentence####quadruples`` records using literal parsing."""

    source = Path(path)
    try:
        content = source.read_text(encoding="utf-8-sig")
    except OSError as error:
        raise DataFormatError(f"could not read OATS split at {source}: {error}") from error

    rows: list[OriginalSplitRow] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            continue
        text, separator, labels = line.partition("####")
        if not separator:
            raise DataFormatError(f"{source}:{line_number}: expected '####' separator")
        try:
            raw_labels = ast.literal_eval(labels.strip())
        except (SyntaxError, ValueError) as error:
            raise DataFormatError(f"{source}:{line_number}: invalid quadruple list: {error}") from error
        if not isinstance(raw_labels, list):
            raise DataFormatError(f"{source}:{line_number}: quadruple payload must be a list")
        rows.append(
            OriginalSplitRow(
                text=text,
                quadruples=[
                    _parse_split_quadruple(value, source, line_number)
                    for value in raw_labels
                ],
                line_number=line_number,
            )
        )
    return rows


def _alignment_text(text: str) -> str:
    return " ".join(text.split())


def _map_original_sentiment(quadruple: Quadruple, sentiment_mode: str) -> Quadruple:
    if sentiment_mode not in {"raw_4_class", "paper_compatible"}:
        raise ValueError("sentiment_mode must be 'raw_4_class' or 'paper_compatible'")
    # No ASQP-specific paper mapping is established in the available source.
    # Keep conflict explicit until a verified mapping is supplied.
    return quadruple


def load_original_splits(
    reviews: Sequence[ReviewRecord],
    split_files: Mapping[str, str | Path],
    *,
    sentiment_mode: str = "raw_4_class",
) -> dict[str, list[ReviewRecord]]:
    """Align TXT rows in source order to XML IDs and group rows by their review.

    Original TXT files contain no review IDs. This method attaches each row to
    an unassigned XML sentence with matching text, preserving the XML review
    link without assuming that the three split files follow source order. If
    repeated sentence text cannot be disambiguated by its labels, it fails
    rather than inventing a review association.
    """

    if sentiment_mode not in {"raw_4_class", "paper_compatible"}:
        raise ValueError("sentiment_mode must be 'raw_4_class' or 'paper_compatible'")
    flattened = [sentence for review in reviews for sentence in review.sentences]
    positions_by_text: dict[str, list[int]] = defaultdict(list)
    for index, sentence in enumerate(flattened):
        positions_by_text[_alignment_text(sentence.text)].append(index)

    consumed_positions: set[int] = set()
    splits: dict[str, list[ReviewRecord]] = {}
    for split_name in ("train", "dev", "test"):
        if split_name not in split_files:
            raise DataFormatError(f"original split files are missing {split_name!r}")
        source = Path(split_files[split_name])
        rows = parse_original_split_file(source)
        grouped: dict[str, list[SentenceRecord]] = {}
        order: list[str] = []
        for row in rows:
            key = _alignment_text(row.text)
            candidates = [
                position
                for position in positions_by_text.get(key, [])
                if position not in consumed_positions
            ]
            if not candidates:
                raise DataFormatError(
                    f"{source}:{row.line_number}: sentence did not match an "
                    "unassigned sentence in the selected OATS XML"
                )
            if len(candidates) == 1:
                position = candidates[0]
            else:
                row_quadruples = tuple(
                    (q.target, q.aspect, q.opinion, q.sentiment)
                    for q in row.quadruples
                )
                label_matches = [
                    position
                    for position in candidates
                    if tuple(
                        (q.target, q.aspect, q.opinion, q.sentiment)
                        for q in flattened[position].quadruples
                    )
                    == row_quadruples
                ]
                if len(label_matches) != 1:
                    raise DataFormatError(
                        f"{source}:{row.line_number}: sentence text matches multiple "
                        "unassigned XML sentences and labels do not identify one uniquely"
                    )
                position = label_matches[0]
            xml_sentence = flattened[position]
            consumed_positions.add(position)
            if xml_sentence.review_id not in grouped:
                grouped[xml_sentence.review_id] = []
                order.append(xml_sentence.review_id)
            grouped[xml_sentence.review_id].append(
                SentenceRecord(
                    review_id=xml_sentence.review_id,
                    sentence_id=xml_sentence.sentence_id,
                    text=row.text,
                    quadruples=[
                        _map_original_sentiment(q, sentiment_mode)
                        for q in row.quadruples
                    ],
                )
            )
        splits[split_name] = [ReviewRecord(review_id, grouped[review_id]) for review_id in order]
    return splits
