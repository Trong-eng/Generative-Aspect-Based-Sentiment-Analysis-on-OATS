"""Shared, dependency-light records for the OATS ASQP pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class Quadruple:
    """One ASQP prediction: target, aspect, opinion phrase, and sentiment."""

    target: str
    aspect: str
    opinion: str
    sentiment: str

    def to_dict(self) -> dict[str, str]:
        return {
            "target": self.target,
            "aspect": self.aspect,
            "opinion": self.opinion,
            "sentiment": self.sentiment,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Quadruple":
        if not isinstance(value, Mapping):
            raise ValueError("quadruple record must be a mapping")
        fields = ("target", "aspect", "opinion", "sentiment")
        missing = [name for name in fields if name not in value]
        if missing:
            raise ValueError(f"quadruple is missing required fields: {', '.join(missing)}")
        return cls(**{name: str(value[name]) for name in fields})


@dataclass
class SentenceRecord:
    """A sentence-level ASQP record that retains its source review identity."""

    review_id: str
    sentence_id: str
    text: str
    quadruples: list[Quadruple] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "review_id": self.review_id,
            "sentence_id": self.sentence_id,
            "text": self.text,
            "quadruples": [quadruple.to_dict() for quadruple in self.quadruples],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SentenceRecord":
        if not isinstance(value, Mapping):
            raise ValueError("sentence record must be a mapping")
        required = ("review_id", "sentence_id", "text", "quadruples")
        missing = [name for name in required if name not in value]
        if missing:
            raise ValueError(f"sentence record is missing required fields: {', '.join(missing)}")
        quadruples = value["quadruples"]
        if not isinstance(quadruples, list):
            raise ValueError("sentence record 'quadruples' must be a list")
        return cls(
            review_id=str(value["review_id"]),
            sentence_id=str(value["sentence_id"]),
            text=str(value["text"]),
            quadruples=[Quadruple.from_dict(item) for item in quadruples],
        )


@dataclass
class ReviewRecord:
    """A review and all of its source-linked sentence records."""

    review_id: str
    sentences: list[SentenceRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.review_id = str(self.review_id)
        for sentence in self.sentences:
            if sentence.review_id != self.review_id:
                raise ValueError(
                    f"sentence {sentence.sentence_id!r} belongs to review "
                    f"{sentence.review_id!r}, not {self.review_id!r}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "review_id": self.review_id,
            "sentences": [sentence.to_dict() for sentence in self.sentences],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReviewRecord":
        if not isinstance(value, Mapping):
            raise ValueError("review record must be a mapping")
        if "review_id" not in value or "sentences" not in value:
            raise ValueError("review record requires 'review_id' and 'sentences'")
        sentences = value["sentences"]
        if not isinstance(sentences, list):
            raise ValueError("review record 'sentences' must be a list")
        return cls(
            review_id=str(value["review_id"]),
            sentences=[SentenceRecord.from_dict(item) for item in sentences],
        )
