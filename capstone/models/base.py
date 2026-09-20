"""Lazy-loading T5-compatible generator shared by GAS and Paraphrase."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class ParsedGeneration:
    quadruples: list[Any] = field(default_factory=list)
    invalid_predictions: list[dict[str, Any]] = field(default_factory=list)


def resolve_checkpoint(path: str | Path) -> Path:
    """Resolve a model checkpoint or experiment directory to its best checkpoint."""
    candidate = Path(path).expanduser()
    selection_file = candidate / "best_checkpoint.json"
    if not selection_file.is_file():
        return candidate
    try:
        selection = json.loads(selection_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read best-checkpoint selection {selection_file}: {error}") from error
    selected = selection.get("path") if isinstance(selection, dict) else None
    if not selected:
        raise ValueError(f"Best-checkpoint selection has no path: {selection_file}")
    resolved = Path(selected).expanduser()
    return resolved if resolved.is_absolute() else (candidate / resolved)


class Seq2SeqGenerator:
    """Thin wrapper that keeps heavyweight ML imports out of data-only use."""

    def __init__(self, tokenizer: Any, model: Any, device: Any = None):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        if device is not None:
            self.model.to(device)

    @classmethod
    def from_pretrained(cls, model_name: str, tokenizer_name: str | None = None, device: Any = None):
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on local environment
            raise RuntimeError("Training/inference requires torch and transformers from requirements.txt") from exc
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        selected_device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return cls(tokenizer, model, selected_device)

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path, device: Any = None):
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on local environment
            raise RuntimeError("Training/inference requires torch and transformers from requirements.txt") from exc
        checkpoint_path = resolve_checkpoint(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
        model = AutoModelForSeq2SeqLM.from_pretrained(str(checkpoint_path))
        selected_device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return cls(tokenizer, model, selected_device)

    def generate(
        self,
        texts: Iterable[str],
        *,
        max_input_length: int = 256,
        num_beams: int = 1,
        max_new_tokens: int = 256,
        early_stopping: bool = False,
    ) -> list[str]:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on local environment
            raise RuntimeError("Inference requires torch from requirements.txt") from exc

        batch = list(texts)
        if not batch:
            return []
        encoded = self.tokenizer(
            batch,
            max_length=max_input_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        if self.device is not None:
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
        self.model.eval()
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded.get("attention_mask"),
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                early_stopping=early_stopping,
            )
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)
