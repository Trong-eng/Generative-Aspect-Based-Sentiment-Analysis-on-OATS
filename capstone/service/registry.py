"""Checkpoint registry for the prediction service.

Maps a (domain, model_type) pair to an experiment directory, loads each model at
most once, and turns every loading failure into a `ModelNotReady` the API can
report as `MODEL_NOT_READY` instead of a stack trace.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from capstone.config import DOMAINS, MODEL_TYPES

LOGGER = logging.getLogger(__name__)

DEFAULT_SERVING_CONFIG_ENV = "ASQP_SERVING_CONFIG"


class ModelNotReady(RuntimeError):
    """No usable checkpoint for the requested (domain, model_type)."""


@dataclass(frozen=True)
class ModelEntry:
    domain: str
    model_type: str
    checkpoint: Path
    config_path: Path | None = None

    @property
    def key(self) -> tuple[str, str]:
        return (self.domain, self.model_type)


def _require_yaml() -> Any:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("PyYAML is required to load the serving config; install requirements.txt") from exc
    return yaml


def load_serving_entries(path: str | Path) -> list[ModelEntry]:
    """Read a serving YAML: `models: [{domain, model_type, checkpoint, config?}, ...]`."""
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Serving config not found: {config_path}")
    yaml = _require_yaml()
    with config_path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    if not isinstance(document, dict) or not isinstance(document.get("models"), list):
        raise ValueError(f"Serving config must be a mapping with a 'models' list: {config_path}")

    root = config_path.parent
    entries: list[ModelEntry] = []
    seen: set[tuple[str, str]] = set()
    for index, raw in enumerate(document["models"]):
        if not isinstance(raw, Mapping):
            raise ValueError(f"models[{index}] must be a mapping in {config_path}")
        domain = str(raw.get("domain", "")).strip()
        model_type = str(raw.get("model_type", "")).strip()
        checkpoint = raw.get("checkpoint")
        if domain not in DOMAINS:
            raise ValueError(f"models[{index}].domain must be one of {sorted(DOMAINS)}")
        if model_type not in MODEL_TYPES:
            raise ValueError(f"models[{index}].model_type must be one of {sorted(MODEL_TYPES)}")
        if not checkpoint:
            raise ValueError(f"models[{index}].checkpoint is required")
        if (domain, model_type) in seen:
            raise ValueError(f"duplicate entry for {domain}/{model_type} in {config_path}")
        seen.add((domain, model_type))

        def _resolve(value: Any) -> Path:
            candidate = Path(str(value)).expanduser()
            return candidate if candidate.is_absolute() else (root / candidate)

        config_value = raw.get("config")
        entries.append(
            ModelEntry(
                domain=domain,
                model_type=model_type,
                checkpoint=_resolve(checkpoint),
                config_path=_resolve(config_value) if config_value else None,
            )
        )
    return entries


@dataclass
class ModelRegistry:
    """Lazily loads and caches one predictor per (domain, model_type)."""

    entries: list[ModelEntry] = field(default_factory=list)
    #: Injected in tests; defaults to `capstone.predict.load_predictor`.
    loader: Callable[[Path, str, Path | None], Any] | None = None
    _cache: dict[tuple[str, str], Any] = field(default_factory=dict, init=False, repr=False)
    _failures: dict[tuple[str, str], str] = field(default_factory=dict, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, **kwargs: Any) -> "ModelRegistry":
        """Build from `ASQP_SERVING_CONFIG`; an unset variable yields an empty registry."""
        source = env if env is not None else os.environ
        path = source.get(DEFAULT_SERVING_CONFIG_ENV, "").strip()
        if not path:
            LOGGER.warning(
                "%s is not set; the service will answer MODEL_NOT_READY for every request",
                DEFAULT_SERVING_CONFIG_ENV,
            )
            return cls(entries=[], **kwargs)
        return cls(entries=load_serving_entries(path), **kwargs)

    def entry_for(self, domain: str, model_type: str) -> ModelEntry | None:
        return next((entry for entry in self.entries if entry.key == (domain, model_type)), None)

    def describe(self) -> list[dict[str, Any]]:
        """Configured pairs and whether their checkpoint path currently exists."""
        return [
            {
                "domain": entry.domain,
                "model_type": entry.model_type,
                "checkpoint": str(entry.checkpoint),
                "checkpoint_exists": entry.checkpoint.exists(),
                "loaded": entry.key in self._cache,
            }
            for entry in self.entries
        ]

    def get(self, domain: str, model_type: str) -> Any:
        """Return a loaded predictor, raising ModelNotReady when it cannot be served."""
        key = (domain, model_type)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        entry = self.entry_for(domain, model_type)
        if entry is None:
            raise ModelNotReady(f"No checkpoint is configured for {domain}/{model_type}.")

        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
            previous_failure = self._failures.get(key)
            if previous_failure is not None:
                raise ModelNotReady(previous_failure)
            if not entry.checkpoint.exists():
                message = f"Checkpoint for {domain}/{model_type} is not available on this server."
                self._failures[key] = message
                raise ModelNotReady(message)
            loader = self.loader or _default_loader
            try:
                predictor = loader(entry.checkpoint, entry.model_type, entry.config_path)
            except Exception as error:  # noqa: BLE001 - every failure becomes MODEL_NOT_READY
                # Keep the real cause in the log; the client only learns the model is unavailable.
                LOGGER.exception("Failed to load %s/%s from %s", domain, model_type, entry.checkpoint)
                message = f"Checkpoint for {domain}/{model_type} could not be loaded: {type(error).__name__}."
                self._failures[key] = message
                raise ModelNotReady(message) from error
            self._cache[key] = predictor
            return predictor


def _default_loader(checkpoint: Path, model_type: str, config_path: Path | None) -> Any:
    from capstone.predict import load_predictor

    return load_predictor(checkpoint, model_type, config_path)
