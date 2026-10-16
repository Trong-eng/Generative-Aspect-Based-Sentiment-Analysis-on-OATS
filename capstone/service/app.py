"""HTTP service exposing the ASQP models to the frontend.

Implements the contract documented in `frontend/README.md`:
    POST /predict        {text, domain, model_type}
    POST /predict/batch  {domain, model_type, items: [{id, text}]}
Errors always use {"error": {"code", "message"}}; internal details stay in the log.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from capstone.config import DOMAINS, MODEL_TYPES
from capstone.service.registry import ModelNotReady, ModelRegistry

LOGGER = logging.getLogger(__name__)

MAX_BATCH_ITEMS = 200
MAX_TEXT_LENGTH = 5000

Domain = Literal["amazon_ff", "coursera", "hotels"]
ModelType = Literal["gas", "paraphrase"]

# Keep the API's accepted values and the pipeline's canonical sets from drifting apart.
assert set(Domain.__args__) == DOMAINS  # type: ignore[attr-defined]
assert set(ModelType.__args__) == MODEL_TYPES  # type: ignore[attr-defined]


class PredictRequest(BaseModel):
    text: str = Field(max_length=MAX_TEXT_LENGTH)
    domain: Domain
    model_type: ModelType

    @field_validator("text")
    @classmethod
    def _non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be empty or whitespace only")
        return value


class BatchItem(BaseModel):
    id: str = Field(min_length=1, max_length=200)
    text: str = Field(max_length=MAX_TEXT_LENGTH)


class BatchPredictRequest(BaseModel):
    domain: Domain
    model_type: ModelType
    items: list[BatchItem] = Field(min_length=1, max_length=MAX_BATCH_ITEMS)

    @field_validator("items")
    @classmethod
    def _unique_ids(cls, items: list[BatchItem]) -> list[BatchItem]:
        ids = [item.id for item in items]
        if len(ids) != len(set(ids)):
            raise ValueError("items must have unique ids")
        return items


def _error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": {"code": code, "message": message}})


def create_app(registry: ModelRegistry | None = None, *, cors_origins: list[str] | None = None) -> FastAPI:
    """Build the app. Tests inject a registry; production reads ASQP_SERVING_CONFIG."""
    app = FastAPI(title="OATS ASQP prediction service", version="1.0.0")
    app.state.registry = registry if registry is not None else ModelRegistry.from_env()

    origins = cors_origins if cors_origins is not None else _cors_origins_from_env()
    if origins:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["POST", "GET", "OPTIONS"],
            allow_headers=["Content-Type", "Accept"],
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        location = ".".join(str(part) for part in first.get("loc", ()) if part != "body")
        detail = first.get("msg", "Invalid request.")
        message = f"{location}: {detail}" if location else str(detail)
        return _error(422, "INVALID_REQUEST", message)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "models": app.state.registry.describe()}

    @app.post("/predict")
    def predict(request: PredictRequest) -> Any:
        try:
            predictor = app.state.registry.get(request.domain, request.model_type)
        except ModelNotReady as error:
            return _error(503, "MODEL_NOT_READY", str(error))
        try:
            outputs = predictor.predict([{"text": request.text}])
        except Exception:  # noqa: BLE001 - never surface internals to the client
            LOGGER.exception("Prediction failed for %s/%s", request.domain, request.model_type)
            return _error(500, "PREDICTION_FAILED", "The model could not process this request.")
        output = outputs[0] if outputs else {"quadruples": [], "raw_generation": "", "invalid_predictions": []}
        return {
            "text": request.text,
            "domain": request.domain,
            "model_type": request.model_type,
            "quadruples": output.get("quadruples", []),
            "raw_generation": output.get("raw_generation", ""),
            "invalid_predictions": output.get("invalid_predictions", []),
        }

    @app.post("/predict/batch")
    def predict_batch(request: BatchPredictRequest) -> Any:
        try:
            predictor = app.state.registry.get(request.domain, request.model_type)
        except ModelNotReady as error:
            return _error(503, "MODEL_NOT_READY", str(error))
        rows = [{"id": item.id, "text": item.text} for item in request.items]
        try:
            outputs = predictor.predict(rows)
        except Exception:  # noqa: BLE001
            LOGGER.exception("Batch prediction failed for %s/%s", request.domain, request.model_type)
            return _error(500, "PREDICTION_FAILED", "The model could not process this request.")
        if len(outputs) != len(rows):
            LOGGER.error("Predictor returned %d outputs for %d items", len(outputs), len(rows))
            return _error(500, "PREDICTION_FAILED", "The model returned an incomplete batch.")
        return {
            "results": [
                {
                    "id": row["id"],
                    "quadruples": output.get("quadruples", []),
                    "invalid_predictions": output.get("invalid_predictions", []),
                }
                # The predictor preserves input order and echoes the row id it was given.
                for row, output in zip(rows, outputs)
            ]
        }

    return app


def _cors_origins_from_env() -> list[str]:
    raw = os.environ.get("ASQP_CORS_ORIGINS", "").strip()
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def get_app() -> FastAPI:
    """Factory for `uvicorn capstone.service.app:get_app --factory`."""
    return create_app()
