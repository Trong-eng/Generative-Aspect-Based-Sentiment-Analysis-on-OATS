import json

import pytest

fastapi = pytest.importorskip("fastapi", reason="serving extras (requirements-serve.txt) are optional")
from fastapi.testclient import TestClient  # noqa: E402

from capstone.service.app import MAX_BATCH_ITEMS, create_app  # noqa: E402
from capstone.service.registry import ModelEntry, ModelNotReady, ModelRegistry, load_serving_entries  # noqa: E402

QUAD = {"target": "room", "aspect": "rooms cleanliness", "opinion": "dirty", "sentiment": "negative"}


class FakePredictor:
    """Stands in for a loaded checkpoint: records calls, echoes row ids."""

    def __init__(self, quadruples=None, error: Exception | None = None, drop_rows: bool = False):
        self.quadruples = [QUAD] if quadruples is None else quadruples
        self.error = error
        self.drop_rows = drop_rows
        self.calls: list[list[dict]] = []

    def predict(self, rows):
        self.calls.append(rows)
        if self.error is not None:
            raise self.error
        outputs = [
            {
                "text": row["text"],
                "quadruples": list(self.quadruples),
                "raw_generation": f"gen::{row['text']}",
                "invalid_predictions": [],
                **({"id": row["id"]} if "id" in row else {}),
            }
            for row in rows
        ]
        return outputs[:-1] if self.drop_rows else outputs


def registry_with(predictor, tmp_path, domain="hotels", model_type="paraphrase"):
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir(parents=True, exist_ok=True)
    entry = ModelEntry(domain=domain, model_type=model_type, checkpoint=checkpoint)
    return ModelRegistry(entries=[entry], loader=lambda *_args: predictor)


@pytest.fixture
def client_and_predictor(tmp_path):
    predictor = FakePredictor()
    registry = registry_with(predictor, tmp_path)
    return TestClient(create_app(registry)), predictor


def test_predict_returns_the_documented_contract(client_and_predictor):
    client, predictor = client_and_predictor

    response = client.post(
        "/predict",
        json={"text": "The room was dirty.", "domain": "hotels", "model_type": "paraphrase"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "text": "The room was dirty.",
        "domain": "hotels",
        "model_type": "paraphrase",
        "quadruples": [QUAD],
        "raw_generation": "gen::The room was dirty.",
        "invalid_predictions": [],
    }
    assert predictor.calls == [[{"text": "The room was dirty."}]]


def test_batch_echoes_ids_in_order_and_sends_every_item(client_and_predictor):
    client, predictor = client_and_predictor
    items = [{"id": "row-2", "text": "Dirty room."}, {"id": "row-3", "text": "Great staff."}]

    response = client.post("/predict/batch", json={"domain": "hotels", "model_type": "paraphrase", "items": items})

    assert response.status_code == 200
    assert response.json() == {
        "results": [
            {"id": "row-2", "quadruples": [QUAD], "invalid_predictions": []},
            {"id": "row-3", "quadruples": [QUAD], "invalid_predictions": []},
        ]
    }
    assert [row["text"] for row in predictor.calls[0]] == ["Dirty room.", "Great staff."]


def test_model_is_loaded_once_and_reused_across_requests(client_and_predictor):
    client, _ = client_and_predictor
    loads = []
    client.app.state.registry.loader = lambda *args: (loads.append(args), FakePredictor())[1]

    for _ in range(3):
        client.post("/predict", json={"text": "Hi.", "domain": "hotels", "model_type": "paraphrase"})

    assert len(loads) == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"text": "   ", "domain": "hotels", "model_type": "paraphrase"},
        {"text": "ok", "domain": "unknown_domain", "model_type": "paraphrase"},
        {"text": "ok", "domain": "hotels", "model_type": "bert"},
        {"domain": "hotels", "model_type": "paraphrase"},
    ],
)
def test_invalid_predict_payloads_use_the_error_shape(client_and_predictor, payload):
    client, predictor = client_and_predictor

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_REQUEST"
    assert isinstance(response.json()["error"]["message"], str)
    assert predictor.calls == []


@pytest.mark.parametrize(
    "items",
    [
        [],
        [{"id": "a", "text": "x"}, {"id": "a", "text": "y"}],
        [{"id": "", "text": "x"}],
        [{"text": "x"}],
        [{"id": f"row-{index}", "text": "x"} for index in range(MAX_BATCH_ITEMS + 1)],
    ],
)
def test_invalid_batch_payloads_are_rejected(client_and_predictor, items):
    client, predictor = client_and_predictor

    response = client.post("/predict/batch", json={"domain": "hotels", "model_type": "paraphrase", "items": items})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_REQUEST"
    assert predictor.calls == []


def test_unconfigured_pair_answers_model_not_ready(client_and_predictor):
    client, _ = client_and_predictor

    for path, payload in [
        ("/predict", {"text": "Hi.", "domain": "coursera", "model_type": "gas"}),
        ("/predict/batch", {"domain": "coursera", "model_type": "gas", "items": [{"id": "a", "text": "x"}]}),
    ]:
        response = client.post(path, json=payload)
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "MODEL_NOT_READY"


def test_missing_checkpoint_answers_model_not_ready_without_loading(tmp_path):
    loads = []
    registry = ModelRegistry(
        entries=[ModelEntry(domain="hotels", model_type="gas", checkpoint=tmp_path / "absent")],
        loader=lambda *args: loads.append(args),
    )
    client = TestClient(create_app(registry))

    response = client.post("/predict", json={"text": "Hi.", "domain": "hotels", "model_type": "gas"})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "MODEL_NOT_READY"
    assert loads == []


def test_loader_failure_is_reported_once_without_leaking_internals(tmp_path):
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    attempts = []

    def failing_loader(*args):
        attempts.append(args)
        raise RuntimeError("torch is not installed at C:/secret/path")

    registry = ModelRegistry(
        entries=[ModelEntry(domain="hotels", model_type="gas", checkpoint=checkpoint)],
        loader=failing_loader,
    )
    client = TestClient(create_app(registry))

    first = client.post("/predict", json={"text": "Hi.", "domain": "hotels", "model_type": "gas"})
    second = client.post("/predict", json={"text": "Hi.", "domain": "hotels", "model_type": "gas"})

    assert first.status_code == second.status_code == 503
    assert first.json()["error"]["code"] == "MODEL_NOT_READY"
    assert "secret" not in json.dumps(first.json())
    assert len(attempts) == 1, "a failed load must not be retried on every request"


def test_prediction_failure_does_not_leak_internals(tmp_path):
    predictor = FakePredictor(error=RuntimeError("CUDA out of memory on device 0"))
    registry = registry_with(predictor, tmp_path)
    client = TestClient(create_app(registry))

    response = client.post("/predict", json={"text": "Hi.", "domain": "hotels", "model_type": "paraphrase"})

    assert response.status_code == 500
    assert response.json()["error"] == {
        "code": "PREDICTION_FAILED",
        "message": "The model could not process this request.",
    }


def test_incomplete_batch_from_the_model_is_an_error_not_a_short_result(tmp_path):
    registry = registry_with(FakePredictor(drop_rows=True), tmp_path)
    client = TestClient(create_app(registry))

    response = client.post(
        "/predict/batch",
        json={"domain": "hotels", "model_type": "paraphrase", "items": [{"id": "a", "text": "x"}, {"id": "b", "text": "y"}]},
    )

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "PREDICTION_FAILED"


def test_health_lists_configured_pairs(tmp_path):
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    registry = ModelRegistry(
        entries=[
            ModelEntry(domain="hotels", model_type="gas", checkpoint=checkpoint),
            ModelEntry(domain="coursera", model_type="paraphrase", checkpoint=tmp_path / "absent"),
        ],
        loader=lambda *_args: FakePredictor(),
    )
    client = TestClient(create_app(registry))

    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert [(model["domain"], model["checkpoint_exists"]) for model in body["models"]] == [
        ("hotels", True),
        ("coursera", False),
    ]


def test_cors_headers_are_sent_only_for_configured_origins(tmp_path):
    registry = registry_with(FakePredictor(), tmp_path)
    client = TestClient(create_app(registry, cors_origins=["http://localhost:5173"]))

    allowed = client.post(
        "/predict",
        json={"text": "Hi.", "domain": "hotels", "model_type": "paraphrase"},
        headers={"Origin": "http://localhost:5173"},
    )
    other = client.post(
        "/predict",
        json={"text": "Hi.", "domain": "hotels", "model_type": "paraphrase"},
        headers={"Origin": "http://evil.example"},
    )

    assert allowed.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "access-control-allow-origin" not in other.headers


class TestServingConfig:
    def _write(self, tmp_path, document: str):
        path = tmp_path / "serving.yaml"
        path.write_text(document, encoding="utf-8")
        return path

    def test_relative_checkpoints_resolve_against_the_config_file(self, tmp_path):
        path = self._write(
            tmp_path,
            "models:\n  - domain: hotels\n    model_type: gas\n    checkpoint: ../runs/gas\n    config: gas.yaml\n",
        )

        entry = load_serving_entries(path)[0]

        assert entry.checkpoint == tmp_path / ".." / "runs" / "gas"
        assert entry.config_path == tmp_path / "gas.yaml"

    @pytest.mark.parametrize(
        "document",
        [
            "models: []\nx: 1\n".replace("models: []", "notmodels: []"),
            "models:\n  - domain: nope\n    model_type: gas\n    checkpoint: x\n",
            "models:\n  - domain: hotels\n    model_type: nope\n    checkpoint: x\n",
            "models:\n  - domain: hotels\n    model_type: gas\n",
            "models:\n  - domain: hotels\n    model_type: gas\n    checkpoint: a\n"
            "  - domain: hotels\n    model_type: gas\n    checkpoint: b\n",
        ],
    )
    def test_invalid_serving_configs_are_rejected(self, tmp_path, document):
        with pytest.raises(ValueError):
            load_serving_entries(self._write(tmp_path, document))

    def test_missing_env_var_yields_an_empty_registry(self):
        registry = ModelRegistry.from_env(env={})

        assert registry.describe() == []
        with pytest.raises(ModelNotReady):
            registry.get("hotels", "gas")

    def test_shipped_example_config_matches_the_full_matrix(self):
        entries = load_serving_entries("configs/serving.example.yaml")

        assert {(entry.domain, entry.model_type) for entry in entries} == {
            (domain, model_type)
            for domain in ("amazon_ff", "coursera", "hotels")
            for model_type in ("gas", "paraphrase")
        }
