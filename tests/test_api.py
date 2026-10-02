from __future__ import annotations

import json
from collections.abc import Iterator

import mlflow
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from src.features import build_pipeline
from src.schema import INPUT_COLUMNS, TARGET
from tests.synthetic import synthetic_bookings, valid_payload


@pytest.fixture()
def client(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    from pathlib import Path

    if not isinstance(tmp_path, Path):
        raise TypeError("tmp_path fixture was not a Path")
    frame = synthetic_bookings()
    pipeline = build_pipeline(
        estimator=LogisticRegression(max_iter=300),
        country_encoding="frequency",
        country_top_n=3,
        adr_cap=1000,
        log_skewed=True,
    )
    pipeline.fit(frame.loc[:, list(INPUT_COLUMNS)], frame[TARGET].astype(int))
    model_dir = tmp_path / "champion"
    mlflow.sklearn.save_model(pipeline, str(model_dir), serialization_format="cloudpickle")
    info = {
        "model_name": "hotel-cancellation",
        "model_version": "test",
        "alias": "champion",
        "data_version": "v2",
        "estimator": "logistic_regression",
        "country_encoding": "frequency",
        "metrics": {
            "roc_auc": 0.5,
            "pr_auc": 0.5,
            "recall_high_risk": 0.5,
            "precision_high_risk": 0.5,
            "accuracy": 0.5,
        },
        "risk_bands": {"reminder_threshold": 0.3, "deposit_threshold": 0.6},
    }
    (model_dir / "model_info.json").write_text(json.dumps(info), encoding="utf-8")
    monkeypatch.setenv("MODEL_PATH", str(model_dir))
    from api.main import app

    with TestClient(app) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info(client: TestClient) -> None:
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["alias"] == "champion"
    assert body["data_version"] == "v2"


def test_predict_probability_is_between_zero_and_one(client: TestClient) -> None:
    response = client.post("/predict", json=valid_payload())
    assert response.status_code == 200
    probability = response.json()["cancellation_probability"]
    assert 0.0 <= probability <= 1.0
    assert response.json()["risk_band"] in {"low", "medium", "high"}


def test_unseen_country_scores(client: TestClient) -> None:
    payload = valid_payload()
    payload["country"] = "ZZ"
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert 0.0 <= response.json()["cancellation_probability"] <= 1.0


def test_negative_adults_get_422(client: TestClient) -> None:
    payload = valid_payload()
    payload["adults"] = -2
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
