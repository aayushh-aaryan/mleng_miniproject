"""Scoring API. The loaded Pipeline is the same object that was trained."""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import BookingRequest, ModelInfoResponse, PredictionResponse
from src.paths import MODEL_DIR
from src.risk import band_for_probability
from src.schema import INPUT_COLUMNS


def model_directory() -> Path:
    override = os.environ.get("MODEL_PATH", "").strip()
    return Path(override) if override else MODEL_DIR


def resolve_model_path(directory: Path) -> Path:
    if (directory / "MLmodel").exists():
        return directory
    if directory.exists():
        for child in directory.iterdir():
            if child.is_dir() and (child / "MLmodel").exists():
                return child
    raise FileNotFoundError(f"no MLflow model under {directory}")


def read_model_info(directory: Path, model_path: Path) -> dict[str, object]:
    for candidate in (directory / "model_info.json", model_path / "model_info.json"):
        if candidate.exists():
            loaded = json.loads(candidate.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                return loaded
    raise FileNotFoundError(f"model_info.json is missing under {directory}")


@asynccontextmanager
async def lifespan(application: FastAPI):
    directory = model_directory()
    model_path = resolve_model_path(directory)
    application.state.model = mlflow.sklearn.load_model(str(model_path))
    application.state.info = read_model_info(directory, model_path)
    yield


app = FastAPI(title="Hotel Cancel Guard", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> dict[str, object]:
    info = app.state.info
    if not isinstance(info, dict):
        raise HTTPException(status_code=503, detail="model info is not loaded")
    return info


@app.post("/predict", response_model=PredictionResponse)
def predict(booking: BookingRequest) -> PredictionResponse:
    info = app.state.info
    if not isinstance(info, dict):
        raise HTTPException(status_code=503, detail="model info is not loaded")
    bands = info.get("risk_bands")
    if not isinstance(bands, dict):
        raise HTTPException(status_code=503, detail="risk bands are missing from the model")
    frame = pd.DataFrame([booking.feature_row()])
    probability = float(app.state.model.predict_proba(frame.loc[:, list(INPUT_COLUMNS)])[0, 1])
    band, action = band_for_probability(
        probability,
        float(bands["reminder_threshold"]),
        float(bands["deposit_threshold"]),
    )
    return PredictionResponse(
        cancellation_probability=probability,
        risk_band=band,
        suggested_action=action,
    )
