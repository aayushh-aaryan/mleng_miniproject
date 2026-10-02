"""Fit one sklearn Pipeline and score it on the summer 2017 holdout."""

from __future__ import annotations

import json
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.features import ONE_HOT_COLUMNS, build_pipeline
from src.leakage import build_leaky_pipeline
from src.metrics import classification_scores
from src.paths import EVAL_METRICS, HOLDOUT_CSV, MLFLOW_DB, MODEL_DIR, TRAIN_METRICS
from src.prepare import filter_rows, modeling_frame, read_bookings
from src.schema import INPUT_COLUMNS, POST_BOOKING_COLUMNS, TARGET

MODEL_REGISTRY_NAME = "hotel-cancellation"


def configure_mlflow() -> None:
    uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if not uri:
        uri = f"sqlite:///{MLFLOW_DB}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("hotel-cancellation")


def param_section(params: dict[str, object], key: str) -> dict[str, object]:
    value = params[key]
    if not isinstance(value, dict):
        raise ValueError(f"params.yaml section '{key}' must be a mapping")
    return value


def build_estimator(name: str, train_params: dict[str, object], random_state: int) -> LogisticRegression | RandomForestClassifier | LGBMClassifier:
    if name == "logistic_regression":
        cfg = train_params["logistic_regression"]
        if not isinstance(cfg, dict):
            raise ValueError("logistic_regression hyperparameters must be a mapping")
        return LogisticRegression(
            C=float(cfg["C"]),
            max_iter=int(cfg["max_iter"]),
            random_state=random_state,
        )
    if name == "random_forest":
        cfg = train_params["random_forest"]
        if not isinstance(cfg, dict):
            raise ValueError("random_forest hyperparameters must be a mapping")
        return RandomForestClassifier(
            n_estimators=int(cfg["n_estimators"]),
            max_depth=int(cfg["max_depth"]),
            min_samples_leaf=int(cfg["min_samples_leaf"]),
            random_state=random_state,
            n_jobs=-1,
        )
    if name == "lightgbm":
        cfg = train_params["lightgbm"]
        if not isinstance(cfg, dict):
            raise ValueError("lightgbm hyperparameters must be a mapping")
        return LGBMClassifier(
            n_estimators=int(cfg["n_estimators"]),
            learning_rate=float(cfg["learning_rate"]),
            num_leaves=int(cfg["num_leaves"]),
            min_child_samples=int(cfg["min_child_samples"]),
            subsample=float(cfg["subsample"]),
            colsample_bytree=float(cfg["colsample_bytree"]),
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )
    raise ValueError(f"unknown estimator: {name}")


def training_matrix(frame: pd.DataFrame, adr_cap: float, post_booking: bool) -> tuple[pd.DataFrame, pd.Series]:
    cleaned = filter_rows(frame, adr_cap) if post_booking else modeling_frame(frame, adr_cap)
    target = cleaned[TARGET].astype(int)
    columns = list(INPUT_COLUMNS)
    if post_booking:
        columns.extend(POST_BOOKING_COLUMNS)
    return cleaned.loc[:, columns], target


def holdout_matrix(adr_cap: float, include_post_booking: bool) -> tuple[pd.DataFrame, pd.Series]:
    cleaned = filter_rows(read_bookings(HOLDOUT_CSV), adr_cap)
    columns = list(INPUT_COLUMNS)
    if include_post_booking:
        columns.extend(POST_BOOKING_COLUMNS)
    target = cleaned[TARGET].astype(int)
    return cleaned.loc[:, columns], target


def holdout_scores(
    pipeline: object,
    adr_cap: float,
    high_risk_threshold: float,
    include_post_booking: bool = False,
) -> dict[str, float]:
    features, target = holdout_matrix(adr_cap, include_post_booking)
    probabilities = pipeline.predict_proba(features)[:, 1]  # type: ignore[attr-defined]
    return classification_scores(target.to_numpy(), probabilities, high_risk_threshold)


def _importance_values(model: object) -> np.ndarray:
    booster = getattr(model, "booster_", None)
    if booster is not None:
        return np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=float)
    coefficients = getattr(model, "coef_", None)
    if coefficients is None:
        raise TypeError("model has no feature importance")
    return np.abs(np.asarray(coefficients[0], dtype=float))


def grouped_importance(pipeline: object) -> list[dict[str, float | str]]:
    model = pipeline.named_steps["model"]  # type: ignore[attr-defined]
    encoder = pipeline.named_steps["encode"]  # type: ignore[attr-defined]
    names = [str(name) for name in encoder.get_feature_names_out()]
    values = _importance_values(model)
    grouped: dict[str, float] = {}
    prefixes = list(ONE_HOT_COLUMNS) + ["country_group", "assigned_room_type"]
    for name, value in zip(names, values):
        bare = name.split("__", 1)[-1]
        field = bare
        for prefix in prefixes:
            if bare.startswith(prefix + "_") or bare.startswith(prefix + "="):
                field = prefix
                break
        grouped[field] = grouped.get(field, 0.0) + float(value)
    total = sum(grouped.values()) or 1.0
    ranked = sorted(grouped.items(), key=lambda item: item[1], reverse=True)
    return [{"feature": key, "share": value / total} for key, value in ranked[:12]]


def leaky_coefficients(pipeline: object) -> list[dict[str, float | str]]:
    """Share of leaky-model gain by original column. reservation_status should dominate."""
    model = pipeline.named_steps["model"]  # type: ignore[attr-defined]
    encoder = pipeline.named_steps["encode"]  # type: ignore[attr-defined]
    names = [str(name) for name in encoder.get_feature_names_out()]
    values = _importance_values(model)
    grouped: dict[str, float] = {}
    for name, value in zip(names, values):
        field = name.split("=", 1)[0]
        if field.endswith("_freq"):
            field = field[: -len("_freq")]
        grouped[field] = grouped.get(field, 0.0) + float(value)
    total = sum(grouped.values()) or 1.0
    ranked = sorted(grouped.items(), key=lambda item: item[1], reverse=True)
    return [{"feature": key, "share": value / total} for key, value in ranked[:8]]


def log_run(
    run_name: str,
    pipeline: object,
    metrics: dict[str, float],
    params: dict[str, str | int | float],
    tags: dict[str, str],
    register: bool,
    model_info: dict[str, object] | None = None,
) -> str:
    configure_mlflow()
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tags(tags)
        if model_info is not None:
            mlflow.log_dict(model_info, "model_info.json")
        if register:
            info = mlflow.sklearn.log_model(
                pipeline,
                name="model",
                registered_model_name=MODEL_REGISTRY_NAME,
                code_paths=["src"],
                serialization_format="cloudpickle",
            )
            version = info.registered_model_version
            client = mlflow.MlflowClient()
            client.set_registered_model_alias(MODEL_REGISTRY_NAME, "champion", version)
            client.set_model_version_tag(MODEL_REGISTRY_NAME, version, "alias", "champion")
            for key, value in tags.items():
                client.set_model_version_tag(MODEL_REGISTRY_NAME, version, key, value)
        else:
            mlflow.sklearn.log_model(
                pipeline,
                name="model",
                code_paths=["src"],
                serialization_format="cloudpickle",
            )
            version = ""
        mlflow.set_tag("registry_version", version or "unregistered")
        return run.info.run_id


def write_model_bundle(
    pipeline: object,
    info: dict[str, object],
) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.sklearn.save_model(
        pipeline,
        str(MODEL_DIR),
        code_paths=["src"],
        serialization_format="cloudpickle",
    )
    (MODEL_DIR / "model_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def fit_clean_model(
    raw: pd.DataFrame,
    estimator_name: str,
    country_encoding: str,
    params: dict[str, object],
    data_version: str,
    include_post_booking: bool = False,
) -> tuple[object, dict[str, float]]:
    prepare_params = param_section(params, "prepare")
    feature_params = param_section(params, "features")
    train_params = param_section(params, "train")
    risk_params = param_section(params, "risk")
    adr_cap = float(prepare_params["adr_cap"])
    random_state = int(train_params["random_state"])
    features, target = training_matrix(raw, adr_cap, include_post_booking)
    estimator = build_estimator(estimator_name, train_params, random_state)
    pipeline = build_pipeline(
        estimator=estimator,
        country_encoding=country_encoding,
        country_top_n=int(feature_params["country_top_n"]),
        adr_cap=adr_cap,
        log_skewed=estimator_name == "logistic_regression",
        include_post_booking=include_post_booking,
    )
    pipeline.fit(features, target)
    del data_version
    scores = holdout_scores(
        pipeline,
        adr_cap,
        float(risk_params["deposit_threshold"]),
        include_post_booking=include_post_booking,
    )
    scores["training_rows"] = float(len(features))
    return pipeline, scores


def fit_leaky_model(
    raw: pd.DataFrame,
    params: dict[str, object],
) -> tuple[object, dict[str, float]]:
    prepare_params = param_section(params, "prepare")
    train_params = param_section(params, "train")
    risk_params = param_section(params, "risk")
    adr_cap = float(prepare_params["adr_cap"])
    cleaned = filter_rows(raw, adr_cap)
    target = cleaned[TARGET].astype(int)
    features = cleaned.drop(columns=[TARGET])
    pipeline = build_leaky_pipeline(int(train_params["random_state"]))
    pipeline.fit(features, target)
    holdout = filter_rows(read_bookings_holdout(), adr_cap)
    holdout_target = holdout[TARGET].astype(int)
    holdout_features = holdout.drop(columns=[TARGET])
    probabilities = pipeline.predict_proba(holdout_features)[:, 1]
    scores = classification_scores(
        holdout_target.to_numpy(),
        probabilities,
        float(risk_params["deposit_threshold"]),
    )
    scores["training_rows"] = float(len(features))
    return pipeline, scores


def read_bookings_holdout() -> pd.DataFrame:
    from src.paths import HOLDOUT_CSV

    return read_bookings(HOLDOUT_CSV)


def champion_info(
    metrics: dict[str, float],
    estimator_name: str,
    country_encoding: str,
    data_version: str,
    params: dict[str, object],
    run_id: str,
    version: str,
) -> dict[str, object]:
    risk_params = param_section(params, "risk")
    public_metrics = {
        key: value
        for key, value in metrics.items()
        if key in {"roc_auc", "pr_auc", "recall_high_risk", "precision_high_risk", "accuracy"}
    }
    return {
        "model_name": MODEL_REGISTRY_NAME,
        "model_version": version,
        "alias": "champion",
        "data_version": data_version,
        "estimator": estimator_name,
        "country_encoding": country_encoding,
        "metrics": public_metrics,
        "risk_bands": {
            "reminder_threshold": float(risk_params["reminder_threshold"]),
            "deposit_threshold": float(risk_params["deposit_threshold"]),
        },
        "run_id": run_id,
    }


def save_champion(
    pipeline: object,
    metrics: dict[str, float],
    estimator_name: str,
    country_encoding: str,
    data_version: str,
    params: dict[str, object],
    run_name: str,
    tags: dict[str, str],
) -> dict[str, object]:
    logged_params: dict[str, str | int | float] = {
        "estimator": estimator_name,
        "country_encoding": country_encoding,
        "data_version": data_version,
        "log_skewed": estimator_name == "logistic_regression",
    }
    info_preview = champion_info(
        metrics,
        estimator_name,
        country_encoding,
        data_version,
        params,
        run_id="",
        version="",
    )
    run_id = log_run(
        run_name,
        pipeline,
        _public_metrics(metrics),
        logged_params,
        tags,
        register=True,
        model_info=info_preview,
    )
    client = mlflow.MlflowClient()
    version = client.get_model_version_by_alias(MODEL_REGISTRY_NAME, "champion").version
    info = champion_info(metrics, estimator_name, country_encoding, data_version, params, run_id, version)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_dict(info, "model_info.json")
    write_model_bundle(pipeline, info)
    write_json(TRAIN_METRICS, {"rows": metrics.get("training_rows", 0.0), "estimator": estimator_name, "data_version": data_version})
    write_json(EVAL_METRICS, _public_metrics(metrics))
    return info


def _public_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {
        key: float(metrics[key])
        for key in ("roc_auc", "pr_auc", "recall_high_risk", "precision_high_risk", "accuracy")
    }
