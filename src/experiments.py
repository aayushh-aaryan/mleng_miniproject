"""Compare the leaky baseline, the clean models, and data v1 against v2."""

from __future__ import annotations

import json
import re

import numpy as np

from src.features import normalize_id
from src.metrics import classification_scores
from src.modeling import (
    configure_mlflow,
    fit_clean_model,
    fit_leaky_model,
    grouped_importance,
    leaky_coefficients,
    log_run,
    param_section,
    save_champion,
)
from src.paths import (
    BOOKINGS_V1,
    BOOKINGS_V2,
    EXAMPLES_PATH,
    EXPERIMENT_SUMMARY,
    HOLDOUT_CSV,
    METRICS_DIR,
    PARAMS_PATH,
)
from src.prepare import filter_rows, load_params, modeling_frame, read_bookings
from src.schema import INPUT_COLUMNS, TARGET


def _public(metrics: dict[str, float]) -> dict[str, float]:
    return {
        key: float(metrics[key])
        for key in ("roc_auc", "pr_auc", "recall_high_risk", "precision_high_risk", "accuracy")
    }


def _log_candidate(
    name: str,
    pipeline: object,
    metrics: dict[str, float],
    tags: dict[str, str],
    logged_params: dict[str, str | int | float],
) -> str:
    return log_run(name, pipeline, _public(metrics), logged_params, tags, register=False)


def _record(
    name: str,
    metrics: dict[str, float],
    tags: dict[str, str],
    eligible: bool,
    estimator: str,
    country_encoding: str,
    data_version: str,
) -> dict[str, object]:
    return {
        "name": name,
        "eligible": eligible,
        "estimator": estimator,
        "country_encoding": country_encoding,
        "data_version": data_version,
        "status": tags.get("status", "candidate"),
        "metrics": _public(metrics),
    }


def _update_params(estimator: str, country_encoding: str, data_version: str) -> None:
    text = PARAMS_PATH.read_text(encoding="utf-8")
    text = re.sub(r"(?m)^  model: .*$", f"  model: {estimator}", text, count=1)
    text = re.sub(
        r"(?m)^  country_encoding: .*$",
        f"  country_encoding: {country_encoding}",
        text,
        count=1,
    )
    text = re.sub(r"(?m)^  data_version: .*$", f"  data_version: {data_version}", text, count=1)
    PARAMS_PATH.write_text(text, encoding="utf-8")


def _json_ready(value: object) -> object:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if number.is_integer():
            return int(number)
        return number
    return value


def export_examples(pipeline: object, adr_cap: float) -> None:
    """Pick a real holdout booking from each end of the score range."""
    prepared = modeling_frame(read_bookings(HOLDOUT_CSV), adr_cap)
    features = prepared.loc[:, list(INPUT_COLUMNS)]
    probabilities = pipeline.predict_proba(features)[:, 1]  # type: ignore[attr-defined]
    risky_index = int(np.argmax(probabilities))
    safe_index = int(np.argmin(probabilities))

    def booking_at(index: int) -> dict[str, object]:
        row = features.iloc[index]
        payload: dict[str, object] = {}
        for column in INPUT_COLUMNS:
            if column in {"agent", "company"}:
                payload[column] = normalize_id(row[column])
            else:
                payload[column] = _json_ready(row[column])
        return payload

    examples = {
        "risky": {
            "title": "High-risk holdout booking",
            "probability": float(probabilities[risky_index]),
            "booking": booking_at(risky_index),
        },
        "safe": {
            "title": "Low-risk holdout booking",
            "probability": float(probabilities[safe_index]),
            "booking": booking_at(safe_index),
        },
    }
    EXAMPLES_PATH.write_text(json.dumps(examples, indent=2), encoding="utf-8")


def main() -> None:
    configure_mlflow()
    params = load_params()
    prepare_params = param_section(params, "prepare")
    feature_params = param_section(params, "features")
    risk_params = param_section(params, "risk")
    adr_cap = float(prepare_params["adr_cap"])
    country_top_n = int(feature_params["country_top_n"])
    v1 = read_bookings(BOOKINGS_V1)
    v2 = read_bookings(BOOKINGS_V2)
    holdout = filter_rows(read_bookings(HOLDOUT_CSV), adr_cap)
    cancel_rate = float(holdout[TARGET].astype(int).mean())
    runs: list[dict[str, object]] = []

    print("fitting leaky baseline", flush=True)
    leaky_pipeline, leaky_scores = fit_leaky_model(v2, params)
    _log_candidate(
        "leaky-baseline",
        leaky_pipeline,
        leaky_scores,
        {"status": "do-not-use", "reason": "uses reservation_status", "data_version": "v2"},
        {"estimator": "lightgbm", "columns": "all", "data_version": "v2"},
    )
    runs.append(
        _record(
            "leaky-baseline",
            leaky_scores,
            {"status": "do-not-use"},
            eligible=False,
            estimator="lightgbm",
            country_encoding="naive",
            data_version="v2",
        )
    )
    leaky_top = leaky_coefficients(leaky_pipeline)

    print("fitting post-booking columns", flush=True)
    post_pipeline, post_scores = fit_clean_model(
        v2,
        estimator_name="lightgbm",
        country_encoding="frequency",
        params=params,
        data_version="v2",
        include_post_booking=True,
    )
    _log_candidate(
        "post-booking-columns",
        post_pipeline,
        post_scores,
        {
            "status": "do-not-use",
            "reason": "assigned_room_type, booking_changes, days_in_waiting_list",
            "data_version": "v2",
        },
        {"estimator": "lightgbm", "country_encoding": "frequency", "data_version": "v2"},
    )
    runs.append(
        _record(
            "post-booking-columns",
            post_scores,
            {"status": "do-not-use"},
            eligible=False,
            estimator="lightgbm",
            country_encoding="frequency",
            data_version="v2",
        )
    )

    candidates: list[dict[str, object]] = []
    recipes: tuple[tuple[str, str], ...] = (
        ("logistic_regression", "frequency"),
        ("random_forest", "frequency"),
        ("lightgbm", "frequency"),
        ("lightgbm", "top20"),
    )
    fitted: dict[tuple[str, str], tuple[object, dict[str, float]]] = {}
    for estimator_name, encoding in recipes:
        print(f"fitting {estimator_name} {encoding} v2", flush=True)
        pipeline, scores = fit_clean_model(
            v2,
            estimator_name=estimator_name,
            country_encoding=encoding,
            params=params,
            data_version="v2",
        )
        fitted[(estimator_name, encoding)] = (pipeline, scores)
        run_name = f"{estimator_name}-{encoding}-v2"
        _log_candidate(
            run_name,
            pipeline,
            scores,
            {
                "status": "candidate",
                "data_version": "v2",
                "estimator": estimator_name,
                "country_encoding": encoding,
            },
            {
                "estimator": estimator_name,
                "country_encoding": encoding,
                "data_version": "v2",
                "log_skewed": estimator_name == "logistic_regression",
                "country_top_n": country_top_n,
            },
        )
        record = _record(
            run_name,
            scores,
            {"status": "candidate"},
            eligible=True,
            estimator=estimator_name,
            country_encoding=encoding,
            data_version="v2",
        )
        runs.append(record)
        candidates.append(record)
        print(run_name, _public(scores))

    def rank_key(record: dict[str, object]) -> tuple[float, float]:
        metrics = record["metrics"]
        if not isinstance(metrics, dict):
            return (0.0, 0.0)
        return (float(metrics["pr_auc"]), float(metrics["roc_auc"]))

    winner = max(candidates, key=rank_key)
    winner_estimator = str(winner["estimator"])
    winner_encoding = str(winner["country_encoding"])
    winner_pipeline, winner_scores = fitted[(winner_estimator, winner_encoding)]

    v1_pipeline, v1_scores = fit_clean_model(
        v1,
        estimator_name=winner_estimator,
        country_encoding=winner_encoding,
        params=params,
        data_version="v1",
    )
    _log_candidate(
        f"{winner_estimator}-{winner_encoding}-v1",
        v1_pipeline,
        v1_scores,
        {
            "status": "comparison",
            "data_version": "v1",
            "estimator": winner_estimator,
            "country_encoding": winner_encoding,
        },
        {
            "estimator": winner_estimator,
            "country_encoding": winner_encoding,
            "data_version": "v1",
        },
    )
    runs.append(
        _record(
            f"{winner_estimator}-{winner_encoding}-v1",
            v1_scores,
            {"status": "comparison"},
            eligible=False,
            estimator=winner_estimator,
            country_encoding=winner_encoding,
            data_version="v1",
        )
    )

    # The live model is the v2 fit. v1 stays in MLflow so the data-version diff is honest.
    _update_params(winner_estimator, winner_encoding, "v2")
    refreshed = load_params()
    info = save_champion(
        winner_pipeline,
        winner_scores,
        estimator_name=winner_estimator,
        country_encoding=winner_encoding,
        data_version="v2",
        params=refreshed,
        run_name=f"champion-{winner_estimator}-{winner_encoding}-v2",
        tags={
            "status": "champion",
            "data_version": "v2",
            "estimator": winner_estimator,
            "country_encoding": winner_encoding,
        },
    )
    export_examples(winner_pipeline, adr_cap)
    importance = grouped_importance(winner_pipeline)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "v1.json").write_text(json.dumps(_public(v1_scores), indent=2), encoding="utf-8")
    (METRICS_DIR / "v2.json").write_text(json.dumps(_public(winner_scores), indent=2), encoding="utf-8")
    summary = {
        "holdout": "June–August 2017",
        "holdout_rows": int(len(holdout)),
        "holdout_cancel_rate": cancel_rate,
        "always_keep_accuracy": 1.0 - cancel_rate,
        "deposit_threshold": float(risk_params["deposit_threshold"]),
        "reminder_threshold": float(risk_params["reminder_threshold"]),
        "runs": runs,
        "leaky_importance": leaky_top,
        "champion_importance": importance,
        "champion": info,
        "v1_metrics": _public(v1_scores),
        "v2_metrics": _public(winner_scores),
    }
    EXPERIMENT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"champion": info["estimator"], "data_version": "v2", "metrics": info["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
