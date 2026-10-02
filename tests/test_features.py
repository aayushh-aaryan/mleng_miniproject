from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.features import build_pipeline
from src.schema import INPUT_COLUMNS, TARGET
from tests.synthetic import BANNED_FROM_MODEL, synthetic_bookings


def _fit(country_encoding: str = "frequency"):
    frame = synthetic_bookings()
    pipeline = build_pipeline(
        estimator=LogisticRegression(max_iter=300),
        country_encoding=country_encoding,
        country_top_n=3,
        adr_cap=1000,
        log_skewed=True,
    )
    features = frame.loc[:, list(INPUT_COLUMNS)]
    pipeline.fit(features, frame[TARGET].astype(int))
    return pipeline, features


def _feature_names(pipeline) -> list[str]:
    return [str(name).split("__", 1)[-1] for name in pipeline.named_steps["encode"].get_feature_names_out()]


def test_leakage_columns_never_reach_the_model() -> None:
    pipeline, _features = _fit()
    names = _feature_names(pipeline)
    for banned in BANNED_FROM_MODEL:
        assert banned not in names


def test_extra_leakage_columns_do_not_change_the_score() -> None:
    pipeline, features = _fit()
    leaked = features.copy()
    leaked["reservation_status"] = "Canceled"
    leaked["reservation_status_date"] = "2016-01-01"
    leaked["assigned_room_type"] = "H"
    leaked["booking_changes"] = 9
    leaked["days_in_waiting_list"] = 40
    clean_score = float(pipeline.predict_proba(features.iloc[[0]])[0, 1])
    leaked_score = float(pipeline.predict_proba(leaked.iloc[[0]])[0, 1])
    assert clean_score == leaked_score


def test_unseen_country_does_not_crash() -> None:
    for encoding in ("frequency", "top20"):
        pipeline, features = _fit(encoding)
        row = features.iloc[[0]].copy()
        row["country"] = "ZZ"
        probability = float(pipeline.predict_proba(row)[0, 1])
        assert 0.0 <= probability <= 1.0


def test_unseen_hotel_does_not_crash() -> None:
    pipeline, features = _fit()
    row = features.iloc[[0]].copy()
    row["hotel"] = "Moon Hotel"
    probability = float(pipeline.predict_proba(row)[0, 1])
    assert np.isfinite(probability)
