"""Sklearn transformers. The saved model contains this step, so the API cannot drift."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.paths import FEATURED_CSV, PREPARED_CSV
from src.prepare import load_params
from src.schema import (
    INPUT_COLUMNS,
    MONTH_TO_NUM,
    POST_BOOKING_COLUMNS,
    TARGET,
)

ONE_HOT_COLUMNS: tuple[str, ...] = (
    "hotel",
    "meal",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
)

NUMERIC_COLUMNS: tuple[str, ...] = (
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "total_nights",
    "total_guests",
    "is_family",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "past_cancel_ratio",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "has_company",
    "has_agent",
    "agent_freq",
    "arrival_month_sin",
    "arrival_month_cos",
    "arrival_week_sin",
    "arrival_week_cos",
    "arrival_weekday_sin",
    "arrival_weekday_cos",
)

SKEWED_COLUMNS: tuple[str, ...] = ("lead_time", "adr")


def normalize_id(value: object) -> str | None:
    """Agent and company identifiers are labels. 240 is not 'more than' 9."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text == "" or text.lower() in {"null", "none", "nan"}:
            return None
        try:
            return str(int(float(text)))
        except ValueError:
            return text
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float):
        if np.isnan(value):
            return None
        return str(int(value))
    try:
        if pd.isna(value):
            return None
    except TypeError:
        return str(value)
    return str(value)


class BookingFeatureBuilder(BaseEstimator, TransformerMixin):
    """Turn a raw booking into the columns the encoder expects.

    Frequency maps are fit on the training rows only. An unseen country or
    agent becomes 0 (frequency) or Other (top-20), and one-hot uses
    handle_unknown='ignore' downstream, so a new category cannot crash scoring.
    """

    def __init__(
        self,
        country_encoding: str = "frequency",
        country_top_n: int = 20,
        adr_cap: float = 1000.0,
        log_skewed: bool = False,
        include_post_booking: bool = False,
    ) -> None:
        self.country_encoding = country_encoding
        self.country_top_n = country_top_n
        self.adr_cap = adr_cap
        self.log_skewed = log_skewed
        self.include_post_booking = include_post_booking

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> BookingFeatureBuilder:
        frame = self._as_frame(X)
        country = frame["country"].astype("string").fillna("Unknown")
        counts = country.value_counts()
        self.country_freq_: dict[str, float] = counts.div(float(counts.sum())).to_dict()
        self.country_top_: list[str] = [str(value) for value in counts.head(self.country_top_n).index]

        agent = frame["agent"].map(normalize_id).fillna("missing")
        agent_counts = agent.value_counts(normalize=True)
        self.agent_freq_: dict[str, float] = {str(key): float(val) for key, val in agent_counts.items()}
        self.feature_columns_: list[str] = self._output_columns()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = self._as_frame(X)
        adults = pd.to_numeric(frame["adults"], errors="coerce").fillna(0)
        children = pd.to_numeric(frame["children"], errors="coerce").fillna(0)
        babies = pd.to_numeric(frame["babies"], errors="coerce").fillna(0)
        weekend = pd.to_numeric(frame["stays_in_weekend_nights"], errors="coerce").fillna(0)
        week = pd.to_numeric(frame["stays_in_week_nights"], errors="coerce").fillna(0)
        lead_time = pd.to_numeric(frame["lead_time"], errors="coerce").fillna(0).clip(lower=0)
        adr = pd.to_numeric(frame["adr"], errors="coerce").fillna(0).clip(lower=0, upper=self.adr_cap)
        if self.log_skewed:
            lead_time = np.log1p(lead_time)
            adr = np.log1p(adr)

        previous_cancellations = pd.to_numeric(
            frame["previous_cancellations"], errors="coerce"
        ).fillna(0)
        previous_kept = pd.to_numeric(
            frame["previous_bookings_not_canceled"], errors="coerce"
        ).fillna(0)
        prev_c = previous_cancellations.to_numpy(dtype=float)
        prev_k = previous_kept.to_numpy(dtype=float)
        history = prev_c + prev_k
        past_cancel_ratio = np.divide(
            prev_c,
            history,
            out=np.zeros(len(frame), dtype=float),
            where=history > 0,
        )

        agent_id = frame["agent"].map(normalize_id)
        company_id = frame["company"].map(normalize_id)
        country = frame["country"].astype("string").fillna("Unknown")
        month_num = frame["arrival_date_month"].map(MONTH_TO_NUM).fillna(1).astype(float)
        week_num = pd.to_numeric(frame["arrival_date_week_number"], errors="coerce").fillna(1).clip(1, 53)
        year = pd.to_numeric(frame["arrival_date_year"], errors="coerce").fillna(2016).astype(int)
        day = (
            pd.to_numeric(frame["arrival_date_day_of_month"], errors="coerce")
            .fillna(1)
            .astype(int)
            .clip(1, 31)
        )
        dates = pd.to_datetime(
            {"year": year, "month": month_num.astype(int), "day": day},
            errors="coerce",
        )
        weekday = dates.dt.dayofweek.fillna(0).astype(float)

        built = pd.DataFrame(
            {
                "hotel": frame["hotel"].astype("string").fillna("Unknown"),
                "meal": frame["meal"].astype("string").fillna("SC").replace({"Undefined": "SC"}),
                "market_segment": frame["market_segment"].astype("string").fillna("Unknown"),
                "distribution_channel": frame["distribution_channel"].astype("string").fillna("Unknown"),
                "reserved_room_type": frame["reserved_room_type"].astype("string").fillna("Unknown"),
                "deposit_type": frame["deposit_type"].astype("string").fillna("Unknown"),
                "customer_type": frame["customer_type"].astype("string").fillna("Unknown"),
                "lead_time": lead_time.astype(float),
                "adr": adr.astype(float),
                "stays_in_weekend_nights": weekend.astype(float),
                "stays_in_week_nights": week.astype(float),
                "adults": adults.astype(float),
                "children": children.astype(float),
                "babies": babies.astype(float),
                "total_nights": (weekend + week).astype(float),
                "total_guests": (adults + children + babies).astype(float),
                "is_family": ((children + babies) > 0).astype(float),
                "is_repeated_guest": pd.to_numeric(frame["is_repeated_guest"], errors="coerce")
                .fillna(0)
                .astype(float),
                "previous_cancellations": previous_cancellations.astype(float),
                "previous_bookings_not_canceled": previous_kept.astype(float),
                "past_cancel_ratio": past_cancel_ratio.astype(float),
                "required_car_parking_spaces": pd.to_numeric(
                    frame["required_car_parking_spaces"], errors="coerce"
                )
                .fillna(0)
                .astype(float),
                "total_of_special_requests": pd.to_numeric(
                    frame["total_of_special_requests"], errors="coerce"
                )
                .fillna(0)
                .astype(float),
                "has_company": company_id.notna().astype(float),
                "has_agent": agent_id.notna().astype(float),
                "agent_freq": agent_id.fillna("missing").map(self.agent_freq_).fillna(0.0).astype(float),
                "arrival_month_sin": np.sin(2 * np.pi * month_num / 12),
                "arrival_month_cos": np.cos(2 * np.pi * month_num / 12),
                "arrival_week_sin": np.sin(2 * np.pi * week_num / 53),
                "arrival_week_cos": np.cos(2 * np.pi * week_num / 53),
                "arrival_weekday_sin": np.sin(2 * np.pi * weekday / 7),
                "arrival_weekday_cos": np.cos(2 * np.pi * weekday / 7),
            }
        )
        if self.country_encoding == "frequency":
            built["country_freq"] = country.map(self.country_freq_).fillna(0.0).astype(float)
        elif self.country_encoding == "top20":
            top = set(self.country_top_)
            built["country_group"] = np.where(country.isin(top), country, "Other")
        else:
            raise ValueError(f"unsupported country_encoding: {self.country_encoding}")

        if self.include_post_booking:
            built["assigned_room_type"] = (
                frame["assigned_room_type"].astype("string").fillna("Unknown")
            )
            built["booking_changes"] = pd.to_numeric(
                frame["booking_changes"], errors="coerce"
            ).fillna(0)
            built["days_in_waiting_list"] = pd.to_numeric(
                frame["days_in_waiting_list"], errors="coerce"
            ).fillna(0)

        return built.loc[:, self._output_columns()]

    def _output_columns(self) -> list[str]:
        columns = list(ONE_HOT_COLUMNS) + list(SKEWED_COLUMNS) + list(NUMERIC_COLUMNS)
        if self.country_encoding == "frequency":
            columns.append("country_freq")
        else:
            columns.append("country_group")
        if self.include_post_booking:
            columns.extend(POST_BOOKING_COLUMNS)
        return columns

    def _as_frame(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            frame = X.copy()
        else:
            frame = pd.DataFrame(X, columns=list(INPUT_COLUMNS))
        for column in INPUT_COLUMNS:
            if column not in frame.columns:
                frame[column] = np.nan
        return frame


def build_encoder(country_encoding: str, scale_skewed: bool, include_post_booking: bool) -> ColumnTransformer:
    categorical = list(ONE_HOT_COLUMNS)
    numeric = list(NUMERIC_COLUMNS)
    if country_encoding == "top20":
        categorical.append("country_group")
    elif country_encoding == "frequency":
        numeric.append("country_freq")
    else:
        raise ValueError(f"unsupported country_encoding: {country_encoding}")
    if include_post_booking:
        categorical.append("assigned_room_type")
        numeric.extend(["booking_changes", "days_in_waiting_list"])

    skewed_step: RobustScaler | str = RobustScaler() if scale_skewed else "passthrough"
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
            ("numeric", "passthrough", numeric),
            ("skewed", skewed_step, list(SKEWED_COLUMNS)),
        ],
        remainder="drop",
    )


def build_pipeline(
    estimator: BaseEstimator,
    country_encoding: str,
    country_top_n: int,
    adr_cap: float,
    log_skewed: bool,
    include_post_booking: bool = False,
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "engineer",
                BookingFeatureBuilder(
                    country_encoding=country_encoding,
                    country_top_n=country_top_n,
                    adr_cap=adr_cap,
                    log_skewed=log_skewed,
                    include_post_booking=include_post_booking,
                ),
            ),
            (
                "encode",
                build_encoder(country_encoding, log_skewed, include_post_booking),
            ),
            ("model", estimator),
        ]
    )


def main() -> None:
    """Write the human-readable feature table. The model refits these steps itself."""
    params = load_params()
    features_section = params["features"]
    prepare_section = params["prepare"]
    if not isinstance(features_section, dict) or not isinstance(prepare_section, dict):
        raise ValueError("params.yaml features and prepare sections must be mappings")
    prepared = pd.read_csv(PREPARED_CSV)
    builder = BookingFeatureBuilder(
        country_encoding=str(features_section["country_encoding"]),
        country_top_n=int(features_section["country_top_n"]),
        adr_cap=float(prepare_section["adr_cap"]),
        log_skewed=False,
        include_post_booking=False,
    )
    transformed = builder.fit_transform(prepared.loc[:, list(INPUT_COLUMNS)])
    transformed[TARGET] = prepared[TARGET].to_numpy()
    FEATURED_CSV.parent.mkdir(parents=True, exist_ok=True)
    transformed.to_csv(FEATURED_CSV, index=False)
    print(f"wrote {FEATURED_CSV} rows={len(transformed)} cols={transformed.shape[1]}")


if __name__ == "__main__":
    main()
