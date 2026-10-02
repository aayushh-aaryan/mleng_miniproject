"""The pitch opener: a model allowed to see the outcome column."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from src.schema import TARGET


class LeakyEncoder(BaseEstimator, TransformerMixin):
    """Naive encoding of every column, including reservation status.

    Low-cardinality fields are one-hot encoded. High-cardinality fields are
    frequency encoded. Nothing is dropped. That is the mistake.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> LeakyEncoder:
        frame = pd.DataFrame(X).copy()
        self.columns_ = [column for column in frame.columns if column != TARGET]
        self.numeric_: list[str] = []
        self.one_hot_: list[str] = []
        self.frequency_: list[str] = []
        self.categories_: dict[str, list[str]] = {}
        self.freqs_: dict[str, dict[str, float]] = {}
        self.medians_: dict[str, float] = {}
        self.feature_names_: list[str] = []
        for column in self.columns_:
            series = frame[column]
            if pd.api.types.is_numeric_dtype(series):
                self.numeric_.append(column)
                median = series.median()
                self.medians_[column] = float(median) if pd.notna(median) else 0.0
                self.feature_names_.append(column)
                continue
            text = series.astype("string").fillna("missing")
            if int(text.nunique(dropna=True)) <= 40:
                categories = sorted(str(value) for value in text.unique())
                self.one_hot_.append(column)
                self.categories_[column] = categories
                self.feature_names_.extend(f"{column}={category}" for category in categories)
            else:
                self.frequency_.append(column)
                counts = text.value_counts(normalize=True)
                self.freqs_[column] = {str(key): float(val) for key, val in counts.items()}
                self.feature_names_.append(f"{column}_freq")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = pd.DataFrame(X).copy()
        pieces: list[pd.Series] = []
        for column in self.numeric_:
            values = pd.to_numeric(frame[column], errors="coerce").fillna(self.medians_[column])
            pieces.append(values.astype(float).reset_index(drop=True))
            pieces[-1].name = column
        for column in self.frequency_:
            text = frame[column].astype("string").fillna("missing")
            encoded = text.map(self.freqs_[column]).fillna(0.0).astype(float).reset_index(drop=True)
            encoded.name = f"{column}_freq"
            pieces.append(encoded)
        for column in self.one_hot_:
            text = frame[column].astype("string").fillna("missing").reset_index(drop=True)
            for category in self.categories_[column]:
                indicator = (text == category).astype(float)
                indicator.name = f"{column}={category}"
                pieces.append(indicator)
        return pd.concat(pieces, axis=1)

    def get_feature_names_out(self, input_features: np.ndarray | None = None) -> np.ndarray:
        return np.asarray(self.feature_names_, dtype=object)


def build_leaky_pipeline(random_state: int) -> Pipeline:
    from lightgbm import LGBMClassifier

    return Pipeline(
        steps=[
            ("encode", LeakyEncoder()),
            (
                "model",
                LGBMClassifier(
                    n_estimators=80,
                    learning_rate=0.08,
                    num_leaves=16,
                    random_state=random_state,
                    verbosity=-1,
                ),
            ),
        ]
    )
