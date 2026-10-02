"""Row cleaning shared by training, evaluation, and the leaky baseline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.paths import HOLDOUT_CSV, PARAMS_PATH, PREPARED_CSV, RAW_BOOKINGS
from src.schema import INPUT_COLUMNS, NA_VALUES, TARGET


def load_params() -> dict[str, object]:
    with PARAMS_PATH.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("params.yaml must contain a mapping")
    return loaded


def read_bookings(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, na_values=list(NA_VALUES), keep_default_na=True)


def filter_rows(frame: pd.DataFrame, adr_cap: float) -> pd.DataFrame:
    """Drop impossible bookings and apply the cleaning rules.

    Zero guests and negative prices are removed. The extreme ADR outlier is
    capped. Undefined and SC both mean "no meal package", so they are merged.
    """
    cleaned = frame.copy()
    for column in ("adults", "children", "babies"):
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce").fillna(0)
    cleaned["adr"] = pd.to_numeric(cleaned["adr"], errors="coerce")
    guests = cleaned["adults"] + cleaned["children"] + cleaned["babies"]
    cleaned = cleaned.loc[guests > 0].copy()
    cleaned = cleaned.loc[cleaned["adr"].notna() & (cleaned["adr"] >= 0)].copy()
    cleaned["adr"] = cleaned["adr"].clip(upper=adr_cap)
    cleaned["meal"] = cleaned["meal"].replace({"Undefined": "SC"})
    cleaned["country"] = cleaned["country"].fillna("Unknown").astype(str)
    return cleaned


def modeling_frame(frame: pd.DataFrame, adr_cap: float) -> pd.DataFrame:
    """Cleaning plus leakage removal. This is the table the clean models train on."""
    cleaned = filter_rows(frame, adr_cap)
    columns = [TARGET] + list(INPUT_COLUMNS)
    missing = [column for column in columns if column not in cleaned.columns]
    if missing:
        raise ValueError(f"bookings file is missing columns: {missing}")
    return cleaned.loc[:, columns].reset_index(drop=True)


def prepare_training_file(adr_cap: float) -> pd.DataFrame:
    prepared = modeling_frame(read_bookings(RAW_BOOKINGS), adr_cap)
    PREPARED_CSV.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(PREPARED_CSV, index=False)
    return prepared


def load_holdout_inputs(adr_cap: float) -> tuple[pd.DataFrame, pd.Series]:
    """Holdout rows with the same filters. Leakage columns are not selected."""
    prepared = modeling_frame(read_bookings(HOLDOUT_CSV), adr_cap)
    features = prepared.loc[:, list(INPUT_COLUMNS)]
    target = prepared[TARGET].astype(int)
    return features, target


def main() -> None:
    params = load_params()
    prepare_section = params["prepare"]
    if not isinstance(prepare_section, dict):
        raise ValueError("params.yaml prepare section must be a mapping")
    adr_cap = float(prepare_section["adr_cap"])
    prepared = prepare_training_file(adr_cap)
    print(f"wrote {PREPARED_CSV} rows={len(prepared)}")


if __name__ == "__main__":
    main()
