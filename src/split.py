"""Time split. June–August 2017 is held out of every training version."""

from __future__ import annotations

import pandas as pd

from src.schema import EARLY_2017_MONTHS, SUMMER_MONTHS


def split_bookings(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return v1, v2, and the fixed summer 2017 holdout.

    v1 is 2015–2016. v2 is 2015 through May 2017. The holdout is never
    part of either training file.
    """
    year = frame["arrival_date_year"]
    month = frame["arrival_date_month"]
    holdout_mask = (year == 2017) & month.isin(SUMMER_MONTHS)
    v1_mask = year.isin([2015, 2016])
    v2_mask = v1_mask | ((year == 2017) & month.isin(EARLY_2017_MONTHS))
    holdout = frame.loc[holdout_mask].copy()
    v1 = frame.loc[v1_mask].copy()
    v2 = frame.loc[v2_mask].copy()
    covered = holdout_mask | v2_mask
    if not bool(covered.all()):
        missing = int((~covered).sum())
        raise ValueError(f"{missing} rows fell outside v2 and the summer holdout")
    if set(v2["arrival_date_month"].loc[v2["arrival_date_year"] == 2017]).intersection(
        SUMMER_MONTHS
    ):
        raise ValueError("summer 2017 rows leaked into training v2")
    return {"v1": v1, "v2": v2, "holdout": holdout}
