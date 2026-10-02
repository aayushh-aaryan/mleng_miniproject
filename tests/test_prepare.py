from __future__ import annotations

import pandas as pd

from src.prepare import filter_rows, modeling_frame
from src.schema import INPUT_COLUMNS, LEAKAGE_COLUMNS, POST_BOOKING_COLUMNS, TARGET
from tests.synthetic import synthetic_bookings


def test_modeling_frame_drops_leakage_and_post_booking_columns() -> None:
    prepared = modeling_frame(synthetic_bookings(), adr_cap=1000)
    for column in LEAKAGE_COLUMNS + POST_BOOKING_COLUMNS:
        assert column not in prepared.columns
    assert TARGET in prepared.columns
    assert list(prepared.columns) == [TARGET, *INPUT_COLUMNS]


def test_filter_drops_zero_guests_and_negative_price() -> None:
    frame = synthetic_bookings(rows=5)
    frame.loc[0, ["adults", "children", "babies"]] = 0
    frame.loc[1, "adr"] = -5
    cleaned = filter_rows(frame, adr_cap=1000)
    assert len(cleaned) == 3


def test_filter_caps_adr_and_merges_undefined_meal() -> None:
    frame = synthetic_bookings(rows=4)
    frame.loc[0, "adr"] = 5400
    frame.loc[0, "meal"] = "Undefined"
    cleaned = filter_rows(frame, adr_cap=1000)
    assert float(cleaned.iloc[0]["adr"]) == 1000
    assert "Undefined" not in set(cleaned["meal"])
    assert (cleaned["meal"] == "SC").any()
