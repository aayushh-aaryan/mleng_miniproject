"""Tiny bookings with both classes, used by the unit tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.schema import INPUT_COLUMNS, LEAKAGE_COLUMNS, POST_BOOKING_COLUMNS, TARGET


def synthetic_bookings(rows: int = 90, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    months = ["January", "April", "July", "December"]
    countries = ["PRT", "GBR", "FRA", "ESP", "DEU"]
    for index in range(rows):
        deposit = ("No Deposit", "Non Refund", "Refundable")[index % 3]
        lead_time = int(rng.integers(0, 320))
        canceled = int(deposit == "Non Refund" or lead_time > 220)
        records.append(
            {
                TARGET: canceled,
                "hotel": "City Hotel" if index % 2 == 0 else "Resort Hotel",
                "lead_time": lead_time,
                "arrival_date_year": 2016,
                "arrival_date_month": months[index % len(months)],
                "arrival_date_week_number": (index % 52) + 1,
                "arrival_date_day_of_month": (index % 27) + 1,
                "stays_in_weekend_nights": index % 3,
                "stays_in_week_nights": index % 5,
                "adults": 2,
                "children": index % 2,
                "babies": 0,
                "meal": ("BB", "HB", "SC", "Undefined")[index % 4],
                "country": countries[index % len(countries)],
                "market_segment": "Online TA" if index % 2 == 0 else "Direct",
                "distribution_channel": "TA/TO" if index % 2 == 0 else "Direct",
                "is_repeated_guest": index % 5 == 0,
                "previous_cancellations": index % 3,
                "previous_bookings_not_canceled": index % 4,
                "reserved_room_type": ("A", "D", "E")[index % 3],
                "deposit_type": deposit,
                "agent": None if index % 7 == 0 else str(9 + (index % 4)),
                "company": None if index % 3 else "40",
                "customer_type": "Transient" if index % 2 == 0 else "Contract",
                "adr": float(40 + (index % 25) * 8),
                "required_car_parking_spaces": index % 2,
                "total_of_special_requests": index % 4,
                "reservation_status": "Canceled" if canceled else "Check-Out",
                "reservation_status_date": "2016-08-01",
                "assigned_room_type": "B" if index % 4 == 0 else "A",
                "booking_changes": index % 3,
                "days_in_waiting_list": index % 5,
            }
        )
    frame = pd.DataFrame.from_records(records)
    frame.loc[0, TARGET] = 0
    frame.loc[1, TARGET] = 1
    return frame


def valid_payload() -> dict[str, object]:
    frame = synthetic_bookings(rows=1)
    row = frame.iloc[0]
    payload: dict[str, object] = {}
    for column in INPUT_COLUMNS:
        value = row[column]
        if column in {"agent", "company"}:
            payload[column] = None if pd.isna(value) else str(value)
        elif column == "adr":
            payload[column] = float(value)
        elif column in {"hotel", "arrival_date_month", "meal", "country", "market_segment", "distribution_channel", "reserved_room_type", "deposit_type", "customer_type"}:
            payload[column] = str(value)
        else:
            payload[column] = int(value)
    payload["adults"] = 2
    payload["arrival_date_day_of_month"] = 15
    payload["arrival_date_month"] = "July"
    payload["arrival_date_year"] = 2016
    return payload


BANNED_FROM_MODEL = LEAKAGE_COLUMNS + POST_BOOKING_COLUMNS + ("arrival_date_year",)
