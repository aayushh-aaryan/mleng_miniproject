"""Columns the manager can know when the booking is made.

Anything recorded only after that moment is omitted on purpose.
`arrival_date_year` is accepted as an input so the weekday can be derived,
then dropped before the model sees it. A future year must not be a feature.
"""

from __future__ import annotations

MONTHS: tuple[str, ...] = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)

MONTH_TO_NUM: dict[str, int] = {name: index for index, name in enumerate(MONTHS, start=1)}

SUMMER_MONTHS: frozenset[str] = frozenset({"June", "July", "August"})
EARLY_2017_MONTHS: frozenset[str] = frozenset(
    {"January", "February", "March", "April", "May"}
)

TARGET = "is_canceled"

# The outcome, written after the stay ends. Keeping these produces a fake 99%.
LEAKAGE_COLUMNS: tuple[str, ...] = (
    "reservation_status",
    "reservation_status_date",
)

# Filled in at check-in or when the booking is later changed.
POST_BOOKING_COLUMNS: tuple[str, ...] = (
    "assigned_room_type",
    "booking_changes",
    "days_in_waiting_list",
)

INPUT_COLUMNS: tuple[str, ...] = (
    "hotel",
    "lead_time",
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "reserved_room_type",
    "deposit_type",
    "agent",
    "company",
    "customer_type",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
)

NA_VALUES: tuple[str, ...] = ("NULL", "null", "None", "none", "NaN", "nan", "")
