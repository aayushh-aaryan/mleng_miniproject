"""Request and response models for the scoring API."""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, Field, field_validator, model_validator

from src.schema import INPUT_COLUMNS, MONTH_TO_NUM


class BookingRequest(BaseModel):
    hotel: str = Field(min_length=1, max_length=64)
    lead_time: int = Field(ge=0, le=1000)
    arrival_date_year: int = Field(ge=2015, le=2035)
    arrival_date_month: str
    arrival_date_week_number: int = Field(ge=1, le=53)
    arrival_date_day_of_month: int = Field(ge=1, le=31)
    stays_in_weekend_nights: int = Field(ge=0, le=30)
    stays_in_week_nights: int = Field(ge=0, le=50)
    adults: int = Field(ge=0, le=10)
    children: int = Field(ge=0, le=10)
    babies: int = Field(ge=0, le=5)
    meal: str = Field(min_length=1, max_length=32)
    country: str = Field(min_length=2, max_length=32)
    market_segment: str = Field(min_length=1, max_length=64)
    distribution_channel: str = Field(min_length=1, max_length=64)
    is_repeated_guest: int = Field(ge=0, le=1)
    previous_cancellations: int = Field(ge=0, le=50)
    previous_bookings_not_canceled: int = Field(ge=0, le=50)
    reserved_room_type: str = Field(min_length=1, max_length=8)
    deposit_type: str = Field(min_length=1, max_length=32)
    agent: str | None = None
    company: str | None = None
    customer_type: str = Field(min_length=1, max_length=32)
    adr: float = Field(ge=0, le=10000)
    required_car_parking_spaces: int = Field(ge=0, le=8)
    total_of_special_requests: int = Field(ge=0, le=10)

    @field_validator("arrival_date_month")
    @classmethod
    def known_month(cls, value: str) -> str:
        if value not in MONTH_TO_NUM:
            raise ValueError("arrival_date_month must be a full English month name")
        return value

    @field_validator("agent", "company", mode="before")
    @classmethod
    def blank_id_is_missing(cls, value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if text == "" or text.lower() in {"none", "null", "nan"}:
            return None
        return text

    @model_validator(mode="after")
    def booking_is_possible(self) -> BookingRequest:
        if self.adults + self.children + self.babies <= 0:
            raise ValueError("at least one guest is required")
        month = MONTH_TO_NUM[self.arrival_date_month]
        try:
            dt.date(self.arrival_date_year, month, self.arrival_date_day_of_month)
        except ValueError as exc:
            raise ValueError("arrival date is not a real calendar date") from exc
        return self

    def feature_row(self) -> dict[str, object]:
        payload = self.model_dump()
        return {column: payload[column] for column in INPUT_COLUMNS}


class PredictionResponse(BaseModel):
    cancellation_probability: float
    risk_band: str
    suggested_action: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    alias: str
    data_version: str
    estimator: str
    country_encoding: str
    metrics: dict[str, float]
    risk_bands: dict[str, float]

    @field_validator("model_version", mode="before")
    @classmethod
    def version_as_text(cls, value: object) -> str:
        return str(value)
