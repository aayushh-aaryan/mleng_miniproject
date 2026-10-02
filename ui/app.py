"""Booking desk. The form scores a booking through the API, never through a second model."""

from __future__ import annotations

import json
import os

import requests
import streamlit as st

from src.paths import EXAMPLES_PATH
from src.schema import MONTHS

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")

HOTELS = ["City Hotel", "Resort Hotel"]
MEALS = ["BB", "HB", "FB", "SC"]
MARKETS = ["Online TA", "Offline TA/TO", "Groups", "Direct", "Corporate", "Complementary", "Aviation"]
CHANNELS = ["TA/TO", "Direct", "Corporate", "GDS"]
DEPOSITS = ["No Deposit", "Non Refund", "Refundable"]
CUSTOMERS = ["Transient", "Transient-Party", "Contract", "Group"]
ROOMS = ["A", "B", "C", "D", "E", "F", "G", "H"]
COUNTRIES = ["PRT", "GBR", "FRA", "ESP", "DEU", "ITA", "IRL", "BEL", "BRA", "NLD", "USA", "CHE"]

FALLBACK_EXAMPLES: dict[str, dict[str, object]] = {
    "risky": {
        "title": "Long-lead non-refundable booking",
        "booking": {
            "hotel": "City Hotel",
            "lead_time": 300,
            "arrival_date_year": 2017,
            "arrival_date_month": "July",
            "arrival_date_week_number": 28,
            "arrival_date_day_of_month": 15,
            "stays_in_weekend_nights": 2,
            "stays_in_week_nights": 5,
            "adults": 2,
            "children": 0,
            "babies": 0,
            "meal": "BB",
            "country": "PRT",
            "market_segment": "Online TA",
            "distribution_channel": "TA/TO",
            "is_repeated_guest": 0,
            "previous_cancellations": 1,
            "previous_bookings_not_canceled": 0,
            "reserved_room_type": "A",
            "deposit_type": "Non Refund",
            "agent": "9",
            "company": None,
            "customer_type": "Transient",
            "adr": 120.0,
            "required_car_parking_spaces": 0,
            "total_of_special_requests": 0,
        },
    },
    "safe": {
        "title": "Repeat guest with parking and a special request",
        "booking": {
            "hotel": "Resort Hotel",
            "lead_time": 7,
            "arrival_date_year": 2017,
            "arrival_date_month": "March",
            "arrival_date_week_number": 12,
            "arrival_date_day_of_month": 18,
            "stays_in_weekend_nights": 1,
            "stays_in_week_nights": 2,
            "adults": 2,
            "children": 1,
            "babies": 0,
            "meal": "HB",
            "country": "GBR",
            "market_segment": "Direct",
            "distribution_channel": "Direct",
            "is_repeated_guest": 1,
            "previous_cancellations": 0,
            "previous_bookings_not_canceled": 2,
            "reserved_room_type": "D",
            "deposit_type": "No Deposit",
            "agent": None,
            "company": None,
            "customer_type": "Transient",
            "adr": 95.0,
            "required_car_parking_spaces": 1,
            "total_of_special_requests": 2,
        },
    },
}


def load_examples() -> dict[str, dict[str, object]]:
    if not EXAMPLES_PATH.exists():
        return FALLBACK_EXAMPLES
    loaded = json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or "risky" not in loaded or "safe" not in loaded:
        return FALLBACK_EXAMPLES
    return loaded


def with_current(options: list[str], current: object) -> list[str]:
    text = str(current)
    if text not in options:
        return [text, *options]
    return options


def option_index(options: list[str], current: object) -> tuple[list[str], int]:
    choices = with_current(options, current)
    return choices, choices.index(str(current))


def fetch_model_info() -> dict[str, object] | None:
    try:
        response = requests.get(f"{API_URL}/model-info", timeout=5)
        response.raise_for_status()
    except requests.RequestException:
        return None
    payload = response.json()
    if isinstance(payload, dict):
        return payload
    return None


def render_sidebar(examples: dict[str, dict[str, object]]) -> None:
    st.sidebar.header("Live model")
    info = fetch_model_info()
    if info is None:
        st.sidebar.warning("The scoring API is not responding.")
    else:
        st.sidebar.metric("Version", str(info.get("model_version", "—")))
        st.sidebar.caption(f"{info.get('estimator')} · data {info.get('data_version')}")
        st.sidebar.caption(f"Country encoding: {info.get('country_encoding')}")
        metrics = info.get("metrics")
        if isinstance(metrics, dict):
            st.sidebar.write(f"ROC-AUC {float(metrics.get('roc_auc', 0)):.3f}")
            st.sidebar.write(f"PR-AUC {float(metrics.get('pr_auc', 0)):.3f}")
            st.sidebar.write(f"High-risk recall {float(metrics.get('recall_high_risk', 0)):.3f}")
    st.sidebar.divider()
    st.sidebar.header("Demo bookings")
    if st.sidebar.button("Load risky booking", use_container_width=True):
        st.session_state.example_key = "risky"
        st.session_state.form_id = int(st.session_state.get("form_id", 0)) + 1
        st.rerun()
    if st.sidebar.button("Load safe booking", use_container_width=True):
        st.session_state.example_key = "safe"
        st.session_state.form_id = int(st.session_state.get("form_id", 0)) + 1
        st.rerun()
    current = examples[str(st.session_state.get("example_key", "safe"))]
    title = current.get("title")
    if isinstance(title, str):
        st.sidebar.caption(title)


def main() -> None:
    st.set_page_config(page_title="Hotel Cancel Guard", layout="wide")
    st.markdown(
        """
        <style>
        .risk-card {border: 1px solid rgba(28, 36, 36, 0.15); border-radius: 8px; padding: 1.25rem 1.4rem; margin-top: 1rem;}
        .risk-low {background: #e7f2ea;}
        .risk-medium {background: #f8f1df;}
        .risk-high {background: #f8e6e2;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    examples = load_examples()
    if "example_key" not in st.session_state:
        st.session_state.example_key = "safe"
    if "form_id" not in st.session_state:
        st.session_state.form_id = 0
    render_sidebar(examples)
    selected = examples[str(st.session_state.example_key)]
    booking = selected["booking"]
    if not isinstance(booking, dict):
        booking = FALLBACK_EXAMPLES["safe"]["booking"]
        if not isinstance(booking, dict):
            raise RuntimeError("fallback booking is missing")

    st.title("Hotel Cancel Guard")
    st.caption("Score a booking with only what the desk knows at the moment it is made.")

    with st.form(f"booking-{st.session_state.form_id}"):
        st.subheader("Stay")
        stay_left, stay_right = st.columns(2)
        hotel_options, hotel_index = option_index(HOTELS, booking["hotel"])
        hotel = stay_left.selectbox("Hotel", hotel_options, index=hotel_index)
        lead_time = stay_right.number_input("Lead time (days)", min_value=0, max_value=1000, value=int(booking["lead_time"]))
        month = stay_left.selectbox(
            "Arrival month",
            MONTHS,
            index=list(MONTHS).index(str(booking["arrival_date_month"])),
        )
        year = stay_right.number_input("Arrival year", min_value=2015, max_value=2035, value=int(booking["arrival_date_year"]))
        day = stay_left.number_input("Arrival day", min_value=1, max_value=31, value=int(booking["arrival_date_day_of_month"]))
        week = stay_right.number_input(
            "Arrival week number",
            min_value=1,
            max_value=53,
            value=int(booking["arrival_date_week_number"]),
        )
        weekend_nights = stay_left.number_input(
            "Weekend nights", min_value=0, max_value=30, value=int(booking["stays_in_weekend_nights"])
        )
        week_nights = stay_right.number_input(
            "Week nights", min_value=0, max_value=50, value=int(booking["stays_in_week_nights"])
        )
        room_options, room_index = option_index(ROOMS, booking["reserved_room_type"])
        room = st.selectbox("Reserved room type", room_options, index=room_index)

        st.subheader("Party")
        party_left, party_mid, party_right = st.columns(3)
        adults = party_left.number_input("Adults", min_value=0, max_value=10, value=int(booking["adults"]))
        children = party_mid.number_input("Children", min_value=0, max_value=10, value=int(booking["children"]))
        babies = party_right.number_input("Babies", min_value=0, max_value=5, value=int(booking["babies"]))

        st.subheader("Commercial terms")
        commercial_left, commercial_right = st.columns(2)
        meal_options, meal_index = option_index(MEALS, booking["meal"])
        meal = commercial_left.selectbox("Meal", meal_options, index=meal_index)
        deposit_options, deposit_index = option_index(DEPOSITS, booking["deposit_type"])
        deposit = commercial_right.selectbox("Deposit", deposit_options, index=deposit_index)
        market_options, market_index = option_index(MARKETS, booking["market_segment"])
        market = commercial_left.selectbox("Market segment", market_options, index=market_index)
        channel_options, channel_index = option_index(CHANNELS, booking["distribution_channel"])
        channel = commercial_right.selectbox("Distribution channel", channel_options, index=channel_index)
        customer_options, customer_index = option_index(CUSTOMERS, booking["customer_type"])
        customer = commercial_left.selectbox("Customer type", customer_options, index=customer_index)
        country_options, country_index = option_index(COUNTRIES, booking["country"])
        country = commercial_right.selectbox("Country", country_options, index=country_index)
        adr = commercial_left.number_input("Average daily rate", min_value=0.0, max_value=10000.0, value=float(booking["adr"]))
        parking = commercial_right.number_input(
            "Parking spaces", min_value=0, max_value=8, value=int(booking["required_car_parking_spaces"])
        )
        requests_count = st.number_input(
            "Special requests", min_value=0, max_value=10, value=int(booking["total_of_special_requests"])
        )

        st.subheader("History")
        history_left, history_right = st.columns(2)
        repeated = history_left.selectbox(
            "Repeated guest",
            [0, 1],
            index=int(booking["is_repeated_guest"]),
        )
        previous_cancellations = history_right.number_input(
            "Previous cancellations",
            min_value=0,
            max_value=50,
            value=int(booking["previous_cancellations"]),
        )
        previous_kept = history_left.number_input(
            "Previous stays kept",
            min_value=0,
            max_value=50,
            value=int(booking["previous_bookings_not_canceled"]),
        )
        agent_default = "" if booking.get("agent") in (None, "") else str(booking["agent"])
        company_default = "" if booking.get("company") in (None, "") else str(booking["company"])
        agent = history_right.text_input("Agent id (blank if none)", value=agent_default)
        company = history_left.text_input("Company id (blank if none)", value=company_default)
        submitted = st.form_submit_button("Score this booking", use_container_width=True)

    if submitted:
        payload = {
            "hotel": hotel,
            "lead_time": int(lead_time),
            "arrival_date_year": int(year),
            "arrival_date_month": month,
            "arrival_date_week_number": int(week),
            "arrival_date_day_of_month": int(day),
            "stays_in_weekend_nights": int(weekend_nights),
            "stays_in_week_nights": int(week_nights),
            "adults": int(adults),
            "children": int(children),
            "babies": int(babies),
            "meal": meal,
            "country": country,
            "market_segment": market,
            "distribution_channel": channel,
            "is_repeated_guest": int(repeated),
            "previous_cancellations": int(previous_cancellations),
            "previous_bookings_not_canceled": int(previous_kept),
            "reserved_room_type": room,
            "deposit_type": deposit,
            "agent": agent.strip() or None,
            "company": company.strip() or None,
            "customer_type": customer,
            "adr": float(adr),
            "required_car_parking_spaces": int(parking),
            "total_of_special_requests": int(requests_count),
        }
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        except requests.RequestException as exc:
            st.error(f"Could not reach the API at {API_URL}. {exc}")
            return
        if response.status_code == 422:
            detail = response.json().get("detail", "The booking was rejected.")
            st.error(f"That booking is not valid. {detail}")
            return
        if response.status_code != 200:
            st.error(f"The API returned {response.status_code}.")
            return
        result = response.json()
        band = str(result["risk_band"])
        probability = float(result["cancellation_probability"])
        st.markdown(
            f"<div class='risk-card risk-{band}'>"
            f"<p style='margin:0;letter-spacing:0.04em;text-transform:uppercase;font-size:0.8rem;'>{band} risk</p>"
            f"<p style='margin:0.2rem 0 0;font-size:2rem;font-weight:600;'>{probability:.1%}</p>"
            f"<p style='margin:0.4rem 0 0;'>{result['suggested_action']}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
else:
    main()
