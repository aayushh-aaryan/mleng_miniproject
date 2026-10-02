"""Business thresholds live in params.yaml so a cutoff change is versioned."""

from __future__ import annotations

LOW_ACTION = "Confirm the booking as usual."
MEDIUM_ACTION = "Send a reminder before arrival."
HIGH_ACTION = "Ask for a deposit, or count this booking toward overbooking."


def band_for_probability(
    probability: float,
    reminder_threshold: float,
    deposit_threshold: float,
) -> tuple[str, str]:
    """Map a cancellation probability to a band and a suggested action.

    Under the reminder cutoff: confirm normally.
    From the reminder cutoff through the deposit cutoff: send a reminder.
    Above the deposit cutoff: ask for a deposit or count it toward overbooking.
    """
    if probability < reminder_threshold:
        return "low", LOW_ACTION
    if probability <= deposit_threshold:
        return "medium", MEDIUM_ACTION
    return "high", HIGH_ACTION
