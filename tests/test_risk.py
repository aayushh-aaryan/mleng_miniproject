from __future__ import annotations

from src.risk import band_for_probability


def test_risk_band_boundaries() -> None:
    assert band_for_probability(0.29, 0.3, 0.6)[0] == "low"
    assert band_for_probability(0.30, 0.3, 0.6)[0] == "medium"
    assert band_for_probability(0.60, 0.3, 0.6)[0] == "medium"
    assert band_for_probability(0.61, 0.3, 0.6)[0] == "high"
    assert "deposit" in band_for_probability(0.8, 0.3, 0.6)[1].lower()
    assert "reminder" in band_for_probability(0.4, 0.3, 0.6)[1].lower()
    assert "Confirm" in band_for_probability(0.1, 0.3, 0.6)[1]
