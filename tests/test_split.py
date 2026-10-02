from __future__ import annotations

import pandas as pd

from src.schema import SUMMER_MONTHS
from src.split import split_bookings


def test_summer_2017_is_held_out_of_both_training_versions() -> None:
    frame = pd.DataFrame(
        {
            "arrival_date_year": [2015, 2016, 2017, 2017, 2017, 2017],
            "arrival_date_month": ["July", "December", "March", "May", "June", "August"],
            "is_canceled": [0, 1, 0, 1, 0, 1],
        }
    )
    parts = split_bookings(frame)
    assert list(parts["v1"]["arrival_date_year"]) == [2015, 2016]
    assert len(parts["v2"]) == 4
    assert set(parts["holdout"]["arrival_date_month"]) == {"June", "August"}
    v2_in_2017 = parts["v2"].loc[parts["v2"]["arrival_date_year"] == 2017, "arrival_date_month"]
    assert set(v2_in_2017) == {"March", "May"}
    assert not set(v2_in_2017).intersection(SUMMER_MONTHS)
