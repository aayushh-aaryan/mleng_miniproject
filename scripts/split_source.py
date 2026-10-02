"""Split the Hotel Booking Demand file into v1, v2, and the summer 2017 holdout."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import BOOKINGS_V1, BOOKINGS_V2, HOLDOUT_CSV, RAW_BOOKINGS, SOURCE_CSV, VERSIONS_DIR
from src.prepare import read_bookings
from src.split import split_bookings


def main() -> None:
    if not SOURCE_CSV.exists():
        raise SystemExit(f"missing source file: {SOURCE_CSV}")
    frame = read_bookings(SOURCE_CSV)
    parts = split_bookings(frame)
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_BOOKINGS.parent.mkdir(parents=True, exist_ok=True)
    parts["v1"].to_csv(BOOKINGS_V1, index=False)
    parts["v2"].to_csv(BOOKINGS_V2, index=False)
    parts["holdout"].to_csv(HOLDOUT_CSV, index=False)
    parts["v2"].to_csv(RAW_BOOKINGS, index=False)
    print(
        f"v1={len(parts['v1'])} v2={len(parts['v2'])} holdout={len(parts['holdout'])} "
        f"cancel_v2={parts['v2']['is_canceled'].mean():.3f} "
        f"cancel_holdout={parts['holdout']['is_canceled'].mean():.3f}"
    )


if __name__ == "__main__":
    main()
