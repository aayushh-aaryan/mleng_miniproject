"""Build notebooks/eda.ipynb with the leakage and missing-value findings."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nbformat
import pandas as pd
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook, new_output

from src.paths import HOLDOUT_CSV, METRICS_DIR, SOURCE_CSV
from src.prepare import read_bookings
from src.schema import LEAKAGE_COLUMNS, POST_BOOKING_COLUMNS
from src.split import split_bookings


def stream(text: str):
    return new_output(output_type="stream", name="stdout", text=text if text.endswith("\n") else text + "\n")


def main() -> None:
    frame = read_bookings(SOURCE_CSV)
    parts = split_bookings(frame)
    cancel_rate = float(frame["is_canceled"].mean())
    missing = (frame.isna().mean() * 100).sort_values(ascending=False)
    missing = missing[missing > 0]
    status = pd.crosstab(frame["reservation_status"], frame["is_canceled"])
    adr_max = float(frame["adr"].max())
    meals = frame["meal"].fillna("missing").value_counts()
    assigned_diff = float((frame["assigned_room_type"] != frame["reserved_room_type"]).mean())
    changes = frame["booking_changes"].fillna(0)
    cancel_when_changed = float(frame.loc[changes > 0, "is_canceled"].mean())
    cancel_when_unchanged = float(frame.loc[changes == 0, "is_canceled"].mean())
    findings = {
        "rows": int(len(frame)),
        "cancel_rate": cancel_rate,
        "majority_class_accuracy": 1 - cancel_rate,
        "company_missing_pct": float(frame["company"].isna().mean() * 100),
        "agent_missing_pct": float(frame["agent"].isna().mean() * 100),
        "country_missing_pct": float(frame["country"].isna().mean() * 100),
        "children_missing_pct": float(frame["children"].isna().mean() * 100),
        "adr_max": adr_max,
        "adr_p99": float(frame["adr"].quantile(0.99)),
        "undefined_meals": int((frame["meal"] == "Undefined").sum()),
        "assigned_differs_pct": assigned_diff * 100,
        "cancel_rate_when_booking_changed": cancel_when_changed,
        "cancel_rate_when_booking_unchanged": cancel_when_unchanged,
        "v1_rows": int(len(parts["v1"])),
        "v2_rows": int(len(parts["v2"])),
        "holdout_rows": int(len(parts["holdout"])),
        "holdout_cancel_rate": float(parts["holdout"]["is_canceled"].mean()),
        "v2_cancel_rate": float(parts["v2"]["is_canceled"].mean()),
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "eda.json").write_text(json.dumps(findings, indent=2), encoding="utf-8")

    cells = [
        new_markdown_cell(
            "# Hotel booking cancellations\n\n"
            "Question: which bookings will cancel, using only what the desk knows when the reservation is made?\n\n"
            "Source: Hotel Booking Demand (Antonio, Almeida, and Nunes), two hotels in Portugal, arrivals from July 2015 through August 2017."
        ),
        new_code_cell(
            "import pandas as pd\n"
            "from src.prepare import read_bookings\n"
            "from src.paths import SOURCE_CSV\n"
            "from src.split import split_bookings\n\n"
            "bookings = read_bookings(SOURCE_CSV)\n"
            "print(bookings.shape)\n"
            "print(bookings['is_canceled'].mean())",
            outputs=[
                stream(f"{frame.shape}\n{cancel_rate}\n")
            ],
        ),
        new_markdown_cell(
            f"## Accuracy is the wrong headline\n\n"
            f"{cancel_rate:.1%} of bookings cancel, so a model that always says \"will stay\" "
            f"is already {1 - cancel_rate:.1%} accurate. ROC-AUC and PR-AUC are the comparison metrics. "
            f"Recall at the high-risk cutoff is what the desk acts on."
        ),
        new_markdown_cell(
            "## `reservation_status` is the outcome\n\n"
            "Check-Out lines up with stays. Canceled and No-Show line up with cancellations. "
            "Keeping this column, or the date it was written, produces a fake score near 99%."
        ),
        new_code_cell(
            "pd.crosstab(bookings['reservation_status'], bookings['is_canceled'])",
            outputs=[
                new_output(
                    output_type="execute_result",
                    data={"text/plain": status.to_string()},
                    metadata={},
                    execution_count=2,
                )
            ],
        ),
        new_markdown_cell(
            "## Missing values are not all the same kind of gap\n\n"
            f"Company is empty on {findings['company_missing_pct']:.1f}% of rows. "
            "That gap means \"not a corporate booking,\" so it becomes a yes/no flag rather than an imputed id.\n\n"
            f"Agent is empty on {findings['agent_missing_pct']:.1f}% of rows. "
            "The id is a label, so the model gets a has-agent flag plus the training frequency of that id. "
            "Agent 240 is not a larger quantity than agent 9.\n\n"
            f"Country is missing on {findings['country_missing_pct']:.2f}% of rows and has well over 100 distinct values. "
            "Full one-hot would add a sparse column per country. Frequency encoding and a top-20 plus Other encoding are both logged, and the holdout picks one.\n\n"
            f"Children is missing on {findings['children_missing_pct']:.2f}% of rows and is treated as zero guests of that type."
        ),
        new_code_cell(
            "missing = bookings.isna().mean().sort_values(ascending=False)\nprint((missing[missing > 0] * 100).round(2))",
            outputs=[stream((missing.round(2)).to_string() + "\n")],
        ),
        new_markdown_cell(
            f"## Price and meal cleaning\n\n"
            f"The maximum ADR is {adr_max:.0f}, against a 99th percentile of {findings['adr_p99']:.0f}. "
            "That single outlier is capped at 1000. Rows with a negative ADR, and rows with zero guests, are dropped.\n\n"
            f"`meal` contains {findings['undefined_meals']} `Undefined` values. "
            "The dataset notes say Undefined and SC both mean no meal package, so they are merged to SC."
        ),
        new_code_cell(
            "print(bookings['adr'].describe())\nprint(bookings['meal'].value_counts(dropna=False))",
            outputs=[
                stream(frame["adr"].describe().to_string() + "\n" + meals.to_string() + "\n")
            ],
        ),
        new_markdown_cell(
            "## Columns filled in after the booking\n\n"
            f"Assigned room differs from the reserved room on {assigned_diff:.1%} of rows. "
            "That reassignment happens at check-in, not when the manager takes the booking.\n\n"
            f"When `booking_changes` is already above zero, the cancellation rate is {cancel_when_changed:.1%}. "
            f"When it is still zero, the rate is {cancel_when_unchanged:.1%}. "
            "Changes accumulate during the life of the booking, so the column is not available at decision time. "
            "`days_in_waiting_list` has the same problem. A later experiment measures how much they inflate the score, then they are left out."
        ),
        new_markdown_cell(
            "## The test set is a future summer\n\n"
            f"v1 is 2015–2016 ({findings['v1_rows']:,} rows). "
            f"v2 adds January–May 2017 ({findings['v2_rows']:,} rows). "
            f"June–August 2017 ({findings['holdout_rows']:,} rows, {findings['holdout_cancel_rate']:.1%} canceled) "
            "is never used for fitting. Every model is scored on that same window.\n\n"
            "`arrival_date_year` is not a model feature. A booking for a later year would otherwise be a number the model has never seen. "
            "The year is used only to derive the weekday, which is then encoded as sine and cosine, along with month and week, so December sits next to January."
        ),
        new_markdown_cell(
            "## What the model is allowed to see\n\n"
            "Dropped outright: " + ", ".join(LEAKAGE_COLUMNS) + ".\n\n"
            "Tested, then dropped: " + ", ".join(POST_BOOKING_COLUMNS) + ".\n\n"
            "Engineered because each one has a desk meaning: total nights, total guests, a family flag, weekday of arrival, and the guest's past cancellation ratio.\n\n"
            "PCA is not used. There are few numeric columns, they are different kinds of quantities, and a manager has to be able to say why a booking is risky."
        ),
    ]
    notebook = new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
    path = Path("notebooks/eda.ipynb")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(nbformat.writes(notebook), encoding="utf-8")
    print(f"wrote {path}")
    print(json.dumps(findings, indent=2))


if __name__ == "__main__":
    main()
