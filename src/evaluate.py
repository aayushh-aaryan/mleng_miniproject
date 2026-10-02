"""Score the saved champion on the fixed summer 2017 holdout."""

from __future__ import annotations

import json

import mlflow

from src.modeling import holdout_scores, param_section, write_json
from src.paths import EVAL_METRICS, MODEL_DIR
from src.prepare import load_params


def main() -> None:
    params = load_params()
    prepare_params = param_section(params, "prepare")
    risk_params = param_section(params, "risk")
    pipeline = mlflow.sklearn.load_model(str(MODEL_DIR))
    scores = holdout_scores(
        pipeline,
        float(prepare_params["adr_cap"]),
        float(risk_params["deposit_threshold"]),
    )
    write_json(EVAL_METRICS, scores)
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
