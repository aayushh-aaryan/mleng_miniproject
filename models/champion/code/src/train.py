"""Train the champion configuration in params.yaml and register it."""

from __future__ import annotations

import pandas as pd

from src.modeling import fit_clean_model, param_section, save_champion
from src.paths import FEATURED_CSV, RAW_BOOKINGS
from src.prepare import load_params, read_bookings


def main() -> None:
    featured = pd.read_csv(FEATURED_CSV, nrows=2)
    if "total_nights" not in featured.columns:
        raise SystemExit("featurize output is missing total_nights; run the featurize stage first")
    params = load_params()
    train_params = param_section(params, "train")
    feature_params = param_section(params, "features")
    estimator_name = str(train_params["model"])
    country_encoding = str(feature_params["country_encoding"])
    data_version = str(train_params["data_version"])
    pipeline, scores = fit_clean_model(
        read_bookings(RAW_BOOKINGS),
        estimator_name=estimator_name,
        country_encoding=country_encoding,
        params=params,
        data_version=data_version,
    )
    info = save_champion(
        pipeline,
        scores,
        estimator_name=estimator_name,
        country_encoding=country_encoding,
        data_version=data_version,
        params=params,
        run_name=f"champion-{estimator_name}-{country_encoding}-{data_version}",
        tags={
            "status": "champion",
            "data_version": data_version,
            "estimator": estimator_name,
            "country_encoding": country_encoding,
        },
    )
    metrics = info["metrics"]
    print(
        f"champion {info['estimator']} {info['country_encoding']} "
        f"data={info['data_version']} version={info['model_version']} metrics={metrics}"
    )


if __name__ == "__main__":
    main()
