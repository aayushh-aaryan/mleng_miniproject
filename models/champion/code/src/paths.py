"""Project paths, resolved from this file so commands work from any cwd."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PARAMS_PATH = ROOT / "params.yaml"
RAW_DIR = ROOT / "data" / "raw"
VERSIONS_DIR = ROOT / "data" / "versions"
SOURCE_CSV = RAW_DIR / "hotels_source.csv"
RAW_BOOKINGS = RAW_DIR / "bookings.csv"
HOLDOUT_CSV = RAW_DIR / "holdout_summer2017.csv"
BOOKINGS_V1 = VERSIONS_DIR / "bookings_v1.csv"
BOOKINGS_V2 = VERSIONS_DIR / "bookings_v2.csv"
PREPARED_CSV = ROOT / "data" / "prepared" / "bookings.csv"
FEATURED_CSV = ROOT / "data" / "featured" / "bookings.csv"
MODEL_DIR = ROOT / "models" / "champion"
METRICS_DIR = ROOT / "metrics"
TRAIN_METRICS = METRICS_DIR / "train.json"
EVAL_METRICS = METRICS_DIR / "eval.json"
EXPERIMENT_SUMMARY = METRICS_DIR / "experiment_summary.json"
EXAMPLES_PATH = ROOT / "ui" / "examples.json"
MLFLOW_DB = ROOT / "mlflow.db"
