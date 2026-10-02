# Hotel Cancel Guard

A booking-time cancellation score for a hotel desk. The model only sees what the manager knows when the reservation is made. It returns a probability, a risk band, and a suggested action.

About 37% of bookings in the Hotel Booking Demand dataset cancel. A model that always says "will stay" is already about 63% accurate, so accuracy is not the metric. Runs are compared on ROC-AUC and PR-AUC, plus recall among bookings flagged as high risk.

## Results

Scored on the cleaned June–August 2017 holdout (15,872 bookings, 39.3% canceled). A model that always says "will stay" is 60.7% accurate on that window. On the full 119,390 rows the cancellation rate is 37.0%, which is the 63% accuracy trap.

| Run | ROC-AUC | PR-AUC | Recall at 60% |
| --- | ---: | ---: | ---: |
| Leaky baseline (do not use) | 1.000 | 1.000 | 1.000 |
| Post-booking columns (do not use) | 0.883 | 0.825 | 0.544 |
| Logistic regression | 0.843 | 0.782 | 0.467 |
| Random forest | 0.865 | 0.799 | 0.446 |
| LightGBM, top-20 countries | 0.875 | 0.813 | 0.531 |
| LightGBM, frequency encoding, data v1 | 0.862 | 0.797 | 0.501 |
| LightGBM, frequency encoding, data v2 (champion) | 0.875 | 0.813 | 0.532 |

Frequency encoding beat top-20 one-hot on PR-AUC by a very small margin, so that is the encoding in the champion. Data v2 improved PR-AUC from 0.797 to 0.813 against the same future summer. The champion's high-risk flags are right about 80% of the time and catch 53% of cancellations.

Gain on the leaky model sits almost entirely on `reservation_status`. Gain on the champion is led by deposit type, lead time, the agent and country frequencies, special requests, the guest's past cancellation ratio, market segment, and parking.

The live alias is `champion` on model `hotel-cancellation`, version 2. `dvc repro` refit the same LightGBM pipeline and moved the alias onto that run. The holdout scores did not change.

## The rule

Use only information available at the moment of booking.

| Column | Decision | Why |
| --- | --- | --- |
| `reservation_status`, `reservation_status_date` | Drop | These record the outcome. Keeping them produces a fake score near 99%. |
| `assigned_room_type`, `booking_changes`, `days_in_waiting_list` | Test, then drop | They are filled in at check-in or as the booking changes later. |
| `arrival_date_year` | Drop as a feature | A future year is a number the model has never seen. The year is used only to derive the weekday. |
| `company` (about 94% empty) | `has_company` flag | Empty means "not a corporate booking." |
| `agent` (about 14% empty) | `has_agent` plus frequency of the id | Agent ids are labels. Agent 240 is not "more" than agent 9. |
| `country` | Frequency, or top 20 plus Other | Full one-hot adds a sparse column per country. The holdout comparison picks the encoding. |
| Other categoricals | One-hot with `handle_unknown='ignore'` | Few values each, and an unseen category does not crash scoring. |
| `lead_time`, `adr` | `log1p` and `RobustScaler`, logistic regression only | Both are skewed, and ADR has an extreme outlier. Trees do not get this transform. |
| Arrival month, week, weekday | Sine and cosine | December sits next to January, and Sunday sits next to Monday. |
| New features | `total_nights`, `total_guests`, `is_family`, past cancellation ratio | Each one is a quantity the desk already understands. |
| PCA | Not used | There are few numeric columns of mixed types, and the desk needs a reason, not a component. |

Cleaning drops rows with zero guests or a negative price, caps ADR at 1000, and merges `Undefined` into `SC` because both mean "no meal package."

The split is by time. June–August 2017 is a fixed holdout. No model trains on it. `data/raw/bookings.csv` version 1 is 2015–2016. Version 2 is 2015 through May 2017.

## Layout

```
hotel-cancel-guard/
├── data/raw/                 # DVC-tracked bookings and the summer holdout
├── notebooks/eda.ipynb
├── src/prepare.py            # cleaning and leakage removal
├── src/features.py           # sklearn transformers
├── src/train.py              # champion training and MLflow registration
├── src/evaluate.py
├── src/experiments.py        # leaky baseline, clean models, v1 vs v2
├── api/main.py
├── ui/app.py
├── tests/
├── dvc.yaml
├── params.yaml               # hyperparameters, country encoding, risk cutoffs
├── Dockerfile.api
├── Dockerfile.ui
├── docker-compose.yml
└── .github/workflows/ci.yml
```

`params.yaml` holds the model hyperparameters, the country encoding, and the risk cutoffs (0.3 and 0.6). Changing a cutoff is a versioned change, same as changing a hyperparameter.

## Risk bands

| Probability | Band | Action |
| --- | --- | --- |
| Under 30% | Low | Confirm the booking as usual. |
| 30% through 60% | Medium | Send a reminder before arrival. |
| Over 60% | High | Ask for a deposit, or count the booking toward overbooking. |

## Setup

Python 3.11 is the version used in CI and in the images. Git is required. Docker Desktop is required for the compose demo.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The raw Hotel Booking Demand file is already split in this handoff. To rebuild the split from `data/raw/hotels_source.csv`:

```bash
python scripts/split_source.py
```

## DVC

DVC answers "which data and code produced this model?"

```bash
dvc repro
dvc metrics diff data-v1 data-v2
```

`dvc repro` runs prepare, featurize, train, and evaluate. The training stage fits the champion configuration in `params.yaml` and registers it. The evaluate stage writes `metrics/eval.json` from the summer 2017 holdout.

To use a DagsHub remote instead of a laptop disk:

```bash
dvc remote add -d origin https://dagshub.com/<user>/hotel-cancel-guard.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <user>
dvc remote modify origin --local password <token>
dvc push
```

Mirror the GitHub repo on DagsHub first. That project includes the DVC remote and an MLflow tracking server the GitHub Action can reach.

```python
import dagshub
dagshub.init(repo_owner="<user>", repo_name="hotel-cancel-guard", mlflow=True)
```

## MLflow

MLflow answers "which run was best, and which one is live?"

Logged runs:

- Leaky baseline, tagged `do-not-use`. It is allowed to see `reservation_status`.
- Post-booking columns, also tagged `do-not-use`.
- Logistic regression, the interpretable floor. Skewed columns are log-scaled here only.
- Random forest.
- LightGBM with frequency-encoded country, and LightGBM with top-20 one-hot.
- The winning recipe trained on data v1 and on data v2.

The saved artifact is one sklearn `Pipeline`: feature builder, encoder, and model. The API loads that object, so serving cannot drift from training.

The registry name is `hotel-cancellation`. The best version carries the alias `champion`. This project uses aliases rather than the retired Staging and Production stages.

Local tracking is a SQLite store:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Probabilities are left unweighted. The risk bands are business thresholds, so the fit is not rebalanced with `class_weight`.

## API

```bash
uvicorn api.main:app --reload
```

| Method | Path | Behavior |
| --- | --- | --- |
| GET | `/health` | Liveness. |
| GET | `/model-info` | Version, data version, encoding, and holdout metrics. |
| POST | `/predict` | Validated with Pydantic. `adults = -2` returns 422. |

`/predict` returns `cancellation_probability`, `risk_band`, and `suggested_action`.

## Desk UI

```bash
streamlit run ui/app.py
```

The form is grouped into stay, party, commercial terms, and history. The sidebar reads `/model-info`. **Load risky booking** and **Load safe booking** fill the form from real holdout rows so a live demo does not depend on typing.

`API_URL` defaults to `http://127.0.0.1:8000`. Compose sets it to `http://api:8000`.

The pitch deck is `pitch/index.html`. Open it in a browser and use the arrow keys.

## Docker

The champion model is copied into the API image, so the demo runs without a tracking server and without a network call at score time.

```bash
docker compose up --build
```

API on port 8000. UI on port 8501.

## GitHub Actions

On every push to `main`, pytest runs first. Images are built only if the tests pass. The workflow downloads `models:/hotel-cancellation@champion`, then pushes both images to Docker Hub as `latest` and as the commit SHA.

Repository secrets:

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`
- `MLFLOW_TRACKING_URI`
- `MLFLOW_TRACKING_USERNAME`
- `MLFLOW_TRACKING_PASSWORD`

## Tests

```bash
pytest
```

The tests cover four promises:

- Leakage columns never reach the model, and adding them does not change the score.
- An unseen country does not crash the pipeline.
- `/predict` returns a probability between 0 and 1.
- A negative adult count returns 422.

## Bonus: run it on EC2

Launch an Ubuntu instance, install Docker, and open ports 8000 and 8501.

```bash
docker login
export DOCKERHUB_USERNAME=<user>
docker pull $DOCKERHUB_USERNAME/hotel-cancel-guard-api:latest
docker pull $DOCKERHUB_USERNAME/hotel-cancel-guard-ui:latest
docker tag $DOCKERHUB_USERNAME/hotel-cancel-guard-api:latest hotel-cancel-guard-api
docker tag $DOCKERHUB_USERNAME/hotel-cancel-guard-ui:latest hotel-cancel-guard-ui
docker compose up -d
```

`docker compose pull` works once `image:` names in `docker-compose.yml` point at Docker Hub. The compose file in this repo builds locally and tags `hotel-cancel-guard-api` and `hotel-cancel-guard-ui`. The retag above matches those names.

## Demo order

1. Show the leaky run in MLflow and the `do-not-use` tag. That score is not a model of cancellation. It is a model of a column that already says the booking was canceled.
2. Show the clean comparison. Logistic regression is the floor. The champion is the best holdout PR-AUC.
3. Run `dvc metrics diff data-v1 data-v2`.
4. Open the Streamlit desk, load the risky example, then the safe example.
5. Send `adults: -2` to `/predict` and show the 422.
