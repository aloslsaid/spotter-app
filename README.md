# Spotter — Freight Rate Prediction

Spotter is a freight-rate regression project. It engineers route and calendar features from load data, trains a scikit-learn regression pipeline, evaluates it on a held-out split, then refits on all labeled training data to produce validation-file predictions.

## Data files

The take-home datasets are intentionally not included in this repository. Place authorized files beside `train.py`:

- `train-test.csv`: labeled training records; must contain `posted_rate`.
- `validation.csv`: unlabeled records with the same feature schema.

The training script expects fields for pickup, delivery, equipment, date, distance and weight. Optional `market_index` and `quote_signal` fields are supported.

## Setup

Python 3.10+:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train.py
```

## Methodology

- Date-derived calendar features and a pickup-to-delivery route feature.
- Numeric median imputation and categorical most-frequent imputation inside the training pipeline.
- One-hot encoding with unknown-category handling.
- Held-out test metrics: R², MAE and RMSE.
- Final refit on all labeled rows, followed by validation predictions.

Preprocessing is fitted only on training data during evaluation; the validation file is transformed by the final fitted pipeline. Keep private customer, carrier and shipment data out of public Git repositories.

## Outputs

The script writes `artifacts/freight_rate_model.joblib`, `artifacts/metrics.json`, and `artifacts/validation_predictions.csv`. These are generated locally and ignored by Git.

Previously reported scores are historical project claims and are not asserted as reproduced by the current entry point. Always state the dataset version, units, split, and metrics.

## License

No license is declared. All rights reserved unless the repository owner adds an explicit license.
