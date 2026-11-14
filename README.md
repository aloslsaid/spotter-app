# Spotter — Freight Rate Prediction

> A supervised machine-learning workflow for estimating freight load rates from route, equipment, shipment, and market signals.

## Overview
Spotter is a tabular regression experiment built around freight-market data. It transforms shipment and route attributes into model-ready features, learns the relationship between those attributes and the posted freight rate, and evaluates predictions on held-out records.

## Feature engineering
The training script:
- Loads a labeled training file named `train-test.csv`.
- Removes identifier and raw latitude/longitude fields that are not used by the current model.
- Parses the shipment date and derives year, month, day, day-of-week, and weekend indicators.
- Builds a route feature by combining pickup and delivery locations.
- Calculates weight per mile while guarding against zero-distance division.
- Handles missing categorical values with the mode and missing numeric values with the median.
- Uses one-hot encoding for categorical route/equipment fields.
- Fits a Linear Regression model inside a Scikit-learn Pipeline.
- Reports MAE, RMSE, and R² on a held-out split.
- Refits the pipeline on all labeled training rows and prepares predictions for a separate validation dataset.

## Expected input files
The current script expects these files in the working directory:
- `train-test.csv` — labeled training data, including the target column `posted_rate`.
- `validation.csv` — records to score after training.

The repository currently contains the script but does not show those two CSV inputs in its root tree. Obtain the authorized data files before running; do not substitute unrelated data without checking the required column schema.

## Installation
Python 3.10+ is recommended.
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
python -m pip install --upgrade pip
pip install pandas numpy scikit-learn
```

## Run
The script filename is `spotter 0000` (no extension):
```bash
python "spotter 0000"
```
Run it from the directory containing both expected CSV files. The script prints regression metrics and generates predictions for the validation rows according to the code's output section.

## Evaluation
- **MAE:** average absolute difference between predicted and actual posted rates.
- **RMSE:** emphasizes larger prediction errors.
- **R²:** measures explained variance relative to a constant baseline.

A score depends on the dataset period, split, route distribution, and market conditions. Freight rates change over time; a random split may not represent future-market performance. For deployment, evaluate on a later time period and monitor drift.

## Skills demonstrated
Freight analytics · Regression · Date-based features · Route feature engineering · Missing-value handling · Categorical encoding · Scikit-learn Pipelines
