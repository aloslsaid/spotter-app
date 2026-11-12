from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "train-test.csv"
VALIDATION_PATH = ROOT / "validation.csv"
OUTPUT_DIR = ROOT / "artifacts"
TARGET = "posted_rate"
CATEGORICAL = ["pickup", "delivery", "equipment", "route"]
NUMERIC = ["distance", "weight", "market_index", "quote_signal", "year",
           "month", "day", "day_of_week", "is_weekend", "weight_per_mile"]


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    if "date" not in df:
        raise ValueError("Input data must contain a 'date' column.")
    parsed = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = parsed.dt.year
    df["month"] = parsed.dt.month
    df["day"] = parsed.dt.day
    df["day_of_week"] = parsed.dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("Int64")
    for col in ("pickup", "delivery", "equipment", "distance", "weight"):
        if col not in df:
            raise ValueError(f"Input data is missing required column '{col}'.")
    df["route"] = df["pickup"].astype("string") + "_" + df["delivery"].astype("string")
    distance = pd.to_numeric(df["distance"], errors="coerce").replace(0, np.nan)
    weight = pd.to_numeric(df["weight"], errors="coerce")
    df["weight_per_mile"] = weight / distance
    df = df.drop(columns=["date", "load_id", "pickup_lat", "pickup_lon",
                          "delivery_lat", "delivery_lon"], errors="ignore")
    for col in ("market_index", "quote_signal"):
        if col not in df:
            df[col] = np.nan
    return df


def build_pipeline():
    preprocessor = ColumnTransformer([
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL),
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), NUMERIC),
    ])
    return Pipeline([("preprocess", preprocessor), ("model", LinearRegression())])


def main():
    if not TRAIN_PATH.is_file() or not VALIDATION_PATH.is_file():
        raise FileNotFoundError(
            "Place the authorized take-home files 'train-test.csv' and 'validation.csv' beside train.py."
        )
    raw = pd.read_csv(TRAIN_PATH)
    if TARGET not in raw:
        raise ValueError(f"Training file must contain '{TARGET}'.")
    raw[TARGET] = pd.to_numeric(raw[TARGET], errors="coerce")
    raw = raw.dropna(subset=[TARGET]).drop_duplicates()
    if len(raw) < 20:
        raise ValueError("At least 20 labeled training rows are required.")
    original_validation = pd.read_csv(VALIDATION_PATH)
    X = engineer_features(raw.drop(columns=[TARGET]))
    y = raw[TARGET].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    evaluation_model = build_pipeline()
    evaluation_model.fit(X_train, y_train)
    test_pred = evaluation_model.predict(X_test)
    metrics = {
        "test_rows": int(len(y_test)),
        "r2": float(r2_score(y_test, test_pred)),
        "mae": float(mean_absolute_error(y_test, test_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, test_pred))),
    }
    final_model = build_pipeline()
    final_model.fit(X, y)
    validation_features = engineer_features(original_validation)
    validation_pred = final_model.predict(validation_features)
    output = original_validation.copy()
    output[TARGET] = validation_pred
    OUTPUT_DIR.mkdir(exist_ok=True)
    joblib.dump(final_model, OUTPUT_DIR / "freight_rate_model.joblib")
    output.to_csv(OUTPUT_DIR / "validation_predictions.csv", index=False)
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved predictions to {OUTPUT_DIR / 'validation_predictions.csv'}")


if __name__ == "__main__":
    main()
