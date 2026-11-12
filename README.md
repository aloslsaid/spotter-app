# Spotter — Freight Rate Prediction

An experimental regression project for estimating freight rates from transportation-load data. The README describes model comparison and a December prediction output, but the repository currently lists a folder named `spotter 0000` and this README. Inspect that folder to confirm the implementation, data files, and exact workflow.

## Getting started

1. Clone or download the repository and inspect `spotter 0000`.
2. Confirm the training and prediction data, target column, load identifier, and any data license.
3. Create a Python virtual environment. There is no root `requirements.txt` in the current file list, so derive dependencies from the project source rather than guessing.
4. Use the script or notebook's documented command once the entry point is confirmed.

The existing project notes report a validation R² of 0.8618, MAE of 126.39, and RMSE of 543.59. These figures are included as historical project claims, not independently verified results; the metric units, validation split, data version, and procedure should be documented before treating them as comparable. No root test suite or reproducible command is currently listed.

## Reproducibility

Keep train/validation separation clear, prevent target leakage, preserve load IDs when exporting predictions, and record the exact input/output schema. Do not publish private customer or shipment data.
