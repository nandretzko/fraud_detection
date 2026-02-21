"""
Credit Card Fraud Detection - Inference Script
==============================================
Load saved artifacts and score new transactions.
"""

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from train import (
    feature_engineering,
    COLS_TO_DROP,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_artifacts(model_dir: str):
    model_dir = Path(model_dir)
    with open(model_dir / "logistic_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(model_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(model_dir / "feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    return model, scaler, feature_cols


def predict(
    input_path: str,
    model_dir: str = "models",
    output_path: str = "outputs/scored.csv",
    threshold: float = 0.01,
):
    logger.info("Loading model artifacts from %s", model_dir)
    model, scaler, feature_cols = load_artifacts(model_dir)

    logger.info("Loading input data from %s", input_path)
    df = pd.read_csv(input_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df_fe = feature_engineering(df, is_train=False)

    # Align to training feature columns
    for col in feature_cols:
        if col not in df_fe.columns and col != "is_fraud":
            df_fe[col] = 0

    # Drop target if present
    X = df_fe[[c for c in feature_cols if c != "is_fraud"]]
    X_sc = scaler.transform(X)

    y_proba = model.predict_proba(X_sc)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    df["fraud_probability"] = y_proba
    df["fraud_predicted"] = y_pred

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Scored file saved to %s  (fraud predicted: %d / %d)", output_path, y_pred.sum(), len(y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fraud Detection - Inference")
    parser.add_argument("--input", required=True, help="CSV file to score")
    parser.add_argument("--model-dir", default="models", help="Directory with saved artifacts")
    parser.add_argument("--output", default="outputs/scored.csv", help="Output CSV path")
    parser.add_argument("--threshold", type=float, default=0.01, help="Classification threshold")
    args = parser.parse_args()
    predict(args.input, args.model_dir, args.output, args.threshold)
