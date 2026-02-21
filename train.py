"""
Credit Card Fraud Detection - Training Pipeline
================================================
Converts notebook Credit_risk_modelling_in_python.ipynb into a production script.
"""

import argparse
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------

def compute_age(df: pd.DataFrame) -> pd.Series:
    """Compute customer age from date of birth relative to transaction date."""
    dob = pd.to_datetime(df["dob"])
    trans_date = pd.to_datetime(df["trans_date_trans_time"])
    age = (trans_date - dob).dt.days / 365.25
    return age


def compute_distance(df: pd.DataFrame) -> pd.Series:
    """Euclidean distance (in degrees) between customer and merchant coordinates."""
    dist = np.sqrt(
        (df["lat"] - df["merch_lat"]) ** 2 + (df["long"] - df["merch_long"]) ** 2
    )
    return dist


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hour and day-of-week from transaction timestamp."""
    dt = pd.to_datetime(df["trans_date_trans_time"])
    df = df.copy()
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    return df


def one_hot_encode_category(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the 'category' column (matching notebook approach)."""
    return pd.get_dummies(df, columns=["category"], prefix="category", dtype=int)


def encode_gender(df: pd.DataFrame) -> pd.DataFrame:
    """Encode gender: F -> 0, M -> 1."""
    df = df.copy()
    df["gender"] = (df["gender"] == "M").astype(int)
    return df


def woe_bin(df: pd.DataFrame, feature: str, target: str, epsilon: float = 0.5) -> pd.DataFrame:
    """
    Compute WoE table for a categorical feature.
    Returns DataFrame sorted by WoE value.
    """
    woe_table = df.groupby(feature)[target].agg(["count", "sum"])
    woe_table.columns = ["total", "fraud"]
    woe_table["non_fraud"] = woe_table["total"] - woe_table["fraud"]
    woe_table["dist_fraud"] = woe_table["fraud"] / woe_table["fraud"].sum()
    woe_table["dist_non_fraud"] = woe_table["non_fraud"] / woe_table["non_fraud"].sum()
    woe_table["WoE"] = np.log(
        (woe_table["dist_non_fraud"] + epsilon) / (woe_table["dist_fraud"] + epsilon)
    )
    woe_table["IV_component"] = (
        (woe_table["dist_non_fraud"] - woe_table["dist_fraud"]) * woe_table["WoE"]
    )
    return woe_table.sort_values("WoE")


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

COLS_TO_DROP = [
    "trans_date_trans_time", "cc_num", "merchant", "first", "last",
    "street", "city", "state", "zip", "job", "dob", "trans_num",
    "unix_time", "lat", "long", "merch_lat", "merch_long",
]

FEATURE_COLS = None  # determined at fit time


def load_data(train_path: str, test_path: str):
    logger.info("Loading data from %s and %s", train_path, test_path)
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Drop unnamed index column if present
    for col in ["Unnamed: 0"]:
        if col in df_train.columns:
            df_train = df_train.drop(columns=[col])
        if col in df_test.columns:
            df_test = df_test.drop(columns=[col])

    logger.info("Train shape: %s | Test shape: %s", df_train.shape, df_test.shape)
    return df_train, df_test


def feature_engineering(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    df = df.copy()

    # Derived features
    df["age"] = compute_age(df)
    df["distance"] = compute_distance(df)
    df = extract_time_features(df)
    df = encode_gender(df)
    df = one_hot_encode_category(df)

    # Drop non-predictive / leakage columns
    cols_to_drop = [c for c in COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    return df


def align_columns(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """Ensure test has same dummy columns as train (fill missing with 0)."""
    missing = set(df_train.columns) - set(df_test.columns)
    for col in missing:
        df_test[col] = 0
    extra = set(df_test.columns) - set(df_train.columns)
    df_test = df_test.drop(columns=list(extra))
    df_test = df_test[df_train.columns]  # reorder
    return df_test


def train(
    train_path: str,
    test_path: str,
    model_dir: str = "models",
    threshold: float = 0.01,
    C: float = 1.0,
    max_iter: int = 1000,
):
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    # 1. Load
    df_train, df_test = load_data(train_path, test_path)

    # 2. Feature engineering
    logger.info("Engineering features...")
    df_train_fe = feature_engineering(df_train, is_train=True)
    df_test_fe = feature_engineering(df_test, is_train=False)
    df_test_fe = align_columns(df_train_fe, df_test_fe)

    # 3. Split X / y
    target = "is_fraud"
    X_train = df_train_fe.drop(columns=[target])
    y_train = df_train_fe[target]
    X_test = df_test_fe.drop(columns=[target])
    y_test = df_test_fe[target]

    logger.info(
        "Train fraud rate: %.4f | Test fraud rate: %.4f",
        y_train.mean(), y_test.mean()
    )

    # 4. Scale
    logger.info("Scaling features...")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # 5. Train logistic regression (class_weight='balanced' for imbalance)
    logger.info("Training Logistic Regression (C=%.4f, max_iter=%d)...", C, max_iter)
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )
    model.fit(X_train_sc, y_train)

    # 6. Predict probabilities
    y_proba_test = model.predict_proba(X_test_sc)[:, 1]

    # 7. Evaluate
    logger.info("Evaluating with threshold=%.3f", threshold)
    y_pred = (y_proba_test >= threshold).astype(int)

    roc_auc = roc_auc_score(y_test, y_proba_test)
    avg_prec = average_precision_score(y_test, y_proba_test)

    logger.info("ROC-AUC : %.4f", roc_auc)
    logger.info("Avg Precision (PR-AUC): %.4f", avg_prec)
    logger.info("\n%s", classification_report(y_test, y_pred, digits=4))

    # Threshold sweep
    logger.info("Threshold sweep:")
    for t in [0.01, 0.05, 0.1, 0.5]:
        y_p = (y_proba_test >= t).astype(int)
        logger.info("  threshold=%.2f\n%s", t, classification_report(y_test, y_p, digits=4, zero_division=0))

    # 8. Save artifacts
    model_path = Path(model_dir) / "logistic_model.pkl"
    scaler_path = Path(model_dir) / "scaler.pkl"
    feature_cols_path = Path(model_dir) / "feature_cols.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(feature_cols_path, "wb") as f:
        pickle.dump(list(X_train.columns), f)

    logger.info("Model saved to %s", model_path)
    logger.info("Scaler saved to %s", scaler_path)
    logger.info("Feature columns saved to %s", feature_cols_path)

    # 9. Save predictions
    output_path = Path(model_dir).parent / "outputs" / "predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y_actual": y_test.values, "y_proba": y_proba_test, "y_pred": y_pred}).to_csv(
        output_path, index=False
    )
    logger.info("Predictions saved to %s", output_path)

    return model, scaler, roc_auc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Fraud Detection - Training Pipeline")
    parser.add_argument("--train", default="data/fraudTrain.csv", help="Path to training CSV")
    parser.add_argument("--test", default="data/fraudTest.csv", help="Path to test CSV")
    parser.add_argument("--model-dir", default="models", help="Directory to save model artifacts")
    parser.add_argument("--threshold", type=float, default=0.01, help="Classification threshold")
    parser.add_argument("--C", type=float, default=1.0, help="Logistic Regression regularization")
    parser.add_argument("--max-iter", type=int, default=1000, help="Max iterations for solver")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        train_path=args.train,
        test_path=args.test,
        model_dir=args.model_dir,
        threshold=args.threshold,
        C=args.C,
        max_iter=args.max_iter,
    )
