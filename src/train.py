"""
Credit Card Fraud Detection - Training Pipeline.

Converts the notebook ``Credit_risk_modelling_in_python.ipynb`` into a
production-grade training script.  The pipeline covers data loading,
feature engineering, scaling, model fitting, evaluation, and artifact
serialization.

Typical usage
-------------
    python train.py \\
        --train      data/fraudTrain.csv \\
        --test       data/fraudTest.csv \\
        --model-dir  models \\
        --threshold  0.01 \\
        --C          1.0 \\
        --max-iter   1000
"""

# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

# ── Module-level logger ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_UNNAMED_COL: str = "Unnamed: 0"
_TARGET_COL: str = "is_fraud"
_THRESHOLD_SWEEP: List[float] = [0.01, 0.05, 0.1, 0.5]

_MODEL_FILENAME: str = "logistic_model.pkl"
_SCALER_FILENAME: str = "scaler.pkl"
_FEATURE_COLS_FILENAME: str = "feature_cols.pkl"
_PREDICTIONS_FILENAME: str = "predictions.csv"

#: Columns dropped before modelling — raw identifiers, coordinates and dates
#: that either leak the target or carry no predictive signal.
COLS_TO_DROP: List[str] = [
    "trans_date_trans_time",
    "cc_num",
    "merchant",
    "first",
    "last",
    "street",
    "city",
    "state",
    "zip",
    "job",
    "dob",
    "trans_num",
    "unix_time",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
]


# ── Feature engineering helpers ───────────────────────────────────────────────

def compute_age(dataframe: pd.DataFrame) -> pd.Series:
    """Compute customer age (in years) at the time of each transaction.

    Parameters
    ----------
    dataframe:
        Raw transaction DataFrame containing ``dob`` and
        ``trans_date_trans_time`` columns.

    Returns
    -------
    pd.Series
        Floating-point age values (years) aligned to *dataframe*'s index.
    """
    date_of_birth = pd.to_datetime(dataframe["dob"])
    transaction_date = pd.to_datetime(dataframe["trans_date_trans_time"])
    return (transaction_date - date_of_birth).dt.days / 365.25


def compute_distance(dataframe: pd.DataFrame) -> pd.Series:
    """Compute Euclidean distance (degrees) between customer and merchant.

    This is a simplified proxy for geographic distance and replicates the
    feature used in the original notebook.

    Parameters
    ----------
    dataframe:
        DataFrame with columns ``lat``, ``long``, ``merch_lat``,
        ``merch_long``.

    Returns
    -------
    pd.Series
        Non-negative distance values aligned to *dataframe*'s index.
    """
    return np.sqrt(
        (dataframe["lat"] - dataframe["merch_lat"]) ** 2
        + (dataframe["long"] - dataframe["merch_long"]) ** 2
    )


def extract_time_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Extract ``hour`` and ``day_of_week`` from the transaction timestamp.

    Parameters
    ----------
    dataframe:
        DataFrame containing a ``trans_date_trans_time`` column parseable
        by :func:`pandas.to_datetime`.

    Returns
    -------
    pd.DataFrame
        Copy of *dataframe* with two new columns:
        ``hour`` (0–23) and ``day_of_week`` (0 = Monday, 6 = Sunday).
    """
    dataframe = dataframe.copy()
    parsed_dt = pd.to_datetime(dataframe["trans_date_trans_time"])
    dataframe["hour"] = parsed_dt.dt.hour
    dataframe["day_of_week"] = parsed_dt.dt.dayofweek
    return dataframe


def one_hot_encode_category(dataframe: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the ``category`` column.

    Replicates the encoding approach used in the original notebook.
    Resulting columns are prefixed with ``category_`` and cast to
    ``int`` dtype.

    Parameters
    ----------
    dataframe:
        DataFrame containing a ``category`` column.

    Returns
    -------
    pd.DataFrame
        DataFrame with the original ``category`` column replaced by
        binary indicator columns.
    """
    return pd.get_dummies(dataframe, columns=["category"], prefix="category", dtype=int)


def encode_gender(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Binary-encode the ``gender`` column (``F`` → 0, ``M`` → 1).

    Parameters
    ----------
    dataframe:
        DataFrame containing a ``gender`` column with values ``'F'``
        and ``'M'``.

    Returns
    -------
    pd.DataFrame
        Copy of *dataframe* with the ``gender`` column replaced by an
        integer binary indicator.
    """
    dataframe = dataframe.copy()
    dataframe["gender"] = (dataframe["gender"] == "M").astype(int)
    return dataframe


def woe_bin(
    dataframe: pd.DataFrame,
    feature: str,
    target: str,
    epsilon: float = 0.5,
) -> pd.DataFrame:
    """Compute a Weight-of-Evidence (WoE) summary table for a categorical feature.

    WoE is defined as::

        WoE_i = log( (dist_non_fraud_i + ε) / (dist_fraud_i + ε) )

    The smoothing term *epsilon* prevents division-by-zero for rare categories.

    Parameters
    ----------
    dataframe:
        DataFrame containing *feature* and *target* columns.
    feature:
        Name of the categorical column to analyse.
    target:
        Name of the binary target column (1 = fraud, 0 = non-fraud).
    epsilon:
        Laplace-style smoothing constant added to both distributions before
        computing the log ratio.  Defaults to ``0.5``.

    Returns
    -------
    pd.DataFrame
        Summary table indexed by *feature* categories with columns:
        ``total``, ``fraud``, ``non_fraud``, ``dist_fraud``,
        ``dist_non_fraud``, ``WoE``, ``IV_component``,
        sorted in ascending WoE order.
    """
    woe_table = dataframe.groupby(feature)[target].agg(["count", "sum"])
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


# ── Core pipeline ─────────────────────────────────────────────────────────────

def load_data(
    train_path: str,
    test_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load training and test CSV files into DataFrames.

    The function also drops the spurious ``Unnamed: 0`` column that pandas
    sometimes writes when saving a DataFrame with its default integer index.

    Parameters
    ----------
    train_path:
        Path to the training CSV (e.g. ``data/fraudTrain.csv``).
    test_path:
        Path to the test CSV (e.g. ``data/fraudTest.csv``).

    Returns
    -------
    df_train:
        Raw training DataFrame.
    df_test:
        Raw test DataFrame.
    """
    logger.info("Loading data — train: '%s' | test: '%s'", train_path, test_path)
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    for dataframe in (df_train, df_test):
        if _UNNAMED_COL in dataframe.columns:
            dataframe.drop(columns=[_UNNAMED_COL], inplace=True)

    logger.info("Train shape: %s | Test shape: %s", df_train.shape, df_test.shape)
    return df_train, df_test


def feature_engineering(
    dataframe: pd.DataFrame,
    is_train: bool = True,  # noqa: ARG001 – reserved for future train-only steps
) -> pd.DataFrame:
    """Apply the full feature-engineering pipeline to a raw transaction DataFrame.

    Steps applied (in order):

    1. Compute ``age`` from date-of-birth and transaction timestamp.
    2. Compute ``distance`` between customer and merchant coordinates.
    3. Extract ``hour`` and ``day_of_week`` from the transaction timestamp.
    4. Binary-encode ``gender``.
    5. One-hot encode ``category``.
    6. Drop non-predictive / leakage columns defined in :data:`COLS_TO_DROP`.

    Parameters
    ----------
    dataframe:
        Raw transaction DataFrame (training or inference).
    is_train:
        Reserved flag for future train-only transformations (e.g. fitting
        encoders).  Currently unused but kept for API compatibility with
        ``predict.py``.

    Returns
    -------
    pd.DataFrame
        Transformed DataFrame ready for scaling and model ingestion.
    """
    dataframe = dataframe.copy()

    dataframe["age"] = compute_age(dataframe)
    dataframe["distance"] = compute_distance(dataframe)
    dataframe = extract_time_features(dataframe)
    dataframe = encode_gender(dataframe)
    dataframe = one_hot_encode_category(dataframe)

    cols_to_drop = [c for c in COLS_TO_DROP if c in dataframe.columns]
    dataframe = dataframe.drop(columns=cols_to_drop)

    return dataframe


def align_columns(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> pd.DataFrame:
    """Align *df_test* columns to match those of *df_train*.

    After one-hot encoding, unseen categories in the test set may create
    extra columns and rare categories may be absent.  This function:

    * Adds columns present in *df_train* but missing from *df_test* (filled
      with ``0``).
    * Drops columns present in *df_test* but absent from *df_train*.
    * Reorders *df_test* columns to match *df_train*.

    Parameters
    ----------
    df_train:
        Feature-engineered training DataFrame (reference schema).
    df_test:
        Feature-engineered test DataFrame to align.

    Returns
    -------
    pd.DataFrame
        *df_test* with columns aligned to *df_train*.
    """
    missing_cols = set(df_train.columns) - set(df_test.columns)
    for col in missing_cols:
        logger.debug("Test set missing column '%s' — filling with 0", col)
        df_test[col] = 0

    extra_cols = set(df_test.columns) - set(df_train.columns)
    if extra_cols:
        logger.debug("Dropping %d extra test columns: %s", len(extra_cols), extra_cols)
        df_test = df_test.drop(columns=list(extra_cols))

    return df_test[df_train.columns]


def _evaluate(
    y_true: pd.Series,
    y_proba: np.ndarray,
    threshold: float,
) -> Tuple[float, float]:
    """Log evaluation metrics and return key scores.

    Computes ROC-AUC, Average Precision (PR-AUC), and a full
    classification report at *threshold*.  Also runs a threshold sweep
    over :data:`_THRESHOLD_SWEEP`.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.
    y_proba:
        Predicted fraud probabilities (positive class).
    threshold:
        Primary classification cut-off used for the main report.

    Returns
    -------
    roc_auc:
        Area under the ROC curve.
    avg_precision:
        Average precision (area under the precision-recall curve).
    """
    roc_auc = roc_auc_score(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)

    logger.info("ROC-AUC            : %.4f", roc_auc)
    logger.info("Avg Precision (PR) : %.4f", avg_precision)

    y_pred = (y_proba >= threshold).astype(int)
    logger.info(
        "Classification report (threshold=%.3f):\n%s",
        threshold,
        classification_report(y_true, y_pred, digits=4, zero_division=0),
    )

    logger.info("Threshold sweep:")
    for sweep_t in _THRESHOLD_SWEEP:
        y_sweep = (y_proba >= sweep_t).astype(int)
        logger.info(
            "  threshold=%.2f\n%s",
            sweep_t,
            classification_report(y_true, y_sweep, digits=4, zero_division=0),
        )

    return roc_auc, avg_precision


def _save_artifacts(
    model: LogisticRegression,
    scaler: StandardScaler,
    feature_cols: List[str],
    model_dir: Path,
) -> None:
    """Serialize trained model artifacts to disk using pickle.

    Saves three files into *model_dir*:

    * ``logistic_model.pkl``
    * ``scaler.pkl``
    * ``feature_cols.pkl``

    Parameters
    ----------
    model:
        Fitted :class:`~sklearn.linear_model.LogisticRegression` instance.
    scaler:
        Fitted :class:`~sklearn.preprocessing.StandardScaler` instance.
    feature_cols:
        Ordered list of feature column names used during training.
    model_dir:
        Target directory (must already exist).
    """
    artifacts = {
        _MODEL_FILENAME: model,
        _SCALER_FILENAME: scaler,
        _FEATURE_COLS_FILENAME: feature_cols,
    }
    for filename, obj in artifacts.items():
        path = model_dir / filename
        with open(path, "wb") as file_handle:
            pickle.dump(obj, file_handle)  # nosec B301 – trusted internal artifacts
        logger.info("Artifact saved → '%s'", path)


def _save_predictions(
    y_true: pd.Series,
    y_proba: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
) -> None:
    """Persist test-set predictions to a CSV file.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.
    y_proba:
        Predicted fraud probabilities.
    y_pred:
        Binary predictions derived from the chosen threshold.
    output_dir:
        Directory where ``predictions.csv`` will be written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / _PREDICTIONS_FILENAME
    pd.DataFrame(
        {"y_actual": y_true.values, "y_proba": y_proba, "y_pred": y_pred}
    ).to_csv(output_path, index=False)
    logger.info("Predictions saved → '%s'", output_path)


def train(
    train_path: str,
    test_path: str,
    model_dir: str = "models",
    threshold: float = 0.01,
    regularization_c: float = 1.0,
    max_iter: int = 1000,
) -> Tuple[LogisticRegression, StandardScaler, float]:
    """Run the end-to-end training pipeline.

    Steps
    -----
    1. Load raw CSV data.
    2. Apply feature engineering to train and test sets.
    3. Align test columns to the training schema.
    4. Split feature matrix and target vector.
    5. Fit a :class:`~sklearn.preprocessing.StandardScaler`.
    6. Train a balanced :class:`~sklearn.linear_model.LogisticRegression`.
    7. Evaluate on the held-out test set.
    8. Serialize model, scaler, and feature column list.
    9. Save test predictions to ``outputs/predictions.csv``.

    Parameters
    ----------
    train_path:
        Path to the training CSV file.
    test_path:
        Path to the test CSV file.
    model_dir:
        Directory where model artifacts will be saved.
    threshold:
        Classification probability threshold for the fraud class.  A low
        value (default ``0.01``) maximises recall on the heavily imbalanced
        dataset (~0.6 % fraud rate).
    regularization_c:
        Inverse regularization strength passed to
        :class:`~sklearn.linear_model.LogisticRegression` as ``C``.
        Smaller values → stronger regularization.
    max_iter:
        Maximum number of solver iterations.

    Returns
    -------
    model:
        Fitted logistic-regression classifier.
    scaler:
        Fitted standard scaler.
    roc_auc:
        ROC-AUC score on the test set.
    """
    artifact_dir = Path(model_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load ──────────────────────────────────────────────────────────────────
    df_train, df_test = load_data(train_path, test_path)

    # 2. Feature engineering ───────────────────────────────────────────────────
    logger.info("Engineering features...")
    df_train_fe = feature_engineering(df_train, is_train=True)
    df_test_fe = feature_engineering(df_test, is_train=False)
    df_test_fe = align_columns(df_train_fe, df_test_fe)

    # 3. Split X / y ───────────────────────────────────────────────────────────
    x_train = df_train_fe.drop(columns=[_TARGET_COL])
    y_train = df_train_fe[_TARGET_COL]
    x_test = df_test_fe.drop(columns=[_TARGET_COL])
    y_test = df_test_fe[_TARGET_COL]

    logger.info(
        "Train fraud rate: %.4f | Test fraud rate: %.4f",
        y_train.mean(),
        y_test.mean(),
    )

    # 4. Scale ─────────────────────────────────────────────────────────────────
    logger.info("Fitting StandardScaler...")
    scaler = StandardScaler()
    x_train_scaled: np.ndarray = scaler.fit_transform(x_train)
    x_test_scaled: np.ndarray = scaler.transform(x_test)

    # 5. Train ─────────────────────────────────────────────────────────────────
    logger.info(
        "Training LogisticRegression (C=%.4f, max_iter=%d)...",
        regularization_c,
        max_iter,
    )
    model = LogisticRegression(
        C=regularization_c,
        max_iter=max_iter,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )
    model.fit(x_train_scaled, y_train)

    # 6. Evaluate ──────────────────────────────────────────────────────────────
    y_proba_test: np.ndarray = model.predict_proba(x_test_scaled)[:, 1]
    roc_auc, _ = _evaluate(y_test, y_proba_test, threshold)

    # 7. Persist artifacts ─────────────────────────────────────────────────────
    _save_artifacts(model, scaler, list(x_train.columns), artifact_dir)

    # 8. Persist predictions ───────────────────────────────────────────────────
    y_pred: np.ndarray = (y_proba_test >= threshold).astype(int)
    _save_predictions(y_test, y_proba_test, y_pred, artifact_dir.parent / "outputs")

    return model, scaler, roc_auc


# ── CLI entry-point ───────────────────────────────────────────────────────────

def _build_argument_parser() -> argparse.ArgumentParser:
    """Construct and return the CLI argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser for all training arguments.
    """
    parser = argparse.ArgumentParser(
        description="Credit Card Fraud Detection — Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train",
        default="data/fraudTrain.csv",
        help="Path to the training CSV file.",
    )
    parser.add_argument(
        "--test",
        default="data/fraudTest.csv",
        help="Path to the test CSV file.",
    )
    parser.add_argument(
        "--model-dir",
        default="models",
        help="Directory where trained artifacts will be saved.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help=(
            "Classification threshold for the fraud class. "
            "Lower values increase recall (recommended for imbalanced data)."
        ),
    )
    parser.add_argument(
        "--C",
        dest="regularization_c",
        type=float,
        default=1.0,
        help="Inverse regularization strength for Logistic Regression.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum number of solver iterations.",
    )
    return parser


def main() -> None:
    """Parse CLI arguments and launch the training pipeline."""
    args = _build_argument_parser().parse_args()
    train(
        train_path=args.train,
        test_path=args.test,
        model_dir=args.model_dir,
        threshold=args.threshold,
        regularization_c=args.regularization_c,
        max_iter=args.max_iter,
    )


if __name__ == "__main__":
    main()
