"""
Credit Card Fraud Detection - Inference Script.

Load saved model artifacts and score new transactions, producing a CSV
with per-row fraud probabilities and binary predictions.

Typical usage
-------------
    python predict.py \\
        --input  data/new_transactions.csv \\
        --model-dir models \\
        --output outputs/scored.csv \\
        --threshold 0.01
"""

# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import logging
import pickle
from pathlib import Path
from typing import Tuple

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ── Local ─────────────────────────────────────────────────────────────────────
from train import COLS_TO_DROP, feature_engineering

# ── Module-level logger ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_MODEL_FILENAME: str = "logistic_model.pkl"
_SCALER_FILENAME: str = "scaler.pkl"
_FEATURE_COLS_FILENAME: str = "feature_cols.pkl"
_UNNAMED_COL: str = "Unnamed: 0"
_TARGET_COL: str = "is_fraud"
_OUTPUT_PROBA_COL: str = "fraud_probability"
_OUTPUT_PRED_COL: str = "fraud_predicted"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_pickle(path: Path) -> object:
    """Deserialize a single pickle artifact from *path*.

    Parameters
    ----------
    path:
        Absolute or relative path to the ``.pkl`` file.

    Returns
    -------
    object
        The deserialized Python object.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist on disk.
    """
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    with open(path, "rb") as file_handle:
        return pickle.load(file_handle)  # nosec B301 – trusted internal artifacts


# ── Public API ────────────────────────────────────────────────────────────────

def load_artifacts(
    model_dir: str,
) -> Tuple[LogisticRegression, StandardScaler, list]:
    """Load the three model artifacts produced by ``train.py``.

    The function expects the following files inside *model_dir*:

    * ``logistic_model.pkl`` – a fitted :class:`sklearn.linear_model.LogisticRegression`
    * ``scaler.pkl``         – a fitted :class:`sklearn.preprocessing.StandardScaler`
    * ``feature_cols.pkl``   – an ordered :class:`list` of feature column names

    Parameters
    ----------
    model_dir:
        Path to the directory that contains the saved artifacts.

    Returns
    -------
    model:
        Trained logistic-regression classifier.
    scaler:
        Feature scaler fitted on the training set.
    feature_cols:
        Ordered list of feature names expected by the model.

    Raises
    ------
    FileNotFoundError
        If any required artifact is missing from *model_dir*.
    """
    artifact_dir = Path(model_dir)
    logger.debug("Loading artifacts from '%s'", artifact_dir.resolve())

    model = _load_pickle(artifact_dir / _MODEL_FILENAME)
    scaler = _load_pickle(artifact_dir / _SCALER_FILENAME)
    feature_cols = _load_pickle(artifact_dir / _FEATURE_COLS_FILENAME)

    logger.info(
        "Artifacts loaded — model: %s, features: %d",
        type(model).__name__,
        len(feature_cols),
    )
    return model, scaler, feature_cols


def _read_transactions(input_path: str) -> pd.DataFrame:
    """Read a transaction CSV and drop the spurious index column if present.

    Parameters
    ----------
    input_path:
        Path to the CSV file containing raw transaction data.

    Returns
    -------
    pd.DataFrame
        Raw transaction data with the ``Unnamed: 0`` column removed (if it
        existed).
    """
    dataframe = pd.read_csv(input_path)
    if _UNNAMED_COL in dataframe.columns:
        dataframe = dataframe.drop(columns=[_UNNAMED_COL])
    logger.info("Loaded %d transactions from '%s'", len(dataframe), input_path)
    return dataframe


def _align_features(
    dataframe: pd.DataFrame,
    feature_cols: list,
) -> pd.DataFrame:
    """Ensure *dataframe* has exactly the columns seen during training.

    Missing columns are added and filled with ``0``; the target column
    ``is_fraud`` is excluded from the returned feature matrix.

    Parameters
    ----------
    dataframe:
        Feature-engineered DataFrame (output of :func:`feature_engineering`).
    feature_cols:
        Ordered list of column names produced during training.

    Returns
    -------
    pd.DataFrame
        DataFrame aligned to *feature_cols*, without the target column.
    """
    for col in feature_cols:
        if col not in dataframe.columns and col != _TARGET_COL:
            logger.debug("Missing column '%s' — filling with 0", col)
            dataframe[col] = 0

    inference_cols = [c for c in feature_cols if c != _TARGET_COL]
    return dataframe[inference_cols]


def predict(
    input_path: str,
    model_dir: str = "models",
    output_path: str = "outputs/scored.csv",
    threshold: float = 0.01,
) -> pd.DataFrame:
    """Score a batch of transactions and write results to *output_path*.

    The function applies the full inference pipeline:

    1. Load model artifacts from *model_dir*.
    2. Read and validate the raw CSV at *input_path*.
    3. Run feature engineering (same transformations as training).
    4. Align the feature matrix to the training schema.
    5. Scale features and compute fraud probabilities.
    6. Apply *threshold* to derive binary predictions.
    7. Append ``fraud_probability`` and ``fraud_predicted`` columns and save.

    Parameters
    ----------
    input_path:
        Path to the CSV file containing transactions to score.
    model_dir:
        Directory that contains the trained model artifacts.
    output_path:
        Destination path for the scored CSV output.
    threshold:
        Probability cut-off for the positive (fraud) class.  A lower value
        increases recall at the cost of precision.  Defaults to ``0.01``
        because the dataset is heavily imbalanced (~0.6 % fraud).

    Returns
    -------
    pd.DataFrame
        The original DataFrame augmented with ``fraud_probability`` and
        ``fraud_predicted`` columns.

    Raises
    ------
    FileNotFoundError
        If *input_path* or any model artifact cannot be found.
    ValueError
        If the feature matrix is empty after alignment.
    """
    logger.info("Loading model artifacts from '%s'", model_dir)
    model, scaler, feature_cols = load_artifacts(model_dir)

    logger.info("Reading input transactions from '%s'", input_path)
    raw_df = _read_transactions(input_path)

    logger.info("Running feature engineering")
    engineered_df = feature_engineering(raw_df, is_train=False)

    feature_matrix = _align_features(engineered_df, feature_cols)

    if feature_matrix.empty:
        raise ValueError("Feature matrix is empty after alignment — check input data.")

    logger.info("Scaling %d features for %d rows", feature_matrix.shape[1], len(feature_matrix))
    scaled_features: np.ndarray = scaler.transform(feature_matrix)

    fraud_proba: np.ndarray = model.predict_proba(scaled_features)[:, 1]
    fraud_pred: np.ndarray = (fraud_proba >= threshold).astype(int)

    raw_df[_OUTPUT_PROBA_COL] = fraud_proba
    raw_df[_OUTPUT_PRED_COL] = fraud_pred

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(output_file, index=False)

    logger.info(
        "Scored file saved to '%s'  — fraud predicted: %d / %d (%.2f %%)",
        output_path,
        int(fraud_pred.sum()),
        len(fraud_pred),
        100.0 * fraud_pred.mean(),
    )
    return raw_df


# ── CLI entry-point ───────────────────────────────────────────────────────────

def _build_argument_parser() -> argparse.ArgumentParser:
    """Construct and return the CLI argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser with all inference arguments.
    """
    parser = argparse.ArgumentParser(
        description="Credit Card Fraud Detection — Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the CSV file containing transactions to score.",
    )
    parser.add_argument(
        "--model-dir",
        default="models",
        help="Directory containing the saved model artifacts.",
    )
    parser.add_argument(
        "--output",
        default="outputs/scored.csv",
        help="Destination path for the scored output CSV.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help=(
            "Probability threshold for the fraud class. "
            "Lower values increase recall (recommended for imbalanced data)."
        ),
    )
    return parser


def main() -> None:
    """Parse CLI arguments and run the inference pipeline."""
    args = _build_argument_parser().parse_args()
    predict(
        input_path=args.input,
        model_dir=args.model_dir,
        output_path=args.output,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
