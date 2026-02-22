# 🔍 Credit Card Fraud Detection

Credit card fraud detection via an industrialized ML pipeline, based on the notebook `Credit_risk_modelling_in_python.ipynb`.

## Project Structure

```
fraud-detection/
├── src/
│   ├── train.py          # Full training pipeline
│   └── predict.py        # Inference script (scoring new transactions)
├── data/                 # ← Place fraudTrain.csv and fraudTest.csv here
├── models/               # Artifacts generated after training (.pkl)
├── outputs/              # Generated predictions
├── Dockerfile            # Docker image for industrialization
├── Makefile              # Common commands
├── requirements.txt      # Python dependencies
└── README.md
```

## Expected Data

Place the following CSV files in `data/` (from Kaggle – [Credit Card Fraud Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)):

| File | Description |
|---|---|
| `fraudTrain.csv` | Training transactions (~1.3M rows) |
| `fraudTest.csv` | Test transactions (~555K rows) |

Main columns: `trans_date_trans_time`, `cc_num`, `merchant`, `category`, `amt`, `gender`, `lat`, `long`, `city_pop`, `dob`, `merch_lat`, `merch_long`, **`is_fraud`** (target).

## Engineered Features

The pipeline faithfully reproduces the notebook:

- **`age`**: customer age computed from `dob` and the transaction date
- **`distance`**: Euclidean distance (degrees) between the customer and the merchant
- **`hour`** / **`day_of_week`**: temporal features
- **`gender`**: binary encoding (F=0, M=1)
- **`category_*`**: one-hot encoding of the transaction category

## Quick Start – Local

### 1. Install dependencies

```bash
pip install -r requirements.txt
# or via Make:
make install
```

### 2. Train the model

```bash
make train
# or directly:
python src/train.py \
  --train data/fraudTrain.csv \
  --test  data/fraudTest.csv \
  --model-dir models \
  --threshold 0.01
```

Artifacts generated in `models/`:
- `logistic_model.pkl` – trained Logistic Regression model
- `scaler.pkl` – StandardScaler fitted on the training set
- `feature_cols.pkl` – ordered list of features

### 3. Score new transactions

```bash
make predict INPUT=data/fraudTest.csv
# or:
python src/predict.py \
  --input data/new_transactions.csv \
  --model-dir models \
  --output outputs/scored.csv \
  --threshold 0.01
```

## Docker Usage

### Build

```bash
make docker-build
```

### Training in Docker

```bash
make docker-train
```

(Automatically mounts `data/`, `models/`, and `outputs/` as local volumes.)

### Inference in Docker

```bash
make docker-predict INPUT=data/fraudTest.csv
```

## Configurable Parameters

| Parameter | Default | Description |
|---|---|---|
| `--train` | `data/fraudTrain.csv` | Path to training CSV |
| `--test` | `data/fraudTest.csv` | Path to test CSV |
| `--model-dir` | `models` | Directory for saving artifacts |
| `--threshold` | `0.01` | Classification threshold (sensitive to fraud) |
| `--C` | `1.0` | Logistic Regression regularization |
| `--max-iter` | `1000` | Maximum number of solver iterations |

> **Note on the threshold**: the dataset is highly imbalanced (~0.6% fraud). A low threshold (0.01) maximizes recall at the expense of precision, which is often preferable in fraud detection.

## Model

**Logistic Regression** with `class_weight='balanced'` to handle class imbalance. The model outputs fraud probabilities for each transaction.

Reported metrics:
- ROC-AUC
- Average Precision (PR-AUC)
- Full classification report (precision, recall, F1)
- Threshold sweep: 0.01, 0.05, 0.1, 0.5
