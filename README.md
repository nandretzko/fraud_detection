# Credit Card Fraud Detection

Credit card fraud detection using an industrialized machine learning pipeline, based on a Kaggle project I previously completed.

# Project Structure
fraud-detection/
├── src/
│   ├── train.py          # End-to-end training pipeline
│   └── predict.py        # Inference script (scoring new transactions)
├── data/                 # ← Place fraudTrain.csv and fraudTest.csv here
├── models/               # Artifacts generated after training (.pkl)
├── outputs/              # Generated predictions
├── Dockerfile            # Docker image for containerized deployment
├── Makefile              # Common commands
├── requirements.txt      # Python dependencies
└── README.md
Expected Data

Data should be placed in data/ with the following CSV files (from Kaggle – Credit Card Fraud Dataset
):

File	Description
fraudTrain.csv	Training transactions (~1.3M rows)
fraudTest.csv	Test transactions (~555K rows)

Main columns: trans_date_trans_time, cc_num, merchant, category, amt, gender, lat, long, city_pop, dob, merch_lat, merch_long, is_fraud (target).

# Engineered Features

The pipeline reproduces the original notebook logic:

age: customer age computed from dob and transaction date

distance: Euclidean distance (in degrees) between customer and merchant

hour / day_of_week: time-based features

gender: binary encoding (F=0, M=1)

category_*: one-hot encoding of transaction category

# Local Usage
1. Install Dependencies
pip install -r requirements.txt
# or via Make:
make install

# 2. Train the Model
make train
# or directly:
python src/train.py \
  --train data/fraudTrain.csv \
  --test  data/fraudTest.csv \
  --model-dir models \
  --threshold 0.01

Generated artifacts in models/:

logistic_model.pkl – trained Logistic Regression model

scaler.pkl – StandardScaler fitted on training data

feature_cols.pkl – ordered list of feature columns

3. Score New Transactions
make predict INPUT=data/fraudTest.csv
# or:
python src/predict.py \
  --input data/new_transactions.csv \
  --model-dir models \
  --output outputs/scored.csv \
  --threshold 0.01
Docker Usage
Build
make docker-build
Training in Docker
make docker-train

(data/, models/, and outputs/ are automatically mounted as local volumes.)

Inference in Docker
make docker-predict INPUT=data/fraudTest.csv
Configurable Parameters
Parameter	Default	Description
--train	data/fraudTrain.csv	Path to training CSV
--test	data/fraudTest.csv	Path to test CSV
--model-dir	models	Directory to store artifacts
--threshold	0.01	Classification threshold (fraud-sensitive)
--C	1.0	Logistic Regression regularization strength
--max-iter	1000	Maximum number of solver iterations

Threshold note: the dataset is highly imbalanced (~0.6% fraud). A low threshold (0.01) increases recall at the expense of precision, which is often preferable in fraud detection settings.

# Model

Logistic Regression with class_weight='balanced' to handle class imbalance.
The model outputs fraud probabilities for each transaction.

# Reported metrics:

ROC-AUC

Average Precision (PR-AUC)

Full classification report (precision, recall, F1)

Threshold sweep: 0.01, 0.05, 0.1, 0.5
