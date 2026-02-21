# ──────────────────────────────────────────────────────────────────────────────
# Fraud Detection – Dockerfile
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Default directories for data / artifacts / outputs
RUN mkdir -p data models outputs

# Default command: run training
# Override at runtime, e.g.:
#   docker run ... python src/predict.py --input data/new_transactions.csv
CMD ["python", "src/train.py", \
     "--train", "data/fraudTrain.csv", \
     "--test",  "data/fraudTest.csv",  \
     "--model-dir", "models",          \
     "--threshold", "0.01"]
