# ──────────────────────────────────────────────────────────────────────────────
# Fraud Detection – Makefile
# ──────────────────────────────────────────────────────────────────────────────

IMAGE   := fraud-detection
TAG     := latest
DATA_DIR := $(shell pwd)/data

.PHONY: help install train predict docker-build docker-train docker-predict clean

help:              ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS=":.*##"}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Local (virtualenv) ────────────────────────────────────────────────────────

install:           ## Install Python dependencies
	pip install -r requirements.txt

train:             ## Train the model locally (needs fraudTrain.csv & fraudTest.csv in data/)
	python src/train.py \
	  --train data/fraudTrain.csv \
	  --test  data/fraudTest.csv \
	  --model-dir models \
	  --threshold 0.01

predict:           ## Score a file locally (set INPUT=data/myfile.csv)
	python src/predict.py \
	  --input $(INPUT) \
	  --model-dir models \
	  --output outputs/scored.csv \
	  --threshold 0.01

# ── Docker ────────────────────────────────────────────────────────────────────

docker-build:      ## Build the Docker image
	docker build -t $(IMAGE):$(TAG) .

docker-train:      ## Run training inside Docker (mounts local data/ and models/)
	docker run --rm \
	  -v $(DATA_DIR):/app/data \
	  -v $(shell pwd)/models:/app/models \
	  -v $(shell pwd)/outputs:/app/outputs \
	  $(IMAGE):$(TAG)

docker-predict:    ## Score a file inside Docker (set INPUT=path/to/file.csv)
	docker run --rm \
	  -v $(DATA_DIR):/app/data \
	  -v $(shell pwd)/models:/app/models \
	  -v $(shell pwd)/outputs:/app/outputs \
	  $(IMAGE):$(TAG) \
	  python src/predict.py \
	    --input /app/$(INPUT) \
	    --model-dir /app/models \
	    --output /app/outputs/scored.csv \
	    --threshold 0.01

clean:             ## Remove generated artifacts
	rm -rf models/*.pkl outputs/*.csv
