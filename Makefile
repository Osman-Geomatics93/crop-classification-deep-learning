.PHONY: setup run-all explore train apply train-pytorch apply-pytorch clean help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Create conda environment from environment.yml
	conda env create -f environment.yml
	@echo "Run: conda activate geodl"

setup-pip: ## Install dependencies with pip
	pip install -r requirements.txt

explore: ## Step 1: Run data exploration
	python 01_data_exploration.py

train: ## Step 2: Train sklearn/XGBoost models
	python 02_preprocessing_and_model.py

apply: ## Step 3: Apply ML model to satellite image
	python 03_apply_to_image.py

train-pytorch: ## Step 4: Train PyTorch deep learning models
	python 04_pytorch_models.py

apply-pytorch: ## Step 5: Apply PyTorch model to satellite image
	python 05_apply_pytorch_to_image.py

diagnose: ## Run band diagnostic tool
	python 03_diagnose_bands.py

run-all: explore train apply train-pytorch apply-pytorch ## Run the full pipeline (Steps 1-5)
	@echo "Pipeline complete!"

clean: ## Remove generated outputs and model files
	rm -rf outputs/ models/ __pycache__/
	rm -f *.png
	@echo "Cleaned outputs, models, and cache."
