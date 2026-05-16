.PHONY: install train clean help

# Default target
help:
	@echo "Available commands:"
	@echo "  install  - Install dependencies using Poetry"
	@echo "  train    - Train the CatBoost model"
	@echo "  clean    - Clean training results (models and metrics)"
	@echo "  help     - Show this help message"

# Install dependencies
install:
	@echo "Installing dependencies with Poetry..."
	poetry install
	@echo "Dependencies installed successfully!"

# Train the model
train:
	@echo "Starting model training..."
	poetry run python train.py
	@echo "Training completed!"

# Clean training results
clean:
	@echo "Cleaning training results..."
	rm -rf models/
	rm -rf metrics/
	rm -rf catboost_info/
	@echo "Cleanup completed!"

# Clean and train (useful for fresh runs)
retrain: clean train
