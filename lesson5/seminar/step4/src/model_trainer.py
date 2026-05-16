"""Model training and evaluation for Continuous Training."""

import json
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles model training and evaluation."""

    def __init__(self, model_params: Dict[str, Any] = None):
        """Initialize ModelTrainer.

        Args:
            model_params: Parameters for CatBoost model
        """
        self.model_params = model_params or {
            "iterations": 100,
            "depth": 6,
            "learning_rate": 0.1,
            "loss_function": "MultiClass",
            "random_seed": 42,
            "verbose": False,
        }
        self.model = None

    def train_model(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> CatBoostClassifier:
        """Train CatBoost model.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Trained CatBoost model
        """
        logger.info(f"Training CatBoost model on {len(X_train)} samples...")

        self.model = CatBoostClassifier(**self.model_params)
        self.model.fit(X_train, y_train)

        logger.info("Model training completed")
        return self.model

    def evaluate_model(
        self, X: np.ndarray, y: np.ndarray, dataset_name: str = "dataset"
    ) -> Dict[str, Any]:
        """Evaluate model on given dataset.

        Args:
            X: Features
            y: True labels
            dataset_name: Name of the dataset for logging

        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        logger.info(f"Evaluating model on {dataset_name} ({len(X)} samples)...")

        y_pred = self.model.predict(X)

        accuracy = accuracy_score(y, y_pred)
        f1_macro = f1_score(y, y_pred, average="macro")
        f1_weighted = f1_score(y, y_pred, average="weighted")

        class_report = classification_report(y, y_pred)

        metrics = {
            "dataset": dataset_name,
            "samples": len(X),
            "accuracy": float(accuracy),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "classification_report": class_report,
        }

        logger.info(f"{dataset_name} metrics:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  F1 Score (macro): {f1_macro:.4f}")
        logger.info(f"  F1 Score (weighted): {f1_weighted:.4f}")

        return metrics

    def save_model(self, filepath: str) -> None:
        """Save trained model.

        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        self.model.save_model(filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> CatBoostClassifier:
        """Load trained model.

        Args:
            filepath: Path to the saved model

        Returns:
            Loaded CatBoost model
        """
        self.model = CatBoostClassifier()
        self.model.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
        return self.model

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model.

        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"status": "not_trained"}

        return {
            "status": "trained",
            "params": self.model_params,
            "feature_count": (
                self.model.get_feature_count()
                if hasattr(self.model, "get_feature_count")
                else None
            ),
        }

    def save_metrics(self, metrics: Dict[str, Any], filepath: str) -> None:
        """Save metrics to JSON file.

        Args:
            metrics: Metrics dictionary
            filepath: Path to save metrics
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        metrics_to_save = {
            k: v for k, v in metrics.items() if k != "classification_report"
        }

        with open(filepath, "w") as f:
            json.dump(metrics_to_save, f, indent=2)

        if "classification_report" in metrics:
            report_filepath = filepath.replace(".json", "_report.txt")
            with open(report_filepath, "w") as f:
                f.write(
                    f"Classification Report - {metrics.get('dataset', 'Unknown')}\n"
                )
                f.write("=" * 50 + "\n")
                f.write(metrics["classification_report"])

        logger.info(f"Metrics saved to {filepath}")
