"""
Training script with MLflow tracking for Iris classification.
"""

import random
from typing import Dict, Any, Tuple
import yaml
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def load_config(config_path: str = "configs/train.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def prepare_data(
    test_size: float, random_state: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and split Iris dataset."""
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


def train_model(
    X_train: np.ndarray, y_train: np.ndarray, **model_params
) -> LogisticRegression:
    """Train logistic regression model."""
    model = LogisticRegression(**model_params)
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: LogisticRegression, X_test: np.ndarray, y_test: np.ndarray
) -> Dict[str, float]:
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
    }

    return metrics


def main() -> None:
    """Main training pipeline."""
    # Load configuration
    config = load_config()

    # Set seed for reproducibility
    set_seed(config["seed"])

    # Setup MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(config["mlflow_experiment"])

    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(
        test_size=config["test_size"], random_state=config["seed"]
    )

    # Train model
    model_params = {**config["model"], "random_state": config["seed"]}
    model = train_model(X_train, y_train, **model_params)

    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)

    # Log to MLflow
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("seed", config["seed"])
        mlflow.log_param("test_size", config["test_size"])
        for param, value in config["model"].items():
            mlflow.log_param(param, value)

        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Log config artifact
        mlflow.log_artifact("configs/train.yaml")

    # Print results for verification
    print("Training completed!")
    print(f"Accuracy: {metrics['accuracy']:.6f}")
    print(f"F1 Score (macro): {metrics['f1_macro']:.6f}")


if __name__ == "__main__":
    main()
