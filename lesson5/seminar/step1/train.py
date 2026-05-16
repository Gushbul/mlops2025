#!/usr/bin/env python3
"""
Step 1: Basic model training for Continuous Training seminar.

This script:
1. Loads the Forest Cover Type dataset from sklearn
2. Splits data into train/test with seed=42
3. Uses only first 10% of train data for training
4. Trains a CatBoost model
5. Saves metrics and model
"""

import json
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from sklearn.datasets import fetch_covtype
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split


def create_directories():
    """Create necessary directories for outputs."""
    Path("models").mkdir(exist_ok=True)
    Path("metrics").mkdir(exist_ok=True)


def load_and_prepare_data():
    """Load Forest Cover Type dataset and prepare train/test splits."""
    print("Loading Forest Cover Type dataset...")

    covtype = fetch_covtype()
    X, y = covtype.data, covtype.target

    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Classes: {np.unique(y)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    n_samples = int(0.1 * len(X_train))
    X_train_small = X_train[:n_samples]
    y_train_small = y_train[:n_samples]

    print(f"Original train size: {len(X_train)}")
    print(f"Reduced train size (10%): {len(X_train_small)}")
    print(f"Test size: {len(X_test)}")

    return X_train_small, X_test, y_train_small, y_test


def train_model(X_train, y_train):
    """Train CatBoost model."""
    print("Training CatBoost model...")

    model = CatBoostClassifier(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        loss_function="MultiClass",
        random_seed=42,
        verbose=False,
    )

    model.fit(X_train, y_train)

    model.save_model("models/catboost_model.cbm")
    print("Model saved to models/catboost_model.cbm")

    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model and save metrics."""
    print("Evaluating model...")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    class_report = classification_report(y_test, y_pred)

    metrics = {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "test_samples": len(y_test),
    }

    with open("metrics/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open("metrics/classification_report.txt", "w") as f:
        f.write("Classification Report\\n")
        f.write("=" * 50 + "\\n")
        f.write(class_report)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (macro): {f1_macro:.4f}")
    print(f"F1 Score (weighted): {f1_weighted:.4f}")
    print("Metrics saved to metrics/")

    return metrics


def main():
    """Main training pipeline."""
    print("Starting Step 1: Basic Model Training")
    print("=" * 50)

    create_directories()

    X_train, X_test, y_train, y_test = load_and_prepare_data()

    model = train_model(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)

    print("\\nTraining completed successfully!")
    print(f"Final accuracy: {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
