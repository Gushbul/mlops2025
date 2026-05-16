"""
Sanity tests for the ML pipeline.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from src.app.train import set_seed, prepare_data, train_model, evaluate_model


def test_accuracy_threshold() -> None:
    """Test that model achieves reasonable accuracy on Iris dataset."""
    # Set seed for reproducibility
    set_seed(42)

    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(test_size=0.2, random_state=42)

    # Train model
    model = train_model(X_train, y_train, C=1.0, max_iter=200, random_state=42)

    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)

    # Assert accuracy is reasonable for Iris dataset
    assert (
        metrics["accuracy"] >= 0.85
    ), f"Accuracy {metrics['accuracy']} is below threshold 0.85"


def test_determinism() -> None:
    """Test that training is deterministic with fixed seed."""

    def run_training_pipeline() -> float:
        """Run full training pipeline and return accuracy."""
        set_seed(42)
        X_train, X_test, y_train, y_test = prepare_data(test_size=0.2, random_state=42)
        model = train_model(X_train, y_train, C=1.0, max_iter=200, random_state=42)
        metrics = evaluate_model(model, X_test, y_test)
        return metrics["accuracy"]

    # Run pipeline twice
    accuracy1 = run_training_pipeline()
    accuracy2 = run_training_pipeline()

    # Should be exactly the same
    assert (
        abs(accuracy1 - accuracy2) <= 0.001
    ), f"Accuracy differs: {accuracy1} vs {accuracy2}"


def test_data_split_determinism() -> None:
    """Test that data splitting is deterministic."""
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split twice with same random state
    X_train1, X_test1, y_train1, y_test1 = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Should be identical
    np.testing.assert_array_equal(X_train1, X_train2)
    np.testing.assert_array_equal(X_test1, X_test2)
    np.testing.assert_array_equal(y_train1, y_train2)
    np.testing.assert_array_equal(y_test1, y_test2)


def test_model_training() -> None:
    """Test basic model training functionality."""
    set_seed(42)
    X_train, X_test, y_train, y_test = prepare_data(test_size=0.2, random_state=42)

    # Train model
    model = train_model(X_train, y_train, C=1.0, max_iter=200, random_state=42)

    # Check model is fitted
    assert hasattr(model, "coef_"), "Model should be fitted and have coefficients"
    assert hasattr(model, "intercept_"), "Model should be fitted and have intercept"

    # Check predictions work
    predictions = model.predict(X_test)
    assert len(predictions) == len(y_test), "Predictions length should match test set"
    assert all(
        pred in [0, 1, 2] for pred in predictions
    ), "All predictions should be valid Iris classes"
