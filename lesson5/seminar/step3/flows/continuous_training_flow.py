"""
Continuous Training flow using Prefect and MLflow.
"""

import sys
import os
import logging
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from prefect import flow, task
from data_manager import DataManager
from model_trainer import ModelTrainer
from mlflow_tracker import MLflowTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@task(name="Initialize Data Manager", log_prints=True)
def initialize_data_manager() -> DataManager:
    """Initialize and load data using DataManager."""
    print("Initializing DataManager...")
    data_manager = DataManager(random_seed=42)
    dataset_info = data_manager.load_and_split_data()

    print(f"Dataset loaded: {dataset_info}")
    return data_manager


@task(name="Initialize MLflow Tracker", log_prints=True)
def initialize_mlflow_tracker() -> MLflowTracker:
    """Initialize MLflow tracker."""
    print("Initializing MLflow tracker...")
    tracker = MLflowTracker(experiment_name="continuous_training_step2")
    experiment_info = tracker.get_experiment_info()
    print(f"MLflow experiment: {experiment_info}")
    return tracker


@task(name="Train Model Iteration", log_prints=True)
def train_model_iteration(
    data_manager: DataManager,
    tracker: MLflowTracker,
    iteration: int,
    train_percentage: float,
) -> dict:
    """Train model for a specific iteration with given percentage of data."""
    print(f"Training iteration {iteration} with {train_percentage*100:.0f}% of data...")

    X_train, y_train = data_manager.get_incremental_train_data(train_percentage)
    X_val, y_val = data_manager.get_validation_data()
    X_test, y_test = data_manager.get_test_data()

    trainer = ModelTrainer()

    model = trainer.train_model(X_train, y_train)

    val_metrics = trainer.evaluate_model(X_val, y_val, "validation")
    test_metrics = trainer.evaluate_model(X_test, y_test, "test")

    model_path = f"models/catboost_model_iter_{iteration:02d}.cbm"
    trainer.save_model(model_path)

    metrics_dir = Path("metrics") / f"iteration_{iteration:02d}"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_metrics(val_metrics, str(metrics_dir / "val_metrics.json"))
    trainer.save_metrics(test_metrics, str(metrics_dir / "test_metrics.json"))

    run_id = tracker.log_training_iteration(
        iteration=iteration,
        train_size=len(X_train),
        train_percentage=train_percentage * 100,
        model_params=trainer.model_params,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        model=model,
    )

    result = {
        "iteration": iteration,
        "train_percentage": train_percentage,
        "train_size": len(X_train),
        "val_accuracy": val_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "val_f1_macro": val_metrics["f1_macro"],
        "test_f1_macro": test_metrics["f1_macro"],
        "model_path": model_path,
        "mlflow_run_id": run_id,
    }

    print(f"Iteration {iteration} completed:")
    print(f"  Train size: {len(X_train)}")
    print(f"  Val accuracy: {val_metrics['accuracy']:.4f}")
    print(f"  Test accuracy: {test_metrics['accuracy']:.4f}")

    return result


@flow(name="Continuous Training Pipeline", log_prints=True)
def continuous_training_pipeline(start_percentage: float = 0.1, iterations: int = 10):
    """
    Main Continuous Training pipeline.

    Args:
        start_percentage: Starting percentage of training data (default: 0.1 = 10%)
        iterations: Number of training iterations (default: 10)
    """
    print("Starting Continuous Training Pipeline")
    print(f"Start percentage: {start_percentage*100:.0f}%")
    print(f"Iterations: {iterations}")
    print("=" * 60)

    data_manager = initialize_data_manager()
    tracker = initialize_mlflow_tracker()

    Path("models").mkdir(exist_ok=True)
    Path("metrics").mkdir(exist_ok=True)

    results = []
    for i in range(1, iterations + 1):
        train_percentage = start_percentage * i

        if train_percentage > 1.0:
            train_percentage = 1.0

        result = train_model_iteration(
            data_manager=data_manager,
            tracker=tracker,
            iteration=i,
            train_percentage=train_percentage,
        )

        results.append(result)

        if train_percentage >= 1.0:
            print(f"Reached 100% of training data at iteration {i}")
            break

    print("\\n" + "=" * 60)
    print("CONTINUOUS TRAINING SUMMARY")
    print("=" * 60)

    for result in results:
        print(
            f"Iteration {result['iteration']:2d}: "
            f"{result['train_percentage']:5.1f}% data, "
            f"Val Acc: {result['val_accuracy']:.4f}, "
            f"Test Acc: {result['test_accuracy']:.4f}"
        )

    best_result = max(results, key=lambda x: x["val_accuracy"])
    print(
        f"\\nBest iteration: {best_result['iteration']} "
        f"(Val Acc: {best_result['val_accuracy']:.4f})"
    )

    print("\\nContinuous Training Pipeline completed!")
    print("Check MLflow UI for detailed experiment tracking")

    return {
        "total_iterations": len(results),
        "results": results,
        "best_iteration": best_result,
    }


@flow(name="Single CT Iteration", log_prints=True)
def single_iteration_pipeline(iteration: int = 1, train_percentage: float = 0.1):
    """
    Run a single Continuous Training iteration.

    Args:
        iteration: Iteration number
        train_percentage: Percentage of training data to use
    """
    print(
        f"Running single CT iteration {iteration} with {train_percentage*100:.0f}% data"
    )

    data_manager = initialize_data_manager()
    tracker = initialize_mlflow_tracker()

    Path("models").mkdir(exist_ok=True)
    Path("metrics").mkdir(exist_ok=True)

    result = train_model_iteration(
        data_manager=data_manager,
        tracker=tracker,
        iteration=iteration,
        train_percentage=train_percentage,
    )

    print(f"Single iteration {iteration} completed!")
    return result


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "single":
            iteration = int(sys.argv[2]) if len(sys.argv) > 2 else 1
            percentage = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
            single_iteration_pipeline(iteration, percentage)
        else:
            iterations = int(sys.argv[1])
            continuous_training_pipeline(iterations=iterations)
    else:
        continuous_training_pipeline()
