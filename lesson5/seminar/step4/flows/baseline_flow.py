"""Prefect flow for baseline model training on full dataset."""

import logging
from prefect import flow, task
from typing import Dict, Any

from src.data_manager import ActiveLearningDataManager
from src.baseline_trainer import BaselineTrainer
from src.mlflow_tracker import MLflowTracker

logger = logging.getLogger(__name__)


@task
def setup_data_manager(random_seed: int = 42) -> ActiveLearningDataManager:
    """Initialize and load data manager."""
    data_manager = ActiveLearningDataManager(random_seed=random_seed)
    data_manager.load_and_split_data()
    return data_manager


@task
def train_baseline_model(
    data_manager: ActiveLearningDataManager, random_seed: int = 42
) -> Dict[str, Any]:
    """Train baseline model on full dataset."""
    baseline_trainer = BaselineTrainer(random_seed=random_seed)
    training_info = baseline_trainer.train_full_dataset(
        X_train=data_manager.X_train_full,
        y_train=data_manager.y_train_full,
        X_val=data_manager.X_val,
        y_val=data_manager.y_val,
        verbose=False,
    )
    val_metrics = baseline_trainer.evaluate_model(
        data_manager.X_val, data_manager.y_val, "validation"
    )
    test_metrics = baseline_trainer.evaluate_model(
        data_manager.X_test, data_manager.y_test, "test"
    )
    baseline_trainer.save_model("models/baseline_full_dataset.pkl")

    return {
        "training_info": training_info,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "model_info": baseline_trainer.get_model_info(),
    }


@task
def log_baseline_to_mlflow(
    baseline_results: Dict[str, Any], experiment_name: str = "step4_baseline"
) -> None:
    """Log baseline results to MLflow."""
    mlflow_tracker = MLflowTracker(experiment_name=experiment_name)
    with mlflow_tracker.start_run(run_name="baseline_full_dataset"):
        mlflow_tracker.log_params(
            {
                "method": "baseline",
                "training_samples": baseline_results["training_info"][
                    "training_samples"
                ],
                "data_usage": "100%",
                "model_type": "CatBoost",
            }
        )
        mlflow_tracker.log_metrics(
            {f"val_{k}": v for k, v in baseline_results["val_metrics"].items()}
        )
        mlflow_tracker.log_metrics(
            {f"test_{k}": v for k, v in baseline_results["test_metrics"].items()}
        )
        mlflow_tracker.log_params(baseline_results["training_info"])


@flow(name="Baseline Training Pipeline")
def baseline_pipeline(
    random_seed: int = 42, experiment_name: str = "step4_baseline"
) -> Dict[str, Any]:
    """Complete baseline training pipeline."""
    data_manager = setup_data_manager(random_seed=random_seed)
    baseline_results = train_baseline_model(data_manager, random_seed=random_seed)
    log_baseline_to_mlflow(
        baseline_results=baseline_results, experiment_name=experiment_name
    )

    summary = {
        "method": "baseline",
        "test_accuracy": baseline_results["test_metrics"]["accuracy"],
        "val_accuracy": baseline_results["val_metrics"]["accuracy"],
        "training_samples": baseline_results["training_info"]["training_samples"],
        "data_usage": "100%",
    }

    print("=== BASELINE RESULTS ===")
    print(f"Test Accuracy: {summary['test_accuracy']:.4f}")
    print(f"Val Accuracy: {summary['val_accuracy']:.4f}")
    print(f"Training Samples: {summary['training_samples']}")
    print(f"Data Usage: {summary['data_usage']}")

    return summary
