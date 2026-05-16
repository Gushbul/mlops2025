"""Prefect flow for Active Learning with incremental data addition."""

import logging
from prefect import flow, task
from typing import Dict, Any

from src.data_manager import ActiveLearningDataManager
from src.model_trainer import ModelTrainer
from src.active_learning import ActiveLearningManager
from src.mlflow_tracker import MLflowTracker

logger = logging.getLogger(__name__)


@task
def setup_data_manager(random_seed: int = 42) -> ActiveLearningDataManager:
    """Initialize and load data manager."""
    data_manager = ActiveLearningDataManager(random_seed=random_seed)
    data_manager.load_and_split_data()
    return data_manager


@task
def run_active_learning_experiment(
    data_manager: ActiveLearningDataManager,
    sampling_strategy: str = "entropy",
    initial_percentage: float = 0.05,
    increment_percentage: float = 0.05,
    max_iterations: int = 15,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Run Active Learning experiment with incremental data addition."""

    al_manager = ActiveLearningManager(uncertainty_strategy=sampling_strategy)
    data_manager.initialize_active_learning(initial_percentage=initial_percentage)
    iteration_results = []

    for iteration in range(max_iterations):
        X_labeled, y_labeled = data_manager.get_labeled_data()

        if len(X_labeled) >= len(data_manager.X_train_full):
            print(f"All training data used at iteration {iteration + 1}")
            break

        model_trainer = ModelTrainer(model_params={"random_seed": random_seed})
        model_trainer.train_model(X_labeled, y_labeled)

        training_info = {
            "training_samples": len(X_labeled),
            "features": X_labeled.shape[1],
        }

        val_metrics = model_trainer.evaluate_model(
            data_manager.X_val, data_manager.y_val, "validation"
        )
        test_metrics = model_trainer.evaluate_model(
            data_manager.X_test, data_manager.y_test, "test"
        )

        current_percentage = len(X_labeled) / len(data_manager.X_train_full) * 100

        iteration_results.append(
            {
                "iteration": iteration + 1,
                "labeled_samples": len(X_labeled),
                "pool_samples": (
                    len(data_manager.X_pool) if not data_manager.is_pool_empty() else 0
                ),
                "data_percentage": current_percentage,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "training_info": training_info,
            }
        )

        print(f"Iteration {iteration + 1}:")
        print(f"  Data usage: {current_percentage:.1f}%")
        print(f"  Val accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Test accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Labeled samples: {len(X_labeled)}")

        if data_manager.is_pool_empty():
            print("Pool is empty, stopping Active Learning")
            break

        increment_samples = max(
            1, int(increment_percentage * len(data_manager.X_train_full))
        )
        batch_size = min(increment_samples, len(data_manager.X_pool))

        if batch_size <= 0:
            break

        X_pool, y_pool = data_manager.get_pool_data()
        selected_X, selected_y, selected_indices = al_manager.select_next_batch(
            X_pool, y_pool, model_trainer.model, batch_size
        )
        data_manager.add_samples_to_labeled_set(selected_indices)

    return {
        "sampling_strategy": sampling_strategy,
        "total_iterations": len(iteration_results),
        "final_labeled_samples": (
            iteration_results[-1]["labeled_samples"] if iteration_results else 0
        ),
        "final_data_percentage": (
            iteration_results[-1]["data_percentage"] if iteration_results else 0
        ),
        "iteration_results": iteration_results,
    }


@task
def log_active_learning_to_mlflow(
    al_results: Dict[str, Any], experiment_name: str = "step4_active_learning"
) -> None:
    """Log Active Learning results to MLflow."""
    mlflow_tracker = MLflowTracker(experiment_name=experiment_name)
    sampling_strategy = al_results["sampling_strategy"]

    with mlflow_tracker.start_run(run_name=f"active_learning_{sampling_strategy}"):
        final_results = al_results["iteration_results"][-1]

        mlflow_tracker.log_params(
            {
                "method": "active_learning",
                "sampling_strategy": sampling_strategy,
                "total_iterations": al_results["total_iterations"],
                "final_labeled_samples": al_results["final_labeled_samples"],
                "final_data_percentage": al_results["final_data_percentage"],
            }
        )

        mlflow_tracker.log_metrics(
            {f"val_{k}": v for k, v in final_results["val_metrics"].items()}
        )
        mlflow_tracker.log_metrics(
            {f"test_{k}": v for k, v in final_results["test_metrics"].items()}
        )

        for i, result in enumerate(al_results["iteration_results"]):
            mlflow_tracker.log_metrics(
                {
                    f"iteration_{i+1}_val_accuracy": result["val_metrics"]["accuracy"],
                    f"iteration_{i+1}_test_accuracy": result["test_metrics"][
                        "accuracy"
                    ],
                    f"iteration_{i+1}_labeled_samples": result["labeled_samples"],
                    f"iteration_{i+1}_data_percentage": result["data_percentage"],
                },
                step=i + 1,
            )


@flow(name="Active Learning Pipeline")
def active_learning_pipeline(
    sampling_strategy: str = "entropy",
    initial_percentage: float = 0.05,
    increment_percentage: float = 0.05,
    max_iterations: int = 15,
    random_seed: int = 42,
    experiment_name: str = "step4_active_learning",
) -> Dict[str, Any]:
    """Complete Active Learning pipeline with incremental data addition."""
    data_manager = setup_data_manager(random_seed=random_seed)

    al_results = run_active_learning_experiment(
        data_manager=data_manager,
        sampling_strategy=sampling_strategy,
        initial_percentage=initial_percentage,
        increment_percentage=increment_percentage,
        max_iterations=max_iterations,
        random_seed=random_seed,
    )

    log_active_learning_to_mlflow(
        al_results=al_results, experiment_name=experiment_name
    )

    final_result = al_results["iteration_results"][-1]
    summary = {
        "method": "active_learning",
        "strategy": sampling_strategy,
        "test_accuracy": final_result["test_metrics"]["accuracy"],
        "val_accuracy": final_result["val_metrics"]["accuracy"],
        "final_labeled_samples": al_results["final_labeled_samples"],
        "final_data_percentage": al_results["final_data_percentage"],
        "total_iterations": al_results["total_iterations"],
    }

    print("=== ACTIVE LEARNING RESULTS ===")
    print(f"Strategy: {summary['strategy']}")
    print(f"Test Accuracy: {summary['test_accuracy']:.4f}")
    print(f"Val Accuracy: {summary['val_accuracy']:.4f}")
    print(f"Final Data Usage: {summary['final_data_percentage']:.1f}%")
    print(f"Final Labeled Samples: {summary['final_labeled_samples']}")
    print(f"Total Iterations: {summary['total_iterations']}")

    return summary
