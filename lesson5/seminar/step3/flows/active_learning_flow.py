"""
Active Learning flow using Prefect and MLflow.
"""

import sys
import os
import logging
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from prefect import flow, task
from data_manager import ActiveLearningDataManager
from model_trainer import ModelTrainer
from mlflow_tracker import MLflowTracker
from active_learning import ActiveLearningManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@task(name="Initialize AL Data Manager", log_prints=True)
def initialize_al_data_manager() -> ActiveLearningDataManager:
    """Initialize and load data using ActiveLearningDataManager."""
    print("Initializing ActiveLearningDataManager...")
    data_manager = ActiveLearningDataManager(random_seed=42)
    dataset_info = data_manager.load_and_split_data()

    print(f"Dataset loaded: {dataset_info}")
    return data_manager


@task(name="Initialize AL Manager", log_prints=True)
def initialize_al_manager(
    uncertainty_strategy: str = "entropy",
) -> ActiveLearningManager:
    """Initialize Active Learning manager."""
    print(
        f"Initializing Active Learning manager with {uncertainty_strategy} strategy..."
    )
    al_manager = ActiveLearningManager(uncertainty_strategy)
    return al_manager


@task(name="Initialize MLflow Tracker", log_prints=True)
def initialize_mlflow_tracker() -> MLflowTracker:
    """Initialize MLflow tracker."""
    print("Initializing MLflow tracker...")
    tracker = MLflowTracker(experiment_name="active_learning_step3")
    experiment_info = tracker.get_experiment_info()
    print(f"MLflow experiment: {experiment_info}")
    return tracker


@task(name="Setup Initial AL", log_prints=True)
def setup_initial_active_learning(
    data_manager: ActiveLearningDataManager, initial_percentage: float = 0.1
) -> dict:
    """Setup initial Active Learning state."""
    print(
        f"Setting up Active Learning with {initial_percentage*100:.0f}% initial data..."
    )

    init_info = data_manager.initialize_active_learning(initial_percentage)

    print("Initial setup completed:")
    print(f"  Labeled samples: {init_info['initial_labeled_size']}")
    print(f"  Pool samples: {init_info['initial_pool_size']}")

    return init_info


@task(name="Train AL Model Iteration", log_prints=True)
def train_al_model_iteration(
    data_manager: ActiveLearningDataManager,
    al_manager: ActiveLearningManager,
    tracker: MLflowTracker,
    iteration: int,
    previous_model_path: str = None,
) -> dict:
    """Train model for Active Learning iteration."""
    print(f"Training Active Learning iteration {iteration}...")

    X_labeled, y_labeled = data_manager.get_labeled_data()
    X_val, y_val = data_manager.get_validation_data()
    X_test, y_test = data_manager.get_test_data()

    trainer = ModelTrainer()

    model = trainer.train_model(X_labeled, y_labeled)

    val_metrics = trainer.evaluate_model(X_val, y_val, "validation")
    test_metrics = trainer.evaluate_model(X_test, y_test, "test")

    model_path = f"models/catboost_model_al_iter_{iteration:02d}.cbm"
    trainer.save_model(model_path)

    metrics_dir = Path("metrics") / f"al_iteration_{iteration:02d}"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_metrics(val_metrics, str(metrics_dir / "val_metrics.json"))
    trainer.save_metrics(test_metrics, str(metrics_dir / "test_metrics.json"))

    data_info = data_manager.get_data_info()
    labeled_percentage = data_info.get("total_labeled_ratio", 0) * 100

    run_id = tracker.log_training_iteration(
        iteration=iteration,
        train_size=len(X_labeled),
        train_percentage=labeled_percentage,
        model_params=trainer.model_params,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        model=model,
    )

    result = {
        "iteration": iteration,
        "labeled_percentage": labeled_percentage,
        "labeled_size": len(X_labeled),
        "val_accuracy": val_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "val_f1_macro": val_metrics["f1_macro"],
        "test_f1_macro": test_metrics["f1_macro"],
        "model_path": model_path,
        "mlflow_run_id": run_id,
        "model": model,
    }

    al_manager.update_training_history(iteration, result)

    print(f"Iteration {iteration} completed:")
    print(f"  Labeled size: {len(X_labeled)} ({labeled_percentage:.1f}%)")
    print(f"  Val accuracy: {val_metrics['accuracy']:.4f}")
    print(f"  Test accuracy: {test_metrics['accuracy']:.4f}")

    return result


@task(name="Select Next AL Batch", log_prints=True)
def select_next_al_batch(
    data_manager: ActiveLearningDataManager,
    al_manager: ActiveLearningManager,
    model,
    batch_size: int,
) -> dict:
    """Select next batch of samples using Active Learning."""
    print(f"Selecting next batch of {batch_size} samples...")

    if data_manager.is_pool_empty():
        print("Pool is empty, no more samples to select")
        return {"samples_selected": 0, "pool_empty": True}

    X_pool, y_pool = data_manager.get_pool_data()

    actual_batch_size = min(batch_size, len(X_pool))

    selected_X, selected_y, selected_indices = al_manager.select_next_batch(
        X_pool, y_pool, model, actual_batch_size
    )

    selection_quality = al_manager.evaluate_selection_quality(selected_y, y_pool)

    update_info = data_manager.add_samples_to_labeled_set(selected_indices)

    result = {
        "samples_selected": len(selected_indices),
        "pool_empty": False,
        "selection_quality": selection_quality,
        "update_info": update_info,
    }

    print(f"Selected {len(selected_indices)} samples:")
    print(f"  New labeled size: {update_info['new_labeled_size']}")
    print(f"  Remaining pool size: {update_info['new_pool_size']}")
    print(f"  Diversity ratio: {selection_quality['diversity_ratio']:.3f}")

    return result


@flow(name="Active Learning Pipeline", log_prints=True)
def active_learning_pipeline(
    initial_percentage: float = 0.1,
    batch_size: int = 1000,
    max_iterations: int = 10,
    uncertainty_strategy: str = "entropy",
):
    """
    Main Active Learning pipeline.

    Args:
        initial_percentage: Initial percentage of training data to label
        batch_size: Size of each AL batch (default: 10% of remaining data)
        max_iterations: Maximum number of AL iterations
        uncertainty_strategy: Uncertainty sampling strategy
    """
    print("Starting Active Learning Pipeline")
    print(f"Initial percentage: {initial_percentage*100:.0f}%")
    print(f"Max iterations: {max_iterations}")
    print(f"Uncertainty strategy: {uncertainty_strategy}")
    print("=" * 60)

    data_manager = initialize_al_data_manager()
    al_manager = initialize_al_manager(uncertainty_strategy)
    tracker = initialize_mlflow_tracker()

    Path("models").mkdir(exist_ok=True)
    Path("metrics").mkdir(exist_ok=True)

    init_info = setup_initial_active_learning(data_manager, initial_percentage)

    if batch_size <= 0:
        batch_size = max(1000, int(0.1 * init_info["initial_pool_size"]))

    print(f"Using batch size: {batch_size}")

    results = []
    current_model = None

    for iteration in range(1, max_iterations + 1):
        print(f"\n--- Active Learning Iteration {iteration} ---")

        train_result = train_al_model_iteration(
            data_manager=data_manager,
            al_manager=al_manager,
            tracker=tracker,
            iteration=iteration,
            previous_model_path=current_model,
        )

        results.append(train_result)
        current_model = train_result["model"]

        if iteration < max_iterations:
            selection_result = select_next_al_batch(
                data_manager=data_manager,
                al_manager=al_manager,
                model=current_model,
                batch_size=batch_size,
            )

            if selection_result["pool_empty"]:
                print(f"Pool exhausted at iteration {iteration}")
                break

            if selection_result["samples_selected"] == 0:
                print(f"No samples selected at iteration {iteration}")
                break

    print("\n" + "=" * 60)
    print("ACTIVE LEARNING SUMMARY")
    print("=" * 60)

    for result in results:
        print(
            f"Iteration {result['iteration']:2d}: "
            f"{result['labeled_percentage']:5.1f}% labeled, "
            f"Val Acc: {result['val_accuracy']:.4f}, "
            f"Test Acc: {result['test_accuracy']:.4f}"
        )

    best_result = max(results, key=lambda x: x["val_accuracy"])
    print(
        f"\nBest iteration: {best_result['iteration']} "
        f"(Val Acc: {best_result['val_accuracy']:.4f})"
    )

    training_summary = al_manager.get_training_summary()

    print("\nActive Learning completed!")
    print(f"Total iterations: {len(results)}")
    print(f"Final labeled percentage: {results[-1]['labeled_percentage']:.1f}%")
    print(
        f"Accuracy improvement: {training_summary.get('accuracy_improvement', 0):.4f}"
    )
    print("Check MLflow UI for detailed experiment tracking")

    return {
        "total_iterations": len(results),
        "results": results,
        "best_iteration": best_result,
        "training_summary": training_summary,
        "final_data_info": data_manager.get_data_info(),
    }


@flow(name="Single AL Iteration", log_prints=True)
def single_al_iteration_pipeline(
    iteration: int = 1,
    initial_percentage: float = 0.1,
    uncertainty_strategy: str = "entropy",
):
    """
    Run a single Active Learning iteration.

    Args:
        iteration: Iteration number
        initial_percentage: Initial percentage of data to label
        uncertainty_strategy: Uncertainty sampling strategy
    """
    print(
        f"Running single AL iteration {iteration} with {initial_percentage*100:.0f}% initial data"
    )

    data_manager = initialize_al_data_manager()
    al_manager = initialize_al_manager(uncertainty_strategy)
    tracker = initialize_mlflow_tracker()

    Path("models").mkdir(exist_ok=True)
    Path("metrics").mkdir(exist_ok=True)

    setup_initial_active_learning(data_manager, initial_percentage)

    result = train_al_model_iteration(
        data_manager=data_manager,
        al_manager=al_manager,
        tracker=tracker,
        iteration=iteration,
    )

    print(f"Single AL iteration {iteration} completed!")
    return result


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "single":
            iteration = int(sys.argv[2]) if len(sys.argv) > 2 else 1
            initial_pct = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
            strategy = sys.argv[4] if len(sys.argv) > 4 else "entropy"
            single_al_iteration_pipeline(iteration, initial_pct, strategy)
        else:
            max_iters = int(sys.argv[1])
            strategy = sys.argv[2] if len(sys.argv) > 2 else "entropy"
            active_learning_pipeline(
                max_iterations=max_iters, uncertainty_strategy=strategy
            )
    else:
        active_learning_pipeline()
