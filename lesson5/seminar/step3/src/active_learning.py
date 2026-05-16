"""Active Learning utilities for uncertainty sampling."""

import logging
from typing import Tuple
import numpy as np
from catboost import CatBoostClassifier

logger = logging.getLogger(__name__)


class UncertaintySampler:
    """Implements uncertainty sampling strategies for Active Learning."""

    def __init__(self, strategy: str = "entropy"):
        """Initialize uncertainty sampler.

        Args:
            strategy: Uncertainty sampling strategy ("entropy", "margin", "least_confident")
        """
        self.strategy = strategy
        self.supported_strategies = ["entropy", "margin", "least_confident"]

        if strategy not in self.supported_strategies:
            raise ValueError(
                f"Strategy {strategy} not supported. Use one of: {self.supported_strategies}"
            )

    def calculate_uncertainty(self, probabilities: np.ndarray) -> np.ndarray:
        """Calculate uncertainty scores for predictions.

        Args:
            probabilities: Prediction probabilities shape (n_samples, n_classes)

        Returns:
            Uncertainty scores for each sample
        """
        if self.strategy == "entropy":
            return self._entropy_uncertainty(probabilities)
        elif self.strategy == "margin":
            return self._margin_uncertainty(probabilities)
        elif self.strategy == "least_confident":
            return self._least_confident_uncertainty(probabilities)

    def _entropy_uncertainty(self, probabilities: np.ndarray) -> np.ndarray:
        """Calculate entropy-based uncertainty."""
        eps = 1e-10
        probabilities = np.clip(probabilities, eps, 1 - eps)

        entropy = -np.sum(probabilities * np.log(probabilities), axis=1)
        return entropy

    def _margin_uncertainty(self, probabilities: np.ndarray) -> np.ndarray:
        """Calculate margin-based uncertainty (difference between top 2 predictions)."""
        sorted_probs = np.sort(probabilities, axis=1)[:, ::-1]

        margin = sorted_probs[:, 0] - sorted_probs[:, 1]
        return -margin

    def _least_confident_uncertainty(self, probabilities: np.ndarray) -> np.ndarray:
        """Calculate least confident uncertainty (1 - max probability)."""
        max_probs = np.max(probabilities, axis=1)
        return 1 - max_probs

    def select_samples(
        self,
        X_pool: np.ndarray,
        y_pool: np.ndarray,
        model: CatBoostClassifier,
        n_samples: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select most uncertain samples from the pool.

        Args:
            X_pool: Pool of unlabeled samples
            y_pool: True labels for pool samples (for evaluation)
            model: Trained model for making predictions
            n_samples: Number of samples to select

        Returns:
            Tuple of (selected_X, selected_y, selected_indices)
        """
        logger.info(
            f"Selecting {n_samples} samples using {self.strategy} uncertainty sampling..."
        )

        probabilities = model.predict_proba(X_pool)

        uncertainty_scores = self.calculate_uncertainty(probabilities)

        uncertain_indices = np.argsort(uncertainty_scores)[::-1][:n_samples]

        selected_X = X_pool[uncertain_indices]
        selected_y = y_pool[uncertain_indices]

        mean_uncertainty = np.mean(uncertainty_scores[uncertain_indices])
        logger.info(f"Selected samples with mean uncertainty: {mean_uncertainty:.4f}")

        return selected_X, selected_y, uncertain_indices


class ActiveLearningManager:
    """Manages the Active Learning process."""

    def __init__(self, uncertainty_strategy: str = "entropy"):
        """Initialize Active Learning manager.

        Args:
            uncertainty_strategy: Strategy for uncertainty sampling
        """
        self.sampler = UncertaintySampler(uncertainty_strategy)
        self.training_history = []

    def select_next_batch(
        self,
        X_pool: np.ndarray,
        y_pool: np.ndarray,
        model: CatBoostClassifier,
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select next batch of samples for labeling.

        Args:
            X_pool: Pool of unlabeled samples
            y_pool: True labels for evaluation
            model: Current trained model
            batch_size: Number of samples to select

        Returns:
            Tuple of (selected_X, selected_y, selected_indices)
        """
        return self.sampler.select_samples(X_pool, y_pool, model, batch_size)

    def evaluate_selection_quality(
        self,
        selected_y: np.ndarray,
        y_pool: np.ndarray,
        class_distribution: dict = None,
    ) -> dict:
        """Evaluate the quality of selected samples.

        Args:
            selected_y: Labels of selected samples
            y_pool: All available labels in pool
            class_distribution: Original class distribution

        Returns:
            Dictionary with selection quality metrics
        """
        unique_classes, counts = np.unique(selected_y, return_counts=True)
        selection_dist = dict(zip(unique_classes, counts))

        pool_unique, pool_counts = np.unique(y_pool, return_counts=True)
        pool_dist = dict(zip(pool_unique, pool_counts))

        diversity = len(unique_classes)

        metrics = {
            "selection_size": len(selected_y),
            "unique_classes_selected": diversity,
            "total_classes_available": len(pool_unique),
            "selection_distribution": selection_dist,
            "pool_distribution": pool_dist,
            "diversity_ratio": diversity / len(pool_unique),
        }

        logger.info(
            f"Selection quality: {diversity}/{len(pool_unique)} classes, "
            f"diversity ratio: {metrics['diversity_ratio']:.3f}"
        )

        return metrics

    def update_training_history(self, iteration: int, metrics: dict):
        """Update training history with iteration results.

        Args:
            iteration: Current iteration number
            metrics: Training and evaluation metrics
        """
        history_entry = {
            "iteration": iteration,
            "timestamp": np.datetime64("now"),
            **metrics,
        }

        self.training_history.append(history_entry)
        logger.info(f"Updated training history for iteration {iteration}")

    def get_training_summary(self) -> dict:
        """Get summary of training progress.

        Returns:
            Dictionary with training summary statistics
        """
        if not self.training_history:
            return {"status": "no_training_history"}

        iterations = [entry["iteration"] for entry in self.training_history]
        val_accuracies = [
            entry.get("val_accuracy", 0) for entry in self.training_history
        ]
        train_sizes = [entry.get("train_size", 0) for entry in self.training_history]

        best_val_idx = np.argmax(val_accuracies)
        best_iteration = self.training_history[best_val_idx]

        summary = {
            "total_iterations": len(iterations),
            "best_iteration": best_iteration["iteration"],
            "best_val_accuracy": best_iteration.get("val_accuracy", 0),
            "best_test_accuracy": best_iteration.get("test_accuracy", 0),
            "final_train_size": train_sizes[-1] if train_sizes else 0,
            "accuracy_improvement": (
                val_accuracies[-1] - val_accuracies[0] if len(val_accuracies) > 1 else 0
            ),
            "training_history": self.training_history,
        }

        return summary
