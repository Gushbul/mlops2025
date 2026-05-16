"""Active Learning utilities for uncertainty sampling."""

import logging
from typing import Tuple
import numpy as np
from catboost import CatBoostClassifier

logger = logging.getLogger(__name__)


class UncertaintySampler:
    """Implements uncertainty sampling strategies for Active Learning."""

    def __init__(self, strategy: str = "entropy"):
        """Initialize uncertainty sampler."""
        self.strategy = strategy
        self.supported_strategies = ["entropy", "margin", "least_confident"]

        if strategy not in self.supported_strategies:
            raise ValueError(
                f"Strategy {strategy} not supported. Use one of: {self.supported_strategies}"
            )

    def calculate_uncertainty(self, probabilities: np.ndarray) -> np.ndarray:
        """Calculate uncertainty scores for predictions."""
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
        """Calculate margin-based uncertainty."""
        sorted_probs = np.sort(probabilities, axis=1)[:, ::-1]
        margin = sorted_probs[:, 0] - sorted_probs[:, 1]
        return -margin

    def _least_confident_uncertainty(self, probabilities: np.ndarray) -> np.ndarray:
        """Calculate least confident uncertainty."""
        max_probs = np.max(probabilities, axis=1)
        return 1 - max_probs

    def select_samples(
        self,
        X_pool: np.ndarray,
        y_pool: np.ndarray,
        model: CatBoostClassifier,
        n_samples: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select most uncertain samples from the pool."""
        probabilities = model.predict_proba(X_pool)
        uncertainty_scores = self.calculate_uncertainty(probabilities)
        uncertain_indices = np.argsort(uncertainty_scores)[::-1][:n_samples]
        selected_X = X_pool[uncertain_indices]
        selected_y = y_pool[uncertain_indices]
        return selected_X, selected_y, uncertain_indices


class ActiveLearningManager:
    """Manages the Active Learning process."""

    def __init__(self, uncertainty_strategy: str = "entropy"):
        """Initialize Active Learning manager."""
        self.uncertainty_strategy = uncertainty_strategy
        self.sampler = UncertaintySampler(strategy=uncertainty_strategy)
        self.selection_history = []

    def select_next_batch(
        self,
        X_pool: np.ndarray,
        y_pool: np.ndarray,
        model: CatBoostClassifier,
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select next batch of samples for labeling."""
        if len(X_pool) == 0:
            raise ValueError("Pool is empty, cannot select samples")

        actual_batch_size = min(batch_size, len(X_pool))
        selected_X, selected_y, selected_indices = self.sampler.select_samples(
            X_pool, y_pool, model, actual_batch_size
        )
        selection_info = {
            "batch_size": actual_batch_size,
            "strategy": self.uncertainty_strategy,
            "pool_size_before": len(X_pool),
            "selected_indices": selected_indices.tolist(),
        }
        self.selection_history.append(selection_info)
        return selected_X, selected_y, selected_indices

    def get_selection_stats(self, selected_y: np.ndarray, pool_y: np.ndarray) -> dict:
        """Get statistics about the selection."""
        unique_classes, class_counts = np.unique(selected_y, return_counts=True)
        selection_distribution = dict(zip(unique_classes, class_counts))
        pool_unique_classes, pool_class_counts = np.unique(pool_y, return_counts=True)
        pool_distribution = dict(zip(pool_unique_classes, pool_class_counts))
        diversity = len(unique_classes)
        return {
            "selection_distribution": selection_distribution,
            "pool_distribution": pool_distribution,
            "diversity": diversity,
            "total_selected": len(selected_y),
            "total_pool": len(pool_y),
        }

    def get_learning_curve_data(self) -> dict:
        """Extract learning curve data from selection history."""
        if not self.selection_history:
            return {"iterations": [], "pool_sizes": [], "batch_sizes": []}

        iterations = list(range(1, len(self.selection_history) + 1))
        pool_sizes = [info["pool_size_before"] for info in self.selection_history]
        batch_sizes = [info["batch_size"] for info in self.selection_history]
        return {
            "iterations": iterations,
            "pool_sizes": pool_sizes,
            "batch_sizes": batch_sizes,
        }

    def reset_history(self):
        """Reset selection history."""
        self.selection_history = []

    def get_strategy_info(self) -> dict:
        """Get information about the current strategy."""
        return {
            "strategy": self.uncertainty_strategy,
            "total_selections": len(self.selection_history),
            "supported_strategies": self.sampler.supported_strategies,
        }
