"""Data management for Active Learning."""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ActiveLearningDataManager:
    """Manages data loading and splitting for Active Learning."""

    def __init__(self, random_seed: int = 42):
        """Initialize ActiveLearningDataManager."""
        self.random_seed = random_seed
        self.X_train_full = None
        self.X_val = None
        self.X_test = None
        self.y_train_full = None
        self.y_val = None
        self.y_test = None
        self._is_loaded = False
        self.X_labeled = None
        self.y_labeled = None
        self.X_pool = None
        self.y_pool = None
        self.labeled_indices = []
        self.pool_indices = []

    def load_and_split_data(self) -> Dict[str, Any]:
        """Load Digits dataset and create train/val/test splits."""
        digits = load_digits()
        X, y = digits.data, digits.target
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_seed, stratify=y
        )
        self.X_train_full, self.X_val, self.y_train_full, self.y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=0.2,
            random_state=self.random_seed,
            stratify=y_temp,
        )
        self._is_loaded = True
        dataset_info = {
            "total_samples": len(X),
            "train_samples": len(self.X_train_full),
            "val_samples": len(self.X_val),
            "test_samples": len(self.X_test),
            "features": X.shape[1],
            "classes": len(np.unique(y)),
            "class_distribution": dict(zip(*np.unique(y, return_counts=True))),
        }
        return dataset_info

    def initialize_active_learning(
        self, initial_percentage: float = 0.1
    ) -> Dict[str, Any]:
        """Initialize Active Learning with a small labeled set."""
        if not self._is_loaded:
            raise ValueError("Data not loaded. Call load_and_split_data() first.")

        n_initial = max(1, int(initial_percentage * len(self.X_train_full)))
        initial_indices = np.random.RandomState(self.random_seed).choice(
            len(self.X_train_full), size=n_initial, replace=False
        )
        self.labeled_indices = initial_indices.tolist()
        self.pool_indices = [
            i for i in range(len(self.X_train_full)) if i not in self.labeled_indices
        ]
        self._update_active_learning_sets()

        init_info = {
            "initial_labeled_samples": len(self.labeled_indices),
            "initial_pool_samples": len(self.pool_indices),
            "initial_percentage": initial_percentage,
            "labeled_class_distribution": dict(
                zip(*np.unique(self.y_labeled, return_counts=True))
            ),
        }
        return init_info

    def _update_active_learning_sets(self):
        """Update labeled and pool sets based on current indices."""
        self.X_labeled = self.X_train_full[self.labeled_indices]
        self.y_labeled = self.y_train_full[self.labeled_indices]
        self.X_pool = self.X_train_full[self.pool_indices]
        self.y_pool = self.y_train_full[self.pool_indices]

    def add_samples_to_labeled_set(
        self, pool_indices_to_add: np.ndarray
    ) -> Dict[str, Any]:
        """Add samples from pool to labeled set."""
        if len(pool_indices_to_add) == 0:
            return {"samples_added": 0}

        actual_indices = [self.pool_indices[i] for i in pool_indices_to_add]
        self.labeled_indices.extend(actual_indices)
        for idx in sorted(pool_indices_to_add, reverse=True):
            self.pool_indices.pop(idx)

        self._update_active_learning_sets()

        update_info = {
            "samples_added": len(pool_indices_to_add),
            "new_labeled_size": len(self.labeled_indices),
            "new_pool_size": len(self.pool_indices),
            "labeled_percentage": len(self.labeled_indices)
            / len(self.X_train_full)
            * 100,
        }
        return update_info

    def get_next_pool_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get next batch from pool for selection."""
        if self.is_pool_empty():
            return np.array([]), np.array([])

        actual_batch_size = min(batch_size, len(self.X_pool))
        batch_indices = np.arange(actual_batch_size)
        return self.X_pool[batch_indices], self.y_pool[batch_indices]

    def get_labeled_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current labeled dataset."""
        if self.X_labeled is None:
            raise ValueError(
                "Active Learning not initialized. Call initialize_active_learning() first."
            )
        return self.X_labeled, self.y_labeled

    def get_pool_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current pool dataset."""
        if self.X_pool is None:
            raise ValueError(
                "Active Learning not initialized. Call initialize_active_learning() first."
            )
        return self.X_pool, self.y_pool

    def is_pool_empty(self) -> bool:
        """Check if pool is empty."""
        return len(self.pool_indices) == 0

    def get_active_learning_stats(self) -> Dict[str, Any]:
        """Get current Active Learning statistics."""
        if not hasattr(self, "labeled_indices"):
            return {"status": "not_initialized"}

        return {
            "labeled_samples": len(self.labeled_indices),
            "pool_samples": len(self.pool_indices),
            "total_train_samples": len(self.X_train_full),
            "labeled_percentage": len(self.labeled_indices)
            / len(self.X_train_full)
            * 100,
            "pool_percentage": len(self.pool_indices) / len(self.X_train_full) * 100,
            "labeled_class_distribution": (
                dict(zip(*np.unique(self.y_labeled, return_counts=True)))
                if self.y_labeled is not None
                else {}
            ),
        }

    def reset_active_learning(self):
        """Reset Active Learning state."""
        self.X_labeled = None
        self.y_labeled = None
        self.X_pool = None
        self.y_pool = None
        self.labeled_indices = []
        self.pool_indices = []

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get comprehensive dataset information."""
        if not self._is_loaded:
            return {"status": "not_loaded"}

        info = {
            "dataset_name": "Digits",
            "total_samples": len(self.X_train_full)
            + len(self.X_val)
            + len(self.X_test),
            "train_samples": len(self.X_train_full),
            "val_samples": len(self.X_val),
            "test_samples": len(self.X_test),
            "features": self.X_train_full.shape[1],
            "classes": len(np.unique(self.y_train_full)),
            "class_names": list(range(len(np.unique(self.y_train_full)))),
        }

        if hasattr(self, "labeled_indices"):
            info.update(self.get_active_learning_stats())

        return info
