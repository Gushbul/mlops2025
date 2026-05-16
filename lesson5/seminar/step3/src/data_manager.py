"""Data management for Active Learning."""

import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ActiveLearningDataManager:
    """Manages data loading and splitting for Active Learning."""

    def __init__(self, random_seed: int = 42):
        """Initialize ActiveLearningDataManager.

        Args:
            random_seed: Random seed for reproducibility
        """
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
        """Load Forest Cover Type dataset and create train/val/test splits.

        Returns:
            Dictionary with dataset information
        """
        logger.info("Loading Forest Cover Type dataset...")

        covtype = fetch_covtype()
        X, y = covtype.data, covtype.target

        logger.info(f"Dataset shape: {X.shape}")
        logger.info(f"Number of classes: {len(np.unique(y))}")

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

        logger.info("Data split completed:")
        logger.info(f"  Train: {dataset_info['train_samples']} samples")
        logger.info(f"  Validation: {dataset_info['val_samples']} samples")
        logger.info(f"  Test: {dataset_info['test_samples']} samples")

        return dataset_info

    def get_incremental_train_data(
        self, percentage: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get incremental training data based on percentage.

        Args:
            percentage: Percentage of training data to use (0.1 to 1.0)

        Returns:
            Tuple of (X_train_subset, y_train_subset)
        """
        if not self._is_loaded:
            raise ValueError("Data not loaded. Call load_and_split_data() first.")

        if not 0.1 <= percentage <= 1.0:
            raise ValueError("Percentage must be between 0.1 and 1.0")

        n_samples = int(percentage * len(self.X_train_full))

        X_train_subset = self.X_train_full[:n_samples]
        y_train_subset = self.y_train_full[:n_samples]

        logger.info(
            f"Using {percentage*100:.0f}% of training data: {n_samples} samples"
        )

        return X_train_subset, y_train_subset

    def get_validation_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get validation data.

        Returns:
            Tuple of (X_val, y_val)
        """
        if not self._is_loaded:
            raise ValueError("Data not loaded. Call load_and_split_data() first.")

        return self.X_val, self.y_val

    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get test data.

        Returns:
            Tuple of (X_test, y_test)
        """
        if not self._is_loaded:
            raise ValueError("Data not loaded. Call load_and_split_data() first.")

        return self.X_test, self.y_test

    def get_data_info(self) -> Dict[str, Any]:
        """Get information about current data splits.

        Returns:
            Dictionary with data information
        """
        if not self._is_loaded:
            return {"status": "not_loaded"}

        al_info = {}
        if self.X_labeled is not None:
            al_info.update(
                {
                    "labeled_samples": len(self.X_labeled),
                    "pool_samples": len(self.X_pool) if self.X_pool is not None else 0,
                    "total_labeled_ratio": (
                        len(self.X_labeled) / len(self.X_train_full)
                        if len(self.X_train_full) > 0
                        else 0
                    ),
                }
            )

        return {
            "status": "loaded",
            "train_samples": len(self.X_train_full),
            "val_samples": len(self.X_val),
            "test_samples": len(self.X_test),
            "features": self.X_train_full.shape[1],
            **al_info,
        }

    def initialize_active_learning(
        self, initial_percentage: float = 0.1
    ) -> Dict[str, Any]:
        """Initialize Active Learning with initial labeled set.

        Args:
            initial_percentage: Percentage of training data to use as initial labeled set

        Returns:
            Dictionary with initialization information
        """
        if not self._is_loaded:
            raise ValueError("Data not loaded. Call load_and_split_data() first.")

        n_initial = int(initial_percentage * len(self.X_train_full))

        self.labeled_indices = list(range(n_initial))
        self.pool_indices = list(range(n_initial, len(self.X_train_full)))

        self.X_labeled = self.X_train_full[self.labeled_indices].copy()
        self.y_labeled = self.y_train_full[self.labeled_indices].copy()
        self.X_pool = self.X_train_full[self.pool_indices].copy()
        self.y_pool = self.y_train_full[self.pool_indices].copy()

        init_info = {
            "initial_labeled_size": len(self.X_labeled),
            "initial_pool_size": len(self.X_pool),
            "initial_percentage": initial_percentage,
            "labeled_class_distribution": dict(
                zip(*np.unique(self.y_labeled, return_counts=True))
            ),
            "pool_class_distribution": dict(
                zip(*np.unique(self.y_pool, return_counts=True))
            ),
        }

        logger.info("Active Learning initialized:")
        logger.info(f"  Initial labeled: {len(self.X_labeled)} samples")
        logger.info(f"  Pool: {len(self.X_pool)} samples")
        logger.info(
            f"  Labeled classes: {list(init_info['labeled_class_distribution'].keys())}"
        )

        return init_info

    def add_samples_to_labeled_set(
        self, selected_indices: np.ndarray
    ) -> Dict[str, Any]:
        """Add selected samples from pool to labeled set.

        Args:
            selected_indices: Indices of samples to move from pool to labeled set

        Returns:
            Dictionary with update information
        """
        if self.X_pool is None or self.y_pool is None:
            raise ValueError(
                "Pool not initialized. Call initialize_active_learning() first."
            )

        selected_X = self.X_pool[selected_indices]
        selected_y = self.y_pool[selected_indices]

        self.X_labeled = np.vstack([self.X_labeled, selected_X])
        self.y_labeled = np.concatenate([self.y_labeled, selected_y])

        pool_indices_to_add = [self.pool_indices[i] for i in selected_indices]
        self.labeled_indices.extend(pool_indices_to_add)

        remaining_indices = np.setdiff1d(range(len(self.X_pool)), selected_indices)
        self.X_pool = self.X_pool[remaining_indices]
        self.y_pool = self.y_pool[remaining_indices]
        self.pool_indices = [self.pool_indices[i] for i in remaining_indices]

        update_info = {
            "samples_added": len(selected_indices),
            "new_labeled_size": len(self.X_labeled),
            "new_pool_size": len(self.X_pool),
            "added_class_distribution": dict(
                zip(*np.unique(selected_y, return_counts=True))
            ),
            "total_labeled_ratio": len(self.X_labeled) / len(self.X_train_full),
        }

        logger.info(f"Added {len(selected_indices)} samples to labeled set")
        logger.info(f"  New labeled size: {len(self.X_labeled)}")
        logger.info(f"  Remaining pool size: {len(self.X_pool)}")

        return update_info

    def get_next_pool_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get next batch from pool for uncertainty evaluation.

        Args:
            batch_size: Size of batch to return

        Returns:
            Tuple of (X_batch, y_batch) from pool
        """
        if self.X_pool is None or len(self.X_pool) == 0:
            raise ValueError("Pool is empty or not initialized")

        actual_batch_size = min(batch_size, len(self.X_pool))
        return self.X_pool[:actual_batch_size], self.y_pool[:actual_batch_size]

    def get_labeled_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current labeled dataset.

        Returns:
            Tuple of (X_labeled, y_labeled)
        """
        if self.X_labeled is None:
            raise ValueError(
                "Labeled set not initialized. Call initialize_active_learning() first."
            )

        return self.X_labeled, self.y_labeled

    def get_pool_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current pool dataset.

        Returns:
            Tuple of (X_pool, y_pool)
        """
        if self.X_pool is None:
            raise ValueError(
                "Pool not initialized. Call initialize_active_learning() first."
            )

        return self.X_pool, self.y_pool

    def is_pool_empty(self) -> bool:
        """Check if pool is empty.

        Returns:
            True if pool is empty, False otherwise
        """
        return self.X_pool is None or len(self.X_pool) == 0
