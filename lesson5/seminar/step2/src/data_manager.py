"""Data management for Continuous Training."""

import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DataManager:
    """Manages data loading and splitting for Continuous Training."""

    def __init__(self, random_seed: int = 42):
        """Initialize DataManager.

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

        return {
            "status": "loaded",
            "train_samples": len(self.X_train_full),
            "val_samples": len(self.X_val),
            "test_samples": len(self.X_test),
            "features": self.X_train_full.shape[1],
        }
