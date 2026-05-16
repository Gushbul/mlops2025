"""MLflow integration for experiment tracking."""

import logging
import mlflow
import mlflow.catboost
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class MLflowTracker:
    """Handles MLflow experiment tracking."""

    def __init__(
        self, experiment_name: str = "continuous_training", tracking_uri: str = None
    ):
        """Initialize MLflow tracker.

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking URI (default: local mlruns)
        """
        self.experiment_name = experiment_name

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(
                    f"Created new experiment: {experiment_name} (ID: {experiment_id})"
                )
            else:
                experiment_id = experiment.experiment_id
                logger.info(
                    f"Using existing experiment: {experiment_name} (ID: {experiment_id})"
                )

            mlflow.set_experiment(experiment_name)

        except Exception as e:
            logger.error(f"Error setting up MLflow experiment: {e}")
            raise

    def start_run(
        self, run_name: Optional[str] = None, tags: Dict[str, str] = None
    ) -> mlflow.ActiveRun:
        """Start a new MLflow run.

        Args:
            run_name: Name for the run
            tags: Tags to add to the run

        Returns:
            Active MLflow run
        """
        run = mlflow.start_run(run_name=run_name, tags=tags)
        logger.info(f"Started MLflow run: {run.info.run_id}")
        return run

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to MLflow.

        Args:
            params: Parameters to log
        """
        for key, value in params.items():
            mlflow.log_param(key, value)
        logger.debug(f"Logged {len(params)} parameters")

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Log metrics to MLflow.

        Args:
            metrics: Metrics to log
            step: Step number for the metrics
        """
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value, step=step)
        logger.debug(f"Logged {len(metrics)} metrics for step {step}")

    def log_model(self, model, model_name: str = "catboost_model") -> None:
        """Log model to MLflow.

        Args:
            model: Trained model to log
            model_name: Name for the model artifact
        """
        try:
            mlflow.catboost.log_model(model, model_name)
            logger.info(f"Logged model: {model_name}")
        except Exception as e:
            logger.error(f"Error logging model: {e}")

    def log_artifact(self, local_path: str, artifact_path: str = None) -> None:
        """Log artifact to MLflow.

        Args:
            local_path: Local path to the artifact
            artifact_path: Path within the artifact store
        """
        try:
            mlflow.log_artifact(local_path, artifact_path)
            logger.debug(f"Logged artifact: {local_path}")
        except Exception as e:
            logger.error(f"Error logging artifact: {e}")

    def log_training_iteration(
        self,
        iteration: int,
        train_size: int,
        train_percentage: float,
        model_params: Dict[str, Any],
        val_metrics: Dict[str, Any],
        test_metrics: Dict[str, Any],
        model=None,
    ) -> str:
        """Log a complete training iteration.

        Args:
            iteration: Iteration number
            train_size: Number of training samples
            train_percentage: Percentage of training data used
            model_params: Model parameters
            val_metrics: Validation metrics
            test_metrics: Test metrics
            model: Trained model (optional)

        Returns:
            Run ID
        """
        run_name = f"ct_iteration_{iteration:02d}"
        tags = {
            "iteration": str(iteration),
            "train_percentage": f"{train_percentage:.1f}%",
            "stage": "continuous_training",
        }

        with self.start_run(run_name=run_name, tags=tags) as run:
            params = {
                "iteration": iteration,
                "train_size": train_size,
                "train_percentage": train_percentage,
                **{f"model_{k}": v for k, v in model_params.items()},
            }
            self.log_params(params)

            val_metrics_prefixed = {
                f"val_{k}": v
                for k, v in val_metrics.items()
                if isinstance(v, (int, float))
            }
            self.log_metrics(val_metrics_prefixed, step=iteration)

            test_metrics_prefixed = {
                f"test_{k}": v
                for k, v in test_metrics.items()
                if isinstance(v, (int, float))
            }
            self.log_metrics(test_metrics_prefixed, step=iteration)

            if model is not None:
                self.log_model(model, f"model_iter_{iteration:02d}")

            logger.info(f"Logged training iteration {iteration} to MLflow")
            return run.info.run_id

    def end_run(self) -> None:
        """End the current MLflow run."""
        mlflow.end_run()
        logger.debug("Ended MLflow run")

    def get_experiment_info(self) -> Dict[str, Any]:
        """Get information about the current experiment.

        Returns:
            Dictionary with experiment information
        """
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment:
                return {
                    "name": experiment.name,
                    "experiment_id": experiment.experiment_id,
                    "artifact_location": experiment.artifact_location,
                    "lifecycle_stage": experiment.lifecycle_stage,
                }
            else:
                return {"status": "not_found"}
        except Exception as e:
            logger.error(f"Error getting experiment info: {e}")
            return {"status": "error", "message": str(e)}
