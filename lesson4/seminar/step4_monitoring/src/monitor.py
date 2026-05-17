from __future__ import annotations

import math
import mimetypes
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .config import AppConfig, Threshold
from .logger import write_metrics


@dataclass
class RequestResult:
    endpoint: str
    ok: bool
    status_code: int | None
    elapsed_ms: float
    payload: dict[str, Any] | None = None
    error: str | None = None


class ServiceMonitor:
    """Monitor /health and /predict endpoints of the FastAPI inference service."""

    def __init__(self, config: AppConfig, logger) -> None:
        self.config = config
        self.logger = logger
        self.consecutive_failures = 0
        self.last_alert_at: dict[str, float] = {}
        self.service_process: subprocess.Popen | None = None
        self._image_cursor = 0

    def start_service_if_needed(self, force: bool = False) -> None:
        """Optionally start FastAPI service from step2 before monitoring."""
        should_start = force or self.config.service.auto_start
        if not should_start:
            return

        if self._service_is_ready():
            self.logger.info("FastAPI service is already available")
            return

        cwd = Path(self.config.service.cwd)
        if not cwd.exists():
            raise FileNotFoundError(f"Service cwd does not exist: {cwd}")

        self.logger.info(
            "Starting FastAPI service",
            extra={
                "event": "service_start",
                "command": self.config.service.command,
                "cwd": str(cwd),
            },
        )

        self.service_process = subprocess.Popen(
            shlex.split(self.config.service.command),
            cwd=cwd,
        )
        self._wait_for_service()

    def stop_service(self) -> None:
        """Terminate a service process started by this monitor."""
        if self.service_process is None:
            return
        if self.service_process.poll() is None:
            self.logger.info(
                "Stopping FastAPI service", extra={"event": "service_stop"}
            )
            self.service_process.terminate()
            try:
                self.service_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.service_process.kill()
        self.service_process = None

    def run(self, iterations: int | None = None) -> None:
        """Run monitoring loop. Use iterations=1 for one-shot checks."""
        check_number = 0
        try:
            while iterations is None or check_number < iterations:
                check_number += 1
                metrics = self.run_check(check_number=check_number)
                write_metrics(self.config.logging.metrics_file, metrics)

                if iterations is not None and check_number >= iterations:
                    break

                time.sleep(self.config.monitoring.check_interval_seconds)
        finally:
            self.stop_service()

    def run_check(self, check_number: int = 1) -> dict[str, Any]:
        """Run a single health + prediction monitoring check."""
        request_results: list[RequestResult] = []

        health_result = self._call_health()
        request_results.append(health_result)
        self._update_consecutive_failures(health_result.ok)

        for _ in range(self.config.monitoring.samples_per_check):
            prediction_result = self._call_predict()
            request_results.append(prediction_result)
            self._update_consecutive_failures(prediction_result.ok)

        metrics = self._build_metrics(check_number, request_results, health_result)
        alert_level, reasons = self._evaluate_alert_level(metrics)
        metrics["alert_level"] = alert_level
        metrics["alert_reasons"] = reasons

        self._log_check_summary(metrics, request_results)
        self._emit_alert_if_needed(alert_level, reasons, metrics)
        return metrics

    def _call_health(self) -> RequestResult:
        start = time.perf_counter()
        try:
            with httpx.Client(
                timeout=self.config.monitoring.request_timeout_seconds, trust_env=False
            ) as client:
                response = client.get(f"{self.config.service.base_url}/health")
            elapsed_ms = (time.perf_counter() - start) * 1000

            payload = self._safe_json(response)
            ok = response.status_code == 200 and payload.get("status") == "healthy"
            return RequestResult(
                endpoint="/health",
                ok=ok,
                status_code=response.status_code,
                elapsed_ms=elapsed_ms,
                payload=payload,
                error=None if ok else response.text,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return RequestResult(
                endpoint="/health",
                ok=False,
                status_code=None,
                elapsed_ms=elapsed_ms,
                error=str(exc),
            )

    def _call_predict(self) -> RequestResult:
        image_path = self._next_image_path()
        if image_path is None:
            return RequestResult(
                endpoint="/predict",
                ok=False,
                status_code=None,
                elapsed_ms=0,
                error="No readable image files configured for inference testing",
            )

        content_type = (
            mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
        )
        start = time.perf_counter()
        try:
            with image_path.open("rb") as image_file:
                files = {
                    self.config.inference.file_field: (
                        image_path.name,
                        image_file,
                        content_type,
                    )
                }
                with httpx.Client(
                    timeout=self.config.monitoring.request_timeout_seconds,
                    trust_env=False,
                ) as client:
                    response = client.post(
                        f"{self.config.service.base_url}/predict", files=files
                    )
            elapsed_ms = (time.perf_counter() - start) * 1000

            payload = self._safe_json(response)
            ok = self._is_valid_prediction_response(response.status_code, payload)
            self._log_prediction(
                image_path, elapsed_ms, response.status_code, payload, ok
            )

            return RequestResult(
                endpoint="/predict",
                ok=ok,
                status_code=response.status_code,
                elapsed_ms=elapsed_ms,
                payload=payload,
                error=None if ok else response.text,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.logger.error(
                "Prediction request failed",
                extra={
                    "event": "predict_error",
                    "image_path": str(image_path),
                    "elapsed_ms": round(elapsed_ms, 2),
                    "error": str(exc),
                },
            )
            return RequestResult(
                endpoint="/predict",
                ok=False,
                status_code=None,
                elapsed_ms=elapsed_ms,
                error=str(exc),
            )

    def _next_image_path(self) -> Path | None:
        readable_images = [
            path for path in self.config.inference.image_paths if path.exists()
        ]
        if not readable_images:
            return None
        image = readable_images[self._image_cursor % len(readable_images)]
        self._image_cursor += 1
        return image

    def _update_consecutive_failures(self, ok: bool) -> None:
        if ok:
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1

    def _build_metrics(
        self,
        check_number: int,
        results: list[RequestResult],
        health_result: RequestResult,
    ) -> dict[str, Any]:
        total_requests = len(results)
        failed_requests = sum(1 for result in results if not result.ok)
        response_times = [result.elapsed_ms for result in results]
        prediction_times = [
            result.elapsed_ms for result in results if result.endpoint == "/predict"
        ]

        return {
            "check_number": check_number,
            "service_url": self.config.service.base_url,
            "health_status": "healthy" if health_result.ok else "unhealthy",
            "health_status_code": health_result.status_code,
            "response_time_ms": round(max(response_times), 2) if response_times else 0,
            "avg_response_time_ms": (
                round(sum(response_times) / total_requests, 2) if total_requests else 0
            ),
            "p95_latency_ms": round(self._p95(response_times), 2),
            "predict_p95_latency_ms": round(self._p95(prediction_times), 2),
            "total_requests": total_requests,
            "failed_requests": failed_requests,
            "successful_requests": total_requests - failed_requests,
            "error_rate_percent": (
                round(failed_requests * 100 / total_requests, 2)
                if total_requests
                else 0
            ),
            "consecutive_failures": self.consecutive_failures,
            "samples_per_check": self.config.monitoring.samples_per_check,
        }

    def _evaluate_alert_level(self, metrics: dict[str, Any]) -> tuple[str, list[str]]:
        critical_reasons: list[str] = []
        warning_reasons: list[str] = []

        if metrics["health_status"] != "healthy":
            critical_reasons.append("health_status is unhealthy")

        self._collect_threshold_reason(
            "response_time_ms",
            metrics["response_time_ms"],
            self.config.thresholds.response_time_ms,
            warning_reasons,
            critical_reasons,
        )
        self._collect_threshold_reason(
            "p95_latency_ms",
            metrics["p95_latency_ms"],
            self.config.thresholds.p95_latency_ms,
            warning_reasons,
            critical_reasons,
        )
        self._collect_threshold_reason(
            "error_rate_percent",
            metrics["error_rate_percent"],
            self.config.thresholds.error_rate_percent,
            warning_reasons,
            critical_reasons,
        )
        self._collect_threshold_reason(
            "consecutive_failures",
            metrics["consecutive_failures"],
            self.config.thresholds.consecutive_failures,
            warning_reasons,
            critical_reasons,
        )

        if critical_reasons:
            return "critical", critical_reasons
        if warning_reasons:
            return "warning", warning_reasons
        return "normal", ["all metrics are within thresholds"]

    def _collect_threshold_reason(
        self,
        metric_name: str,
        value: float,
        threshold: Threshold,
        warning_reasons: list[str],
        critical_reasons: list[str],
    ) -> None:
        if value >= threshold.critical:
            critical_reasons.append(
                f"{metric_name}={value} >= critical={threshold.critical}"
            )
        elif value >= threshold.warning:
            warning_reasons.append(
                f"{metric_name}={value} >= warning={threshold.warning}"
            )

    def _log_check_summary(
        self, metrics: dict[str, Any], results: list[RequestResult]
    ) -> None:
        level = metrics["alert_level"]
        message = (
            f"[{level.upper()}] health={metrics['health_status']} | "
            f"p95={metrics['p95_latency_ms']:.2f} ms | "
            f"max_response={metrics['response_time_ms']:.2f} ms | "
            f"error_rate={metrics['error_rate_percent']:.2f}% | "
            f"consecutive_failures={metrics['consecutive_failures']}"
        )
        self.logger.info(
            message,
            extra={
                "event": "monitoring_check",
                "alert_level": level,
                "metrics": metrics,
                "requests": [result.__dict__ for result in results],
            },
        )

    def _emit_alert_if_needed(
        self, alert_level: str, reasons: list[str], metrics: dict[str, Any]
    ) -> None:
        if not self.config.alerts.enabled or alert_level == "normal":
            return

        now = time.monotonic()
        cooldown_seconds = self.config.alerts.cooldown_minutes * 60
        last_alert = self.last_alert_at.get(alert_level, 0)
        if now - last_alert < cooldown_seconds:
            return

        self.last_alert_at[alert_level] = now
        log_method = (
            self.logger.error if alert_level == "critical" else self.logger.warning
        )
        log_method(
            f"{alert_level.upper()} alert: {'; '.join(reasons)}",
            extra={
                "event": "alert",
                "alert_level": alert_level,
                "reasons": reasons,
                "metrics": metrics,
            },
        )

    def _log_prediction(
        self,
        image_path: Path,
        elapsed_ms: float,
        status_code: int,
        payload: dict[str, Any],
        ok: bool,
    ) -> None:
        result = payload.get("result") if isinstance(payload, dict) else None
        prediction = None
        model_timing = None
        if isinstance(result, dict):
            prediction = result.get("prediction")
            model_timing = result.get("timing")

        self.logger.info(
            f"Prediction processed: {image_path.name}, status={status_code}, "
            f"response_time={elapsed_ms:.2f} ms, success={ok}",
            extra={
                "event": "prediction",
                "image_path": str(image_path),
                "status_code": status_code,
                "elapsed_ms": round(elapsed_ms, 2),
                "success": ok,
                "prediction": prediction,
                "model_timing": model_timing,
            },
        )

    def _service_is_ready(self) -> bool:
        return self._call_health().ok

    def _wait_for_service(self) -> None:
        deadline = time.monotonic() + self.config.service.startup_timeout_seconds
        while time.monotonic() < deadline:
            if self._service_is_ready():
                self.logger.info("FastAPI service is ready")
                return
            time.sleep(2)
        raise TimeoutError(
            f"Service did not become healthy within "
            f"{self.config.service.startup_timeout_seconds} seconds"
        )

    @staticmethod
    def _safe_json(response: httpx.Response) -> dict[str, Any]:
        try:
            payload = response.json()
            return payload if isinstance(payload, dict) else {"payload": payload}
        except Exception:
            return {"text": response.text}

    @staticmethod
    def _is_valid_prediction_response(
        status_code: int, payload: dict[str, Any]
    ) -> bool:
        if status_code != 200:
            return False
        if payload.get("success") is not True:
            return False
        result = payload.get("result")
        if not isinstance(result, dict):
            return False
        if result.get("success") is not True:
            return False
        if "prediction" not in result:
            return False
        if "timing" not in result:
            return False
        return True

    @staticmethod
    def _p95(values: list[float]) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = max(0, math.ceil(0.95 * len(ordered)) - 1)
        return ordered[index]
