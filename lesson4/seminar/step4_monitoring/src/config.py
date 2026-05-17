from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ServiceConfig:
    host: str = "localhost"
    port: int = 8000
    base_url: str = "http://localhost:8000"
    auto_start: bool = False
    startup_timeout_seconds: int = 60
    command: str = "poetry run python main.py"
    cwd: str = "../step2_fastapi_inference"


@dataclass
class MonitoringConfig:
    check_interval_seconds: int = 30
    samples_per_check: int = 3
    request_timeout_seconds: int = 10


@dataclass
class Threshold:
    warning: float
    critical: float


@dataclass
class ThresholdsConfig:
    response_time_ms: Threshold = field(default_factory=lambda: Threshold(2000, 5000))
    p95_latency_ms: Threshold = field(default_factory=lambda: Threshold(3000, 6000))
    error_rate_percent: Threshold = field(default_factory=lambda: Threshold(10, 25))
    consecutive_failures: Threshold = field(default_factory=lambda: Threshold(3, 5))


@dataclass
class AlertsConfig:
    enabled: bool = True
    cooldown_minutes: int = 5


@dataclass
class LoggingConfig:
    console_colors: bool = True
    log_file: Path = Path("logs/monitoring.log")
    metrics_file: Path = Path("logs/metrics.jsonl")


@dataclass
class InferenceConfig:
    image_paths: list[Path] = field(
        default_factory=lambda: [Path("../step2_fastapi_inference/test_images/img.jpg")]
    )
    file_field: str = "file"


@dataclass
class AppConfig:
    service: ServiceConfig = field(default_factory=ServiceConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    alerts: AlertsConfig = field(default_factory=AlertsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    project_root: Path = Path(".")


def _read_threshold(data: dict[str, Any], name: str, default: Threshold) -> Threshold:
    raw = data.get(name, {})
    return Threshold(
        warning=float(raw.get("warning", default.warning)),
        critical=float(raw.get("critical", default.critical)),
    )


def _resolve_path(path: str | Path, base_dir: Path) -> Path:
    result = Path(path)
    if result.is_absolute():
        return result
    return (base_dir / result).resolve()


def load_config(config_path: str | Path) -> AppConfig:
    """Load monitoring configuration from YAML and resolve relative paths."""
    config_path = Path(config_path).resolve()
    base_dir = config_path.parent.parent

    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    defaults = AppConfig(project_root=base_dir)

    service_raw = raw.get("service", {})
    service = ServiceConfig(
        host=service_raw.get("host", defaults.service.host),
        port=int(service_raw.get("port", defaults.service.port)),
        base_url=service_raw.get("base_url", defaults.service.base_url).rstrip("/"),
        auto_start=bool(service_raw.get("auto_start", defaults.service.auto_start)),
        startup_timeout_seconds=int(
            service_raw.get(
                "startup_timeout_seconds", defaults.service.startup_timeout_seconds
            )
        ),
        command=service_raw.get("command", defaults.service.command),
        cwd=str(_resolve_path(service_raw.get("cwd", defaults.service.cwd), base_dir)),
    )

    monitoring_raw = raw.get("monitoring", {})
    monitoring = MonitoringConfig(
        check_interval_seconds=int(
            monitoring_raw.get(
                "check_interval_seconds", defaults.monitoring.check_interval_seconds
            )
        ),
        samples_per_check=int(
            monitoring_raw.get(
                "samples_per_check", defaults.monitoring.samples_per_check
            )
        ),
        request_timeout_seconds=int(
            monitoring_raw.get(
                "request_timeout_seconds", defaults.monitoring.request_timeout_seconds
            )
        ),
    )

    thresholds_raw = raw.get("thresholds", {})
    thresholds = ThresholdsConfig(
        response_time_ms=_read_threshold(
            thresholds_raw,
            "response_time_ms",
            defaults.thresholds.response_time_ms,
        ),
        p95_latency_ms=_read_threshold(
            thresholds_raw,
            "p95_latency_ms",
            defaults.thresholds.p95_latency_ms,
        ),
        error_rate_percent=_read_threshold(
            thresholds_raw,
            "error_rate_percent",
            defaults.thresholds.error_rate_percent,
        ),
        consecutive_failures=_read_threshold(
            thresholds_raw,
            "consecutive_failures",
            defaults.thresholds.consecutive_failures,
        ),
    )

    alerts_raw = raw.get("alerts", {})
    alerts = AlertsConfig(
        enabled=bool(alerts_raw.get("enabled", defaults.alerts.enabled)),
        cooldown_minutes=int(
            alerts_raw.get("cooldown_minutes", defaults.alerts.cooldown_minutes)
        ),
    )

    logging_raw = raw.get("logging", {})
    logging_config = LoggingConfig(
        console_colors=bool(
            logging_raw.get("console_colors", defaults.logging.console_colors)
        ),
        log_file=_resolve_path(
            logging_raw.get("log_file", "logs/monitoring.log"), base_dir
        ),
        metrics_file=_resolve_path(
            logging_raw.get("metrics_file", "logs/metrics.jsonl"), base_dir
        ),
    )

    inference_raw = raw.get("inference", {})
    image_paths = inference_raw.get("image_paths", None)
    if image_paths is None:
        image_paths = [str(path) for path in defaults.inference.image_paths]
    inference = InferenceConfig(
        image_paths=[_resolve_path(path, base_dir) for path in image_paths],
        file_field=inference_raw.get("file_field", defaults.inference.file_field),
    )

    return AppConfig(
        service=service,
        monitoring=monitoring,
        thresholds=thresholds,
        alerts=alerts,
        logging=logging_config,
        inference=inference,
        project_root=base_dir,
    )
