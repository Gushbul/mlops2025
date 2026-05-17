from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JsonFormatter(logging.Formatter):
    """Format log records as JSON lines."""

    RESERVED = {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
        "asctime",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key not in self.RESERVED and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


class ColorConsoleFormatter(logging.Formatter):
    """Human-readable colored console formatter."""

    COLORS = {
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "reset": "\033[0m",
        "grey": "\033[90m",
    }

    def __init__(self, use_colors: bool = True) -> None:
        super().__init__(
            "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
        )
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        if not self.use_colors:
            return message

        alert_level = getattr(record, "alert_level", None)
        color = {
            "normal": "green",
            "warning": "yellow",
            "critical": "red",
        }.get(alert_level)

        if color is None:
            if record.levelno >= logging.ERROR:
                color = "red"
            elif record.levelno >= logging.WARNING:
                color = "yellow"

        if color is None:
            return message

        return f"{self.COLORS[color]}{message}{self.COLORS['reset']}"


def setup_logger(
    log_file: Path,
    console_colors: bool = True,
    logger_name: str = "service_monitor",
) -> logging.Logger:
    """Create logger with JSON file logs and colored console output."""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(ColorConsoleFormatter(use_colors=console_colors))

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(JsonFormatter())

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def write_metrics(metrics_file: Path, metrics: dict[str, Any]) -> None:
    """Append a metrics record to JSONL file."""
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **metrics,
    }
    with metrics_file.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
