from __future__ import annotations

import argparse

from src.config import load_config
from src.logger import setup_logger
from src.monitor import ServiceMonitor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitoring for the ONNX FastAPI inference service"
    )
    parser.add_argument(
        "--config",
        default="config/monitoring_config.yaml",
        help="Path to monitoring YAML config",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run exactly one monitoring check and exit",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Run a fixed number of checks and exit",
    )
    parser.add_argument(
        "--start-service",
        action="store_true",
        help="Start step2 FastAPI service before monitoring",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logger(
        log_file=config.logging.log_file,
        console_colors=config.logging.console_colors,
    )

    iterations = 1 if args.once else args.iterations
    monitor = ServiceMonitor(config=config, logger=logger)
    monitor.start_service_if_needed(force=args.start_service)
    monitor.run(iterations=iterations)


if __name__ == "__main__":
    main()
