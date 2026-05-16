.PHONY: help install setup start-services stop-services prefect-server mlflow-ui run-baseline run-al run-al-entropy run-al-margin run-al-confident run-both benchmark clean

help:  ## Show help message
	@echo "Available commands:"
	@echo "  install         - Install dependencies using Poetry"
	@echo "  setup           - Setup project directories"
	@echo "  start-services  - Start both Prefect and MLflow services"
	@echo "  prefect-server  - Start Prefect server (run in separate terminal)"
	@echo "  mlflow-ui       - Start MLflow UI (run in separate terminal)"
	@echo "  run-baseline    - Run baseline training on full dataset"
	@echo "  run-al          - Run Active Learning pipeline (entropy strategy)"
	@echo "  run-al-entropy  - Run AL with entropy uncertainty sampling"
	@echo "  run-al-margin   - Run AL with margin uncertainty sampling"
	@echo "  run-al-confident- Run AL with least confident uncertainty sampling"
	@echo "  run-both        - Run both baseline and AL (entropy) pipelines"
	@echo "  benchmark       - Run comprehensive benchmark of all AL strategies"
	@echo "  clean           - Clean all generated files and artifacts"
	@echo "  stop-services   - Stop all running services"
	@echo ""
	@echo "Example usage:"
	@echo "  make install && make setup"
	@echo "  make start-services      # Start both services"
	@echo "  make run-both           # Run both pipelines"

install:  ## Install dependencies
	@echo "Installing dependencies with Poetry..."
	poetry install
	@echo "Dependencies installed successfully!"

setup:  ## Setup project directories
	@echo "Setting up project directories..."
	mkdir -p models metrics mlruns
	@echo "Project setup completed!"

start-services:  ## Start Prefect and MLflow services
	@echo "Starting Prefect server in background..."
	poetry run prefect server start --host 0.0.0.0 > prefect.log 2>&1 &
	@echo "Starting MLflow UI in background..."
	poetry run mlflow ui --host 0.0.0.0 --port 5000 > mlflow.log 2>&1 &
	@echo "Waiting for services to start..."
	sleep 10
	@echo "Services started!"
	@echo "  Prefect UI: http://localhost:4200"
	@echo "  MLflow UI: http://localhost:5000"

prefect-server:  ## Start Prefect server
	@echo "Starting Prefect server..."
	poetry run prefect server start --host 0.0.0.0

mlflow-ui:  ## Start MLflow UI
	@echo "Starting MLflow UI..."
	poetry run mlflow ui --host 0.0.0.0 --port 5000

run-baseline:  ## Run baseline training pipeline
	@echo "Running baseline training pipeline..."
	poetry run python -c "from flows.baseline_flow import baseline_pipeline; baseline_pipeline()"

run-al:  ## Run Active Learning pipeline (entropy)
	@echo "Running Active Learning pipeline with entropy sampling..."
	poetry run python -c "from flows.active_learning_flow import active_learning_pipeline; active_learning_pipeline(sampling_strategy='entropy')"

run-al-entropy:  ## Run AL with entropy sampling
	@echo "Running Active Learning with entropy sampling..."
	poetry run python -c "from flows.active_learning_flow import active_learning_pipeline; active_learning_pipeline(sampling_strategy='entropy')"

run-al-margin:  ## Run AL with margin sampling
	@echo "Running Active Learning with margin sampling..."
	poetry run python -c "from flows.active_learning_flow import active_learning_pipeline; active_learning_pipeline(sampling_strategy='margin')"

run-al-confident:  ## Run AL with least confident sampling
	@echo "Running Active Learning with least confident sampling..."
	poetry run python -c "from flows.active_learning_flow import active_learning_pipeline; active_learning_pipeline(sampling_strategy='least_confident')"

run-both:  ## Run both baseline and AL pipelines
	@echo "Running both baseline and Active Learning pipelines..."
	@echo "=== Running Baseline Pipeline ==="
	poetry run python -c "from flows.baseline_flow import baseline_pipeline; baseline_pipeline()"
	@echo ""
	@echo "=== Running Active Learning Pipeline ==="
	poetry run python -c "from flows.active_learning_flow import active_learning_pipeline; active_learning_pipeline(sampling_strategy='entropy')"

benchmark:  ## Run comprehensive benchmark
	@echo "Running comprehensive benchmark of all methods..."
	@echo "=== Baseline ==="
	poetry run python -c "from flows.baseline_flow import baseline_pipeline; baseline_pipeline()"
	@echo ""
	@echo "=== Active Learning - Entropy ==="
	poetry run python -c "from flows.active_learning_flow import active_learning_pipeline; active_learning_pipeline(sampling_strategy='entropy')"
	@echo ""
	@echo "=== Active Learning - Margin ==="
	poetry run python -c "from flows.active_learning_flow import active_learning_pipeline; active_learning_pipeline(sampling_strategy='margin')"
	@echo ""
	@echo "=== Active Learning - Least Confident ==="
	poetry run python -c "from flows.active_learning_flow import active_learning_pipeline; active_learning_pipeline(sampling_strategy='least_confident')"

clean:  ## Clean generated files
	@echo "Cleaning generated files..."
	rm -rf models/
	rm -rf metrics/
	rm -rf mlruns/
	rm -rf catboost_info/
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	rm -rf flows/__pycache__/
	@echo "Cleanup completed!"

stop-services:  ## Stop services
	@echo "Stopping services..."
	pkill -f "prefect server" || true
	pkill -f "mlflow ui" || true
	@echo "Services stopped!"
