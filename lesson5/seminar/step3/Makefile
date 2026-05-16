.PHONY: help install setup prefect-server mlflow-ui run-al run-al-entropy run-al-margin run-al-confident run-single clean stop-services

# Default target
help:  ## Show help message
	@echo "Available commands:"
	@echo "  install         - Install dependencies using Poetry"
	@echo "  setup           - Setup project directories"
	@echo "  prefect-server  - Start Prefect server (run in separate terminal)"
	@echo "  mlflow-ui       - Start MLflow UI (run in separate terminal)"
	@echo "  run-al          - Run Active Learning pipeline (entropy strategy)"
	@echo "  run-al-entropy  - Run AL with entropy uncertainty sampling"
	@echo "  run-al-margin   - Run AL with margin uncertainty sampling"
	@echo "  run-al-confident- Run AL with least confident uncertainty sampling"
	@echo "  run-single      - Run single AL iteration (ITER=1 PERCENT=0.1)"
	@echo "  clean           - Clean all generated files and artifacts"
	@echo "  stop-services   - Stop all running services"
	@echo ""
	@echo "Example usage:"
	@echo "  make install && make setup"
	@echo "  make prefect-server  # in terminal 1"
	@echo "  make mlflow-ui       # in terminal 2"
	@echo "  make run-al          # in terminal 3"

# Install dependencies
install:  ## Install dependencies
	@echo "Installing dependencies with Poetry..."
	poetry install
	@echo "Dependencies installed successfully!"

# Setup project directories
setup:  ## Setup project directories
	@echo "Setting up project directories..."
	mkdir -p models metrics mlruns
	@echo "Project setup completed!"

# Start Prefect server
prefect-server:  ## Start Prefect server
	@echo "Starting Prefect server..."
	@echo "Prefect UI will be available at: http://localhost:4200"
	poetry run prefect server start --host 0.0.0.0

# Start MLflow UI
mlflow-ui:  ## Start MLflow UI
	@echo "Starting MLflow UI..."
	@echo "MLflow UI will be available at: http://localhost:5000"
	poetry run mlflow ui --host 0.0.0.0 --port 5000

# Run Active Learning pipeline with entropy strategy (default)
run-al:  ## Run Active Learning pipeline
	@echo "Running Active Learning pipeline with entropy strategy..."
	poetry run python flows/active_learning_flow.py

# Run Active Learning with entropy uncertainty sampling
run-al-entropy:  ## Run AL with entropy strategy
	@echo "Running Active Learning with entropy uncertainty sampling..."
	poetry run python flows/active_learning_flow.py 10 entropy

# Run Active Learning with margin uncertainty sampling
run-al-margin:  ## Run AL with margin strategy
	@echo "Running Active Learning with margin uncertainty sampling..."
	poetry run python flows/active_learning_flow.py 10 margin

# Run Active Learning with least confident uncertainty sampling
run-al-confident:  ## Run AL with least confident strategy
	@echo "Running Active Learning with least confident uncertainty sampling..."
	poetry run python flows/active_learning_flow.py 10 least_confident

# Run single iteration
run-single:  ## Run single AL iteration
	@echo "Running single AL iteration..."
	@echo "Iteration: $(or $(ITER),1), Initial percentage: $(or $(PERCENT),0.1)"
	poetry run python flows/active_learning_flow.py single $(or $(ITER),1) $(or $(PERCENT),0.1) entropy

# Run with custom number of iterations
run-custom:  ## Run AL with custom iterations (ITERS=5 STRATEGY=entropy)
	@echo "Running AL with $(or $(ITERS),5) iterations and $(or $(STRATEGY),entropy) strategy..."
	poetry run python flows/active_learning_flow.py $(or $(ITERS),5) $(or $(STRATEGY),entropy)

# Compare different strategies
run-comparison:  ## Run comparison of all uncertainty strategies
	@echo "Running comparison of uncertainty strategies..."
	@echo "This will run 3 experiments with different strategies"
	@echo "1. Running with entropy strategy..."
	poetry run python flows/active_learning_flow.py 5 entropy
	@echo "2. Running with margin strategy..."
	poetry run python flows/active_learning_flow.py 5 margin
	@echo "3. Running with least_confident strategy..."
	poetry run python flows/active_learning_flow.py 5 least_confident
	@echo "Comparison completed! Check MLflow UI for results."

# Clean all generated files
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

# Stop all services
stop-services:  ## Stop all services
	@echo "Stopping services..."
	-pkill -f "prefect server"
	-pkill -f "mlflow ui"
	@echo "Services stopped!"

# Full demo setup
demo-setup:  ## Setup and start services for demo
	@echo "Setting up Active Learning demo environment..."
	make install
	make setup
	@echo ""
	@echo "Demo setup completed!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Run 'make prefect-server' in terminal 1"
	@echo "2. Run 'make mlflow-ui' in terminal 2"
	@echo "3. Run 'make run-al' in terminal 3"
	@echo ""
	@echo "Or run everything in background:"
	@echo "  make start-services && sleep 10 && make run-al"

# Start services in background (for testing)
start-services:  ## Start services in background
	@echo "Starting services in background..."
	poetry run prefect server start --host 0.0.0.0 > prefect.log 2>&1 &
	poetry run mlflow ui --host 0.0.0.0 --port 5000 > mlflow.log 2>&1 &
	@echo "Services started in background"
	@echo "  - Prefect UI: http://localhost:4200"
	@echo "  - MLflow UI: http://localhost:5000"
	@echo "  - Logs: prefect.log, mlflow.log"

# Test single iteration quickly
test:  ## Quick test with single iteration
	@echo "Running quick Active Learning test..."
	make clean
	make setup
	poetry run python flows/active_learning_flow.py single 1 0.1 entropy
	@echo "Test completed!"

# Benchmark different strategies
benchmark:  ## Benchmark all uncertainty strategies (short runs)
	@echo "Benchmarking uncertainty strategies with 3 iterations each..."
	make clean && make setup
	@echo "1. Entropy strategy..."
	poetry run python flows/active_learning_flow.py 3 entropy
	make clean && make setup
	@echo "2. Margin strategy..."
	poetry run python flows/active_learning_flow.py 3 margin
	make clean && make setup
	@echo "3. Least confident strategy..."
	poetry run python flows/active_learning_flow.py 3 least_confident
	@echo "Benchmark completed! Check MLflow UI to compare results."
