.PHONY: help install setup prefect-server mlflow-ui run-ct run-single clean stop-services

# Default target
help:  ## Show help message
	@echo "Available commands:"
	@echo "  install       - Install dependencies using Poetry"
	@echo "  setup         - Setup project directories"
	@echo "  prefect-server- Start Prefect server (run in separate terminal)"
	@echo "  mlflow-ui     - Start MLflow UI (run in separate terminal)"
	@echo "  run-ct        - Run full Continuous Training pipeline (10 iterations)"
	@echo "  run-single    - Run single CT iteration (ITER=1 PERCENT=0.1)"
	@echo "  clean         - Clean all generated files and artifacts"
	@echo "  stop-services - Stop all running services"
	@echo ""
	@echo "Example usage:"
	@echo "  make install && make setup"
	@echo "  make prefect-server  # in terminal 1"
	@echo "  make mlflow-ui       # in terminal 2"
	@echo "  make run-ct          # in terminal 3"

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

# Run full Continuous Training pipeline
run-ct:  ## Run Continuous Training pipeline
	@echo "Running Continuous Training pipeline (10 iterations)..."
	poetry run python flows/continuous_training_flow.py

# Run single iteration
run-single:  ## Run single CT iteration
	@echo "Running single CT iteration..."
	@echo "Iteration: $(or $(ITER),1), Percentage: $(or $(PERCENT),0.1)"
	poetry run python flows/continuous_training_flow.py single $(or $(ITER),1) $(or $(PERCENT),0.1)

# Run with custom number of iterations
run-custom:  ## Run CT with custom iterations (ITERS=5)
	@echo "Running CT with $(or $(ITERS),5) iterations..."
	poetry run python flows/continuous_training_flow.py $(or $(ITERS),5)

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
	@echo "Setting up demo environment..."
	make install
	make setup
	@echo ""
	@echo "Demo setup completed!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Run 'make prefect-server' in terminal 1"
	@echo "2. Run 'make mlflow-ui' in terminal 2"
	@echo "3. Run 'make run-ct' in terminal 3"
	@echo ""
	@echo "Or run everything in background:"
	@echo "  make start-services && sleep 10 && make run-ct"

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
	@echo "Running quick test..."
	make clean
	make setup
	poetry run python flows/continuous_training_flow.py single 1 0.1
	@echo "Test completed!"
