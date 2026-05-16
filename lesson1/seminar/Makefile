.PHONY: install lint test train mlflow-ui mlflow-ui-docker docker-build-dev docker-build-prod docker-build docker-run docker-run-dev docker-test docker-shell docker-clean compose-check compose-up compose-up-training compose-up-dev compose-up-mlflow compose-up-jupyter compose-down compose-logs compose-clean docker-mlflow-ui docker-jupyter

install:
	@echo "Installing dependencies with Poetry..."
	poetry install

lint:
	@echo "Running linting..."
	poetry run ruff check .
	poetry run black --check .

test:
	@echo "Running tests..."
	poetry run pytest -q

train:
	@echo "Running training..."
	poetry run python -m src.app.train

mlflow-ui:
	@echo "Starting MLflow UI (local experiments)..."
	poetry run mlflow ui --backend-store-uri ./mlruns

mlflow-ui-docker:
	@echo "Starting MLflow UI (Docker experiments)..."
	poetry run mlflow ui --backend-store-uri ./mlruns_docker

docker-build-dev:
	@echo "Building Docker development image..."
	docker build --target development -t lesson1-mlops:dev .

docker-build-prod:
	@echo "Building Docker production image..."
	docker build --target production -t lesson1-mlops:0.1 .

docker-build: docker-build-prod

docker-run:
	@echo "Running Docker container..."
	docker run --rm -v "$(PWD)/mlruns_docker:/app/mlruns" lesson1-mlops:0.1

docker-run-dev:
	@echo "Running Docker development container..."
	docker run --rm -it -v "$(PWD)/mlruns_docker:/app/mlruns" -v "$(PWD)/src:/app/src" lesson1-mlops:dev

docker-test:
	@echo "Running tests in Docker..."
	docker run --rm lesson1-mlops:dev poetry run pytest

docker-shell:
	@echo "Opening shell in Docker container..."
	docker run --rm -it --entrypoint /bin/bash lesson1-mlops:dev

docker-clean:
	@echo "Cleaning Docker images..."
	docker rmi lesson1-mlops:dev lesson1-mlops:0.1 || true
	docker system prune -f

# Docker Compose commands (requires docker-compose installation)
compose-check:
	@command -v docker-compose >/dev/null 2>&1 || { echo "❌ docker-compose not found. Install with: brew install docker-compose"; exit 1; }

compose-up: compose-check
	@echo "Starting services with docker-compose..."
	docker-compose up --build

compose-up-training: compose-check
	@echo "Running ML training with docker-compose..."
	docker-compose up --build ml-training

compose-up-dev: compose-check
	@echo "Starting development environment..."
	docker-compose up --build ml-dev

compose-up-mlflow: compose-check
	@echo "Starting MLflow UI..."
	docker-compose up --build mlflow-ui

compose-up-jupyter: compose-check
	@echo "Starting Jupyter Lab..."
	docker-compose up --build jupyter

compose-down: compose-check
	@echo "Stopping docker-compose services..."
	docker-compose down

compose-logs: compose-check
	@echo "Showing docker-compose logs..."
	docker-compose logs -f

compose-clean: compose-check
	@echo "Cleaning docker-compose..."
	docker-compose down --volumes --remove-orphans
	docker-compose rm -f

# Alternative Docker commands without compose
docker-mlflow-ui:
	@echo "Starting MLflow UI in Docker..."
	docker run --rm -p 5000:5000 -v "$(PWD)/mlruns_docker:/app/mlruns" lesson1-mlops:dev mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000

docker-jupyter:
	@echo "Starting Jupyter Lab in Docker..."
	docker run --rm -p 8888:8888 -v "$(PWD)/mlruns_docker:/app/mlruns" -v "$(PWD)/src:/app/src" -v "$(PWD)/notebooks:/app/notebooks" lesson1-mlops:dev jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
