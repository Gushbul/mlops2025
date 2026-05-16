.PHONY: install start-ollama pull start-pose start-all stop logs-ollama logs-pose test test-api notebook clean help

help:
	@echo "Available commands:"
	@echo "  install        - Install dependencies using Poetry"
	@echo "  start-ollama   - Start Ollama service"
	@echo "  pull           - Pull qwen2.5:1.5b model (~1GB GGUF)"
	@echo "  start-pose     - Start Pose Visualization service"
	@echo "  start-all      - Start all services (Ollama + Pose API)"
	@echo "  stop           - Stop all services"
	@echo "  logs-ollama    - Show Ollama service logs"
	@echo "  logs-pose      - Show Pose API service logs"
	@echo "  test           - Test the agent with function calling"
	@echo "  test-api       - Test Pose API directly (without LLM)"
	@echo "  notebook       - Start Jupyter notebook"
	@echo "  clean          - Stop services and clean up"

install:
	@echo "Installing dependencies with Poetry..."
	poetry install
	@echo "Dependencies installed!"

start-ollama:
	@echo "Starting Ollama service..."
	docker-compose up -d ollama
	@sleep 5
	@echo "Ollama ready. Now run: make pull"

pull:
	@echo "Pulling qwen2.5:1.5b model (~1GB GGUF)..."
	docker exec ollama_server ollama pull qwen2.5:1.5b
	@echo "Model ready!"

start-pose:
	@echo "Building and starting Pose Visualization service..."
	docker-compose up -d --build pose_api
	@echo "Pose API ready at http://localhost:8001"

start-all:
	@echo "Starting all services..."
	docker-compose up -d --build
	@sleep 5
	@echo "Services ready. Run: make pull"

stop:
	@echo "Stopping all services..."
	docker-compose down
	@echo "Stopped!"

logs-ollama:
	docker-compose logs -f ollama

logs-pose:
	docker-compose logs -f pose_api

test:
	@echo "Testing Pose Agent with function calling..."
	poetry run python test_agent.py

test-api:
	@echo "Testing Pose API..."
	poetry run python test_pose_api.py

notebook:
	@echo "Starting Jupyter notebook..."
	poetry run jupyter notebook notebooks/

clean: stop
	@echo "Cleaning up..."
	docker-compose down -v
	@echo "Cleanup completed!"

restart: stop start-all
	@echo "Services restarted!"
