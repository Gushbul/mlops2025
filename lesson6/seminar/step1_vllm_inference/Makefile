.PHONY: install start pull stop logs test notebook clean help

help:
	@echo "Available commands:"
	@echo "  install   - Install dependencies using Poetry"
	@echo "  start     - Start Ollama service"
	@echo "  pull      - Pull qwen2.5:1.5b model (GGUF, ~1GB)"
	@echo "  stop      - Stop Ollama service"
	@echo "  logs      - Show Ollama service logs"
	@echo "  test      - Test the client"
	@echo "  notebook  - Start Jupyter notebook"
	@echo "  clean     - Stop services and clean up"

install:
	@echo "Installing dependencies with Poetry..."
	poetry install
	@echo "Dependencies installed!"

start:
	@echo "Starting Ollama service..."
	docker-compose up -d
	@sleep 5
	@echo "Ollama ready at http://localhost:11434"
	@echo "Now run: make pull"

pull:
	@echo "Pulling qwen2.5:1.5b model (~1GB GGUF)..."
	docker exec ollama_server ollama pull qwen2.5:1.5b
	@echo "Model ready!"

stop:
	@echo "Stopping Ollama service..."
	docker-compose down
	@echo "Stopped!"

logs:
	docker-compose logs -f ollama

test:
	@echo "Testing Ollama client..."
	poetry run python test_scripts/test_client.py

notebook:
	@echo "Starting Jupyter notebook..."
	poetry run jupyter notebook notebooks/

clean: stop
	@echo "Cleaning up..."
	docker-compose down -v
	@echo "Cleanup completed!"

restart: stop start
	@echo "Service restarted!"
