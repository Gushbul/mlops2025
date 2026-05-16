.PHONY: help install setup start-services stop-services deploy start-agent clean

help:  ## Показать справку
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Установить зависимости
	poetry install
	poetry run pre-commit install

setup:  ## Настроить проект
	poetry run dvc init --no-scm || echo "DVC уже инициализирован"
	mkdir -p data/raw data/processed models metrics

start-services:  ## Запустить Prefect server и MLflow
	@echo "Запуск Prefect server..."
	poetry run prefect server start --host 0.0.0.0 &
	@echo "Ожидание запуска Prefect server..."
	sleep 10
	@echo "Запуск MLflow UI..."
	poetry run mlflow ui --host 0.0.0.0 --port 5000 &
	@echo "Сервисы запущены:"
	@echo "  - Prefect UI: http://localhost:4200"
	@echo "  - MLflow UI: http://localhost:5000"

stop-services:  ## Остановить сервисы
	@echo "Остановка сервисов..."
	-pkill -f "prefect server"
	-pkill -f "mlflow ui"

deploy:  ## Создать деплойменты
	poetry run python deployments/setup_deployment.py

start-agent:  ## Запустить Prefect agent
	poetry run prefect agent start -q default

run-manual:  ## Запустить пайплайн вручную для батча (make run-manual BATCH=1)
	poetry run python flows/automated_training_flow.py $(BATCH)

clean:  ## Очистить сгенерированные файлы
	rm -rf data/processed/* models/* metrics/*
	rm -f batch_state.json
	rm -rf mlruns/ .dvc/

demo:  ## Демо: запустить весь процесс
	make setup
	make start-services
	sleep 15
	make deploy
	@echo "Запустите в отдельном терминале: make start-agent"
	@echo "Автоматический пайплайн запустится через 2 минуты"
