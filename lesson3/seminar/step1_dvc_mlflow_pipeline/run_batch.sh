#!/bin/bash

BATCH_NUMBER=$1

if [ -z "$BATCH_NUMBER" ]; then
    echo "Usage: ./run_batch.sh <batch_number>"
    echo "Example: ./run_batch.sh 1"
    exit 1
fi

echo "🚀 Запуск пайплайна для батча $BATCH_NUMBER..."

# Проверяем доступность MLflow
echo "📊 Проверка MLflow сервера..."
if curl -s http://localhost:5000/health > /dev/null 2>&1; then
    echo "✅ MLflow сервер доступен"
else
    echo "❌ MLflow сервер недоступен на http://localhost:5000"
    echo "   Запустите в отдельном терминале: poetry run mlflow ui --host 0.0.0.0 --port 5000"
fi

echo ""
echo "📦 Подготовка батча $BATCH_NUMBER..."
poetry run python src/prepare_batch.py $BATCH_NUMBER

echo ""
echo "🔄 Объединение данных..."
poetry run python src/merge_data.py $BATCH_NUMBER

echo ""
echo "⚙️  Предобработка данных..."
poetry run python src/preprocess.py $BATCH_NUMBER

echo ""
echo "🤖 Обучение модели..."
poetry run python src/train.py $BATCH_NUMBER

echo ""
echo "📈 Оценка модели..."
poetry run python src/evaluate.py $BATCH_NUMBER

echo ""
echo "✅ Пайплайн завершен для батча $BATCH_NUMBER"
echo "🔍 Результаты:"
echo "   - Модель: models/model_v$BATCH_NUMBER.pkl"
echo "   - Метрики обучения: metrics/metrics_v$BATCH_NUMBER.json"
echo "   - Метрики тестирования: metrics/test_metrics_v$BATCH_NUMBER.json"
echo "   - Данные: data/processed/dataset_processed_v$BATCH_NUMBER.csv"
