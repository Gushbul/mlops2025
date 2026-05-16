#!/bin/bash

BATCH_NUMBER=$1

if [ -z "$BATCH_NUMBER" ]; then
    echo "Usage: ./run_prefect_batch.sh <batch_number>"
    echo "Example: ./run_prefect_batch.sh 1"
    exit 1
fi

echo "🚀 Запуск Prefect пайплайна для батча $BATCH_NUMBER..."

# Проверяем доступность MLflow
echo "📊 Проверка MLflow сервера..."
if curl -s http://localhost:5000/health > /dev/null 2>&1; then
    echo "✅ MLflow сервер доступен на http://localhost:5000"
else
    echo "❌ MLflow сервер недоступен на http://localhost:5000"
    echo "   Запустите в отдельном терминале: poetry run mlflow ui --host 0.0.0.0 --port 5000"
fi

# Проверяем доступность Prefect (опционально)
echo "🔧 Проверка Prefect сервера..."
if curl -s http://localhost:4200 > /dev/null 2>&1; then
    echo "✅ Prefect UI доступен на http://localhost:4200"
else
    echo "ℹ️  Prefect сервер не запущен - будет использоваться временный сервер"
fi

echo ""
echo "🤖 Запуск ML пайплайна через Prefect..."

# Запуск пайплайна
poetry run python flows/training_flow.py $BATCH_NUMBER

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Prefect пайплайн завершен успешно для батча $BATCH_NUMBER"
    echo ""
    echo "🔍 Результаты:"

    # Проверяем созданные файлы
    if [ -f "models/model_v$BATCH_NUMBER.pkl" ]; then
        model_size=$(stat -f%z "models/model_v$BATCH_NUMBER.pkl" 2>/dev/null || stat -c%s "models/model_v$BATCH_NUMBER.pkl" 2>/dev/null)
        echo "   - ✅ Модель: models/model_v$BATCH_NUMBER.pkl ($model_size байт)"
    else
        echo "   - ❌ Модель: models/model_v$BATCH_NUMBER.pkl (не найдена)"
    fi

    if [ -f "metrics/test_metrics_v$BATCH_NUMBER.json" ]; then
        echo "   - ✅ Тестовые метрики: metrics/test_metrics_v$BATCH_NUMBER.json"
        if command -v jq > /dev/null 2>&1; then
            accuracy=$(jq -r '.test_accuracy' "metrics/test_metrics_v$BATCH_NUMBER.json" 2>/dev/null)
            f1_score=$(jq -r '.test_f1_score' "metrics/test_metrics_v$BATCH_NUMBER.json" 2>/dev/null)
            if [ "$accuracy" != "null" ] && [ "$f1_score" != "null" ]; then
                echo "     📈 Точность: $accuracy, F1-score: $f1_score"
            fi
        fi
    else
        echo "   - ❌ Тестовые метрики: metrics/test_metrics_v$BATCH_NUMBER.json (не найдены)"
    fi

    if [ -f "data/processed/dataset_processed_v$BATCH_NUMBER.csv" ]; then
        data_lines=$(wc -l < "data/processed/dataset_processed_v$BATCH_NUMBER.csv")
        echo "   - ✅ Обработанные данные: data/processed/dataset_processed_v$BATCH_NUMBER.csv ($((data_lines-1)) записей)"
    else
        echo "   - ❌ Обработанные данные: data/processed/dataset_processed_v$BATCH_NUMBER.csv (не найдены)"
    fi

    echo ""
    echo "🌐 Интерфейсы:"
    echo "   - MLflow UI: http://localhost:5000"
    echo "   - Prefect UI: http://localhost:4200 (если запущен сервер)"

else
    echo ""
    echo "❌ Ошибка выполнения пайплайна для батча $BATCH_NUMBER"
    echo "   Проверьте логи выше и убедитесь что MLflow сервер запущен"
    exit 1
fi
