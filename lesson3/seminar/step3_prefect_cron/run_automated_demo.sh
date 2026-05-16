#!/bin/bash

echo "🚀 Запуск автоматизированной демонстрации Prefect cron пайплайна..."

# Проверяем доступность сервисов
echo ""
echo "📊 Проверка сервисов..."

# Проверка Prefect
if curl -s http://localhost:4200 > /dev/null 2>&1; then
    echo "✅ Prefect UI доступен на http://localhost:4200"
else
    echo "❌ Prefect UI недоступен на http://localhost:4200"
    echo "   Запустите: poetry run prefect server start --host 0.0.0.0"
    PREFECT_MISSING=true
fi

# Проверка MLflow
if curl -s http://localhost:5000/health > /dev/null 2>&1; then
    echo "✅ MLflow UI доступен на http://localhost:5000"
else
    echo "❌ MLflow UI недоступен на http://localhost:5000"
    echo "   Запустите: poetry run mlflow ui --host 0.0.0.0 --port 5000"
    MLFLOW_MISSING=true
fi

if [ "$PREFECT_MISSING" = true ] || [ "$MLFLOW_MISSING" = true ]; then
    echo ""
    echo "⚠️  Некоторые сервисы недоступны. Продолжить демонстрацию?"
    echo "   (Пайплайны будут работать, но без веб-интерфейса для мониторинга)"
    read -p "Продолжить? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "🤖 Демонстрация автоматизированного пайплайна..."

# Показываем текущее состояние
if [ -f "batch_state.json" ]; then
    echo ""
    echo "📋 Текущее состояние batch_state.json:"
    cat batch_state.json | head -20
    echo ""

    # Извлекаем next_batch
    NEXT_BATCH=$(python3 -c "import json; data=json.load(open('batch_state.json')); print(data.get('next_batch', 1))" 2>/dev/null || echo "1")
    echo "➡️  Следующий батч для обработки: $NEXT_BATCH"
else
    echo "📋 batch_state.json не найден - создастся автоматически"
    NEXT_BATCH=1
fi

echo ""
echo "🔄 Запуск автоматического режима пайплайна..."
echo "(Пайплайн определит следующий батч автоматически)"

# Запускаем автоматический пайплайн
poetry run python flows/automated_training_flow.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Автоматический пайплайн выполнен успешно!"

    echo ""
    echo "🔍 Результаты выполнения:"

    # Показываем обновленное состояние
    if [ -f "batch_state.json" ]; then
        NEW_NEXT_BATCH=$(python3 -c "import json; data=json.load(open('batch_state.json')); print(data.get('next_batch', 1))" 2>/dev/null || echo "?")
        echo "   📊 Обновлено состояние: следующий батч $NEW_NEXT_BATCH"

        # Показываем последний обработанный батч
        PROCESSED_BATCH=$(python3 -c "import json; data=json.load(open('batch_state.json')); batches=data.get('processed_batches', []); print(batches[-1]['batch_number'] if batches else 0)" 2>/dev/null || echo "?")
        if [ "$PROCESSED_BATCH" != "?" ] && [ "$PROCESSED_BATCH" != "0" ]; then
            echo "   🎯 Обработан батч: $PROCESSED_BATCH"

            # Показываем метрики если есть
            if [ -f "metrics/test_metrics_v$PROCESSED_BATCH.json" ]; then
                ACCURACY=$(python3 -c "import json; data=json.load(open('metrics/test_metrics_v$PROCESSED_BATCH.json')); print(f'{data[\"test_accuracy\"]:.4f}')" 2>/dev/null || echo "?")
                F1_SCORE=$(python3 -c "import json; data=json.load(open('metrics/test_metrics_v$PROCESSED_BATCH.json')); print(f'{data[\"test_f1_score\"]:.4f}')" 2>/dev/null || echo "?")
                echo "   📈 Метрики: точность=$ACCURACY, F1-score=$F1_SCORE"
            fi
        fi
    fi

    # Проверяем созданные файлы
    echo ""
    echo "📂 Созданные артефакты:"

    MODEL_COUNT=$(ls models/*.pkl 2>/dev/null | wc -l)
    echo "   - 🤖 Моделей: $MODEL_COUNT"

    METRICS_COUNT=$(ls metrics/test_metrics_*.json 2>/dev/null | wc -l)
    echo "   - 📊 Файлов метрик: $METRICS_COUNT"

    DATA_COUNT=$(ls data/processed/dataset_processed_*.csv 2>/dev/null | wc -l)
    echo "   - 📦 Обработанных датасетов: $DATA_COUNT"

    echo ""
    echo "🌐 Интерфейсы для мониторинга:"
    echo "   - Prefect UI: http://localhost:4200"
    echo "   - MLflow UI: http://localhost:5000"

    echo ""
    echo "⏰ Информация об автоматизации:"
    echo "   - Cron расписание: каждые 2 минуты (*/2 * * * *)"
    echo "   - Для полной автоматизации запустите worker:"
    echo "     poetry run prefect worker start --pool default-process-pool"
    echo "   - После этого пайплайн будет выполняться автоматически каждые 2 минуты!"

    echo ""
    echo "🎉 Демонстрация завершена успешно!"
    echo "   Следующий автоматический запуск произойдет через 2 минуты (если worker запущен)"

else
    echo ""
    echo "❌ Ошибка выполнения автоматического пайплайна"
    echo "   Проверьте логи выше и убедитесь что сервисы запущены"
    exit 1
fi
