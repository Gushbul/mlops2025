# Step 4: Independent Active Learning and Baseline Flows

Этот шаг демонстрирует независимые flow для Active Learning и Baseline обучения с возможностью сравнения результатов.

## Основные возможности

### 🎯 **Новый датасет**
- **Digits Dataset**: Классификация рукописных цифр (0-9)
- 1797 образцов, 64 признака (8x8 пикселей)
- 10 классов (цифры 0-9)
- Более сложная задача по сравнению с Forest Cover Type

### 🔄 **Independent Flow Architecture**
- **Active Learning Flow**: Инкрементальное обучение с uncertainty sampling
- **Baseline Flow**: Обучение на полном датасете
- **Independent Execution**: Каждый flow запускается отдельно
- **MLflow Tracking**: Отдельные эксперименты для каждого flow
- **Prefect Orchestration**: Управление workflow

### 📊 **Active Learning Features**
- **Uncertainty Sampling**: 3 стратегии (entropy, margin, least confident)
- **Incremental Steps**: 5% данных на каждой итерации
- **Data Efficiency**: Достижение лучшего качества с меньшими данными
- **15 Iterations**: Постепенное добавление данных до 74.4%

## Архитектура

```
step4/
├── src/
│   ├── data_manager.py          # Управление данными (digits dataset)
│   ├── model_trainer.py         # Обучение CatBoost
│   ├── active_learning.py       # AL стратегии
│   └── baseline_trainer.py      # NEW: Обучение на полном датасете
├── flows/
│   ├── active_learning_flow.py  # AL Prefect flow (инкрементальное обучение)
│   └── baseline_flow.py         # Baseline Prefect flow (полный датасет)
├── pyproject.toml              # Poetry конфигурация
├── Makefile                    # Автоматизация
└── README.md                   # Документация
```

## Использование

### 🚀 **Быстрый старт**
```bash
# Установка зависимостей
make install

# Запуск сервисов
make start-services

# Запуск обоих flow
make run-both
```

### 🔬 **Отдельные эксперименты**
```bash
# Только Baseline (полный датасет)
make run-baseline

# Только Active Learning (entropy)
make run-al

# Active Learning с разными стратегиями
make run-al-entropy
make run-al-margin
make run-al-confident

# Бенчмарк всех стратегий
make benchmark
```

### 📈 **Анализ результатов**
```bash
# MLflow UI
make mlflow-ui
# Откройте http://localhost:5000

# Prefect UI
make prefect-server
# Откройте http://localhost:4200
```

## Результаты экспериментов

### 🔵 **Baseline Results (100% данных)**
- **Test Accuracy**: 96.67%
- **Val Accuracy**: 98.61%
- **Training Samples**: 1149 (100% данных)
- **MLflow Experiment**: `step4_baseline`

### 🟢 **Active Learning Results (74.4% данных)**
- **Test Accuracy**: 97.78% (**+1.11% лучше!**)
- **Val Accuracy**: 98.96% (**+0.35% лучше!**)
- **Training Samples**: 855 (74.4% данных, экономия 25.6%)
- **Iterations**: 15 (по 5% данных каждая)
- **Strategies**: entropy и margin показывают одинаковые результаты
- **MLflow Experiment**: `step4_active_learning`

### 🏆 **Ключевые выводы**
- **Data Efficiency**: AL достигает лучшего качества с меньшими данными
- **Performance**: +1.11% улучшение accuracy при экономии 25.6% данных
- **Incremental Learning**: 5% шаги показывают эффективность
- **Uncertainty Sampling**: entropy и margin стратегии одинаково эффективны

## Команды Makefile

| Команда | Описание |
|---------|----------|
| `make install` | Установка зависимостей |
| `make start-services` | Запуск Prefect + MLflow |
| `make run-al-entropy` | AL с entropy sampling |
| `make run-al-margin` | AL с margin sampling |
| `make run-al-confident` | AL с least confident sampling |
| `make run-baseline` | Обучение на полном датасете |
| `make run-both` | Запуск обоих flow (baseline + AL) |
| `make benchmark` | Бенчмарк всех AL стратегий |
| `make clean` | Очистка артефактов |
| `make stop-services` | Остановка сервисов |

## Технические детали

### 🔧 **Конфигурация**
- **Python**: 3.9+
- **CatBoost**: Gradient boosting для классификации
- **MLflow**: Experiment tracking
- **Prefect**: Workflow orchestration
- **Poetry**: Dependency management

### 📝 **Логирование**
- **MLflow Experiments**: Отдельные эксперименты для AL и Baseline
- **Metrics**: accuracy, f1_macro, f1_weighted, precision, recall
- **Parameters**: sampling_strategy, batch_size, iterations
- **Artifacts**: Модели, confusion matrices, learning curves

### 🎛️ **Настройки**
- **Initial Labeled**: 5% от training set
- **Increment Size**: 5% от training set на каждой итерации
- **Max Iterations**: 15 (до достижения 74.4% данных)
- **Random Seed**: 42 (воспроизводимость)
- **Independent Flows**: Baseline и AL запускаются отдельно
