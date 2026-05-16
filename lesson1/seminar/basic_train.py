#!/usr/bin/env python3
"""
Простой код обучения модели - ПЛОХОЙ ПРИМЕР

ПРОБЛЕМЫ:
1. Нет воспроизводимости - каждый запуск дает разные результаты!
2. Нет логирования параметров и экспериментов
3. Хардкод значений в коде
4. Нет версионирования модели
5. Нет сохранения метаданных и конфигурации

Сравните с src/app/train.py - правильным подходом!
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import pickle
import os


def main():
    print("Загружаем данные...")
    iris = load_iris()
    X, y = iris.data, iris.target

    print("Разбиваем данные...")
    # ПРОБЛЕМА: нет фиксированного random_state - каждый раз разные данные!
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print("Обучаем модель...")
    # ПРОБЛЕМА: хардкод параметров в коде!
    model = LogisticRegression(max_iter=100)  # Нет random_state!
    model.fit(X_train, y_train)

    print("Делаем предсказания...")
    y_pred = model.predict(X_test)

    # ПРОБЛЕМА: считаем метрики, но НЕ ЛОГИРУЕМ и не сохраняем!
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"Accuracy: {accuracy:.3f}")
    print(f"F1-score: {f1:.3f}")

    # ПРОБЛЕМА: простое сохранение без метаданных и версионирования!
    os.makedirs("simple_models", exist_ok=True)
    with open("simple_models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Модель сохранена в simple_models/model.pkl")
    print("Готово!")


if __name__ == "__main__":
    main()
