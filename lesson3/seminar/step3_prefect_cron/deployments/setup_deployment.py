"""
Настройка деплойментов для автоматического пайплайна (Prefect 3.0).
"""

import sys
import os
import asyncio


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from flows.automated_training_flow import (
    automated_training_pipeline,
    manual_training_pipeline,
)


async def create_automated_deployment():
    """Создание деплоймента с cron расписанием для Prefect 3.0."""

    print("Создание автоматического деплоймента...")

    deployment = await automated_training_pipeline.to_deployment(
        name="automated-ml-pipeline",
        description="Автоматический ML пайплайн, запускающийся каждые 2 минуты",
        version="1.0.0",
        cron="*/2 * * * *",
        tags=["ml", "automated", "cron"],
    )

    deployment_id = await deployment.apply()
    print(f"✅ Создан автоматический деплоймент: {deployment_id}")
    return deployment_id


async def create_manual_deployment():
    """Создание деплоймента для ручного запуска для Prefect 3.0."""

    print("Создание ручного деплоймента...")

    deployment = await manual_training_pipeline.to_deployment(
        name="manual-ml-pipeline",
        description="Ручной запуск ML пайплайна для конкретного батча",
        version="1.0.0",
        tags=["ml", "manual"],
    )

    deployment_id = await deployment.apply()
    print(f"✅ Создан ручной деплоймент: {deployment_id}")
    return deployment_id


async def main():
    """Основная функция для создания всех деплойментов."""
    print("🚀 Создание Prefect 3.0 деплойментов...")

    try:

        auto_id = await create_automated_deployment()
        manual_id = await create_manual_deployment()

        print("\n✅ Все деплойменты созданы успешно!")
        print(f"   - Автоматический: {auto_id}")
        print(f"   - Ручной: {manual_id}")

        print("\n🔧 Для запуска:")
        print(
            "   1. Запустите worker: poetry run prefect worker start --pool default-process-pool"
        )
        print("   2. Или используйте: poetry run prefect agent start -q default")
        print("   3. Автоматический пайплайн начнет работать через 2 минуты")
        print("\n🌐 Мониторинг:")
        print("   - Prefect UI: http://localhost:4200")
        print("   - MLflow UI: http://localhost:5000")

    except Exception as e:
        print(f"❌ Ошибка создания деплойментов: {e}")
        print(
            "💡 Убедитесь что Prefect сервер запущен: poetry run prefect server start"
        )
        return False

    return True


if __name__ == "__main__":

    success = asyncio.run(main())
