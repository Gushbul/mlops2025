"""Тест Pose Agent с function calling"""

import time
import base64
from pathlib import Path

from src.pose_agent import PoseAgent


def save_image(image_base64: str, filename: str):
    """Сохранить изображение"""
    if image_base64:
        image_data = base64.b64decode(image_base64)
        with open(filename, "wb") as f:
            f.write(image_data)
        print(f"  💾 {filename}")


def main():
    """Тест Pose Agent"""
    print("🤖 Тест Pose Agent")
    print("=" * 60)

    agent = PoseAgent(
        llm_base_url="http://localhost:8000/v1",
        pose_api_url="http://localhost:8001",
    )

    print("\n✅ Agent OK")

    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)

    test_cases = [
        "Создай T-позу",
        "Создай позу прыжка",
        "Создай креативную позу танцора",
    ]

    for i, message in enumerate(test_cases, 1):
        print(f"\n{i}. {message}")

        try:
            result = agent.chat(message)
            print(f"   {result['text']}")

            if result.get("image"):
                filename = output_dir / f"pose_{i}.png"
                save_image(result["image"], str(filename))

        except Exception as e:
            print(f"   ❌ {e}")

        time.sleep(1)

    print("\n" + "=" * 60)
    print(f"✅ Готово! Изображения в {output_dir.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
