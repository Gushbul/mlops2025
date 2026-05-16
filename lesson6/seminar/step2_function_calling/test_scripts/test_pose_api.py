"""Тест Pose API без vLLM"""

import base64
import requests
from pathlib import Path


def save_image(image_base64: str, filename: str):
    """Сохранить base64 изображение в файл"""
    if image_base64:
        image_data = base64.b64decode(image_base64)
        with open(filename, "wb") as f:
            f.write(image_data)
        print(f"  💾 {filename}")


def main():
    """Тест Pose API"""
    print("🎨 Тест Pose API")
    print("=" * 60)

    api_url = "http://localhost:8001"
    output_dir = Path("test_output_api")
    output_dir.mkdir(exist_ok=True)

    print("\n1. Health Check")
    response = requests.get(f"{api_url}/health")
    print(f"   {response.json()['status']}")

    poses = {
        "t_pose": {
            "Torso": [0, 0],
            "Head": [0, 60],
            "RH": [50, 35],
            "LH": [-50, 35],
            "RK": [15, -50],
            "LK": [-15, -50],
        },
        "jump": {
            "Torso": [0, 0],
            "Head": [0, 60],
            "RH": [25, 55],
            "LH": [-25, 55],
            "RK": [10, -30],
            "LK": [-10, -30],
        },
        "star": {
            "Torso": [0, 0],
            "Head": [0, 60],
            "RH": [60, 40],
            "LH": [-60, 40],
            "RK": [40, -60],
            "LK": [-40, -60],
        },
    }

    print("\n2. Визуализация поз")
    for i, (name, pose) in enumerate(poses.items(), 1):
        response = requests.post(f"{api_url}/visualize", json={"pose": pose})
        result = response.json()

        if result["success"]:
            save_image(result["image"], output_dir / f"{name}.png")

    print("\n" + "=" * 60)
    print(f"✅ Готово! Изображения в {output_dir.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
