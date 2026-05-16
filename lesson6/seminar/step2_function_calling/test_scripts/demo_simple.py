"""Простая демонстрация без LLM - прямое использование API"""

import requests
import base64
from pathlib import Path

print("🎨 Простая демонстрация Pose API")
print("=" * 60)

output_dir = Path("demo_output")
output_dir.mkdir(exist_ok=True)

# Заранее определенные позы
poses = {
    "T-Pose": {
        "Torso": [0, 0],
        "Head": [0, 60],
        "RH": [50, 35],
        "LH": [-50, 35],
        "RK": [15, -50],
        "LK": [-15, -50],
    },
    "Jump": {
        "Torso": [0, 0],
        "Head": [0, 60],
        "RH": [25, 55],
        "LH": [-25, 55],
        "RK": [10, -30],
        "LK": [-10, -30],
    },
    "Wave": {
        "Torso": [0, 0],
        "Head": [0, 60],
        "RH": [30, 70],
        "LH": [-40, 20],
        "RK": [15, -50],
        "LK": [-15, -50],
    },
    "Star": {
        "Torso": [0, 0],
        "Head": [0, 60],
        "RH": [60, 40],
        "LH": [-60, 40],
        "RK": [40, -60],
        "LK": [-40, -60],
    },
    "Sitting": {
        "Torso": [0, -20],
        "Head": [0, 40],
        "RH": [30, -10],
        "LH": [-30, -10],
        "RK": [25, -50],
        "LK": [-25, -50],
    },
}

for name, pose_data in poses.items():
    print(f"\n{name}:")
    try:
        response = requests.post(
            "http://localhost:8001/visualize", json={"pose": pose_data}, timeout=5
        )
        result = response.json()

        if result["success"]:
            # Сохраняем изображение
            image_data = base64.b64decode(result["image"])
            filename = output_dir / f"{name.lower().replace(' ', '_')}.png"
            with open(filename, "wb") as f:
                f.write(image_data)
            print(f"  ✅ Saved to {filename}")
        else:
            print("  ❌ Error")

    except Exception as e:
        print(f"  ❌ Error: {e}")

print("\n" + "=" * 60)
print(f"✅ Готово! Изображения в {output_dir.absolute()}")
print("=" * 60)
