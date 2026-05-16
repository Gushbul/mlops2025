"""Демонстрация создания анимации без LLM"""

import requests
import base64
import io
from pathlib import Path
from PIL import Image

print("🎬 Создание анимаций действий")
print("=" * 60)

output_dir = Path("demo_animations")
output_dir.mkdir(exist_ok=True)

# Предопределенные анимации
animations = {
    "wave": [
        {
            "Torso": [0, 0],
            "Head": [0, 60],
            "RH": [20, 40],
            "LH": [-40, 30],
            "RK": [15, -50],
            "LK": [-15, -50],
        },
        {
            "Torso": [0, 0],
            "Head": [0, 60],
            "RH": [30, 70],
            "LH": [-40, 30],
            "RK": [15, -50],
            "LK": [-15, -50],
        },
        {
            "Torso": [0, 0],
            "Head": [0, 60],
            "RH": [40, 50],
            "LH": [-40, 30],
            "RK": [15, -50],
            "LK": [-15, -50],
        },
        {
            "Torso": [0, 0],
            "Head": [0, 60],
            "RH": [30, 70],
            "LH": [-40, 30],
            "RK": [15, -50],
            "LK": [-15, -50],
        },
    ],
    "jump": [
        {
            "Torso": [0, 0],
            "Head": [0, 60],
            "RH": [25, 35],
            "LH": [-25, 35],
            "RK": [15, -50],
            "LK": [-15, -50],
        },
        {
            "Torso": [0, 10],
            "Head": [0, 70],
            "RH": [30, 55],
            "LH": [-30, 55],
            "RK": [10, -30],
            "LK": [-10, -30],
        },
        {
            "Torso": [0, 0],
            "Head": [0, 60],
            "RH": [25, 35],
            "LH": [-25, 35],
            "RK": [15, -50],
            "LK": [-15, -50],
        },
    ],
    "squat": [
        {
            "Torso": [0, 0],
            "Head": [0, 60],
            "RH": [40, 30],
            "LH": [-40, 30],
            "RK": [15, -50],
            "LK": [-15, -50],
        },
        {
            "Torso": [0, -30],
            "Head": [0, 30],
            "RH": [40, 0],
            "LH": [-40, 0],
            "RK": [20, -60],
            "LK": [-20, -60],
        },
        {
            "Torso": [0, 0],
            "Head": [0, 60],
            "RH": [40, 30],
            "LH": [-40, 30],
            "RK": [15, -50],
            "LK": [-15, -50],
        },
    ],
    "dance": [
        {
            "Torso": [0, 0],
            "Head": [0, 60],
            "RH": [50, 35],
            "LH": [-50, 35],
            "RK": [15, -50],
            "LK": [-15, -50],
        },
        {
            "Torso": [5, 0],
            "Head": [5, 60],
            "RH": [55, 50],
            "LH": [-45, 20],
            "RK": [20, -50],
            "LK": [-10, -50],
        },
        {
            "Torso": [-5, 0],
            "Head": [-5, 60],
            "RH": [45, 20],
            "LH": [-55, 50],
            "RK": [10, -50],
            "LK": [-20, -50],
        },
        {
            "Torso": [0, 0],
            "Head": [0, 60],
            "RH": [50, 35],
            "LH": [-50, 35],
            "RK": [15, -50],
            "LK": [-15, -50],
        },
    ],
}

for action_name, poses in animations.items():
    print(f"\n{action_name.upper()}:")
    try:
        # Получаем изображения для каждой позы
        frames = []
        for i, pose in enumerate(poses):
            response = requests.post(
                "http://localhost:8001/visualize", json={"pose": pose}, timeout=5
            )
            result = response.json()

            if result.get("success") and result.get("image"):
                img_data = base64.b64decode(result["image"])
                img = Image.open(io.BytesIO(img_data))
                frames.append(img)
                print(f"  ✓ Frame {i+1}/{len(poses)}")
            else:
                print(f"  ✗ Frame {i+1} failed")

        if frames:
            # Сохраняем GIF
            gif_path = output_dir / f"{action_name}.gif"
            frames[0].save(
                gif_path,
                format="GIF",
                save_all=True,
                append_images=frames[1:],
                duration=500,
                loop=0,
            )
            print(f"  ✅ GIF saved: {gif_path}")
        else:
            print("  ❌ No frames generated")

    except Exception as e:
        print(f"  ❌ Error: {e}")

print("\n" + "=" * 60)
print(f"✅ Готово! Анимации в {output_dir.absolute()}")
print("=" * 60)
