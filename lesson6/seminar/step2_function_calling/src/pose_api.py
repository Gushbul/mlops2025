import base64
import io
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class PoseData(BaseModel):
    Torso: List[float]
    Head: List[float]
    RH: List[float]
    LH: List[float]
    RK: List[float]
    LK: List[float]


class PoseRequest(BaseModel):
    pose: PoseData


def draw_pose(pose: PoseData) -> str:
    fig, ax = plt.subplots(figsize=(6, 8))

    torso_x, torso_y = pose.Torso
    head_x, head_y = pose.Head
    rh_x, rh_y = pose.RH
    lh_x, lh_y = pose.LH
    rk_x, rk_y = pose.RK
    lk_x, lk_y = pose.LK

    shoulder_offset = 15
    hip_offset = 10

    r_shoulder = (torso_x + shoulder_offset, torso_y + 20)
    l_shoulder = (torso_x - shoulder_offset, torso_y + 20)
    r_hip = (torso_x + hip_offset, torso_y - 20)
    l_hip = (torso_x - hip_offset, torso_y - 20)

    head_circle = plt.Circle((head_x, head_y), 8, color="#FFD700", zorder=3)
    ax.add_patch(head_circle)

    ax.plot(
        [l_shoulder[0], r_shoulder[0]],
        [l_shoulder[1], r_shoulder[1]],
        "o-",
        color="#4A90E2",
        linewidth=4,
        markersize=8,
    )
    ax.plot(
        [torso_x, torso_x], [r_shoulder[1], torso_y], "-", color="#4A90E2", linewidth=4
    )
    ax.plot([torso_x, torso_x], [torso_y, r_hip[1]], "-", color="#4A90E2", linewidth=4)
    ax.plot(
        [l_hip[0], r_hip[0]],
        [l_hip[1], r_hip[1]],
        "o-",
        color="#4A90E2",
        linewidth=4,
        markersize=8,
    )

    ax.plot(
        [torso_x, head_x],
        [r_shoulder[1], head_y - 8],
        "-",
        color="#4A90E2",
        linewidth=3,
    )

    ax.plot(
        [r_shoulder[0], rh_x],
        [r_shoulder[1], rh_y],
        "o-",
        color="#E74C3C",
        linewidth=3,
        markersize=10,
    )

    ax.plot(
        [l_shoulder[0], lh_x],
        [l_shoulder[1], lh_y],
        "o-",
        color="#2ECC71",
        linewidth=3,
        markersize=10,
    )

    ax.plot(
        [r_hip[0], rk_x],
        [r_hip[1], rk_y],
        "o-",
        color="#E74C3C",
        linewidth=3,
        markersize=10,
    )

    ax.plot(
        [l_hip[0], lk_x],
        [l_hip[1], lk_y],
        "o-",
        color="#2ECC71",
        linewidth=3,
        markersize=10,
    )

    ax.set_xlim(-150, 150)
    ax.set_ylim(-150, 150)
    ax.set_aspect("equal")
    ax.axis("off")

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/visualize")
async def visualize_pose(request: PoseRequest):
    image_base64 = draw_pose(request.pose)
    return {"success": True, "image": image_base64, "format": "base64_png"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
