from __future__ import annotations

import argparse
from pathlib import Path

from src.animation import GifAnimationBuilder
from src.config import load_config
from src.ollama_client import OllamaClient
from src.planner import PoseSequencePlanner
from src.pose_client import LocalPoseRenderer, PoseApiClient
from src.rag import PoseDatabase


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG Animation System")
    parser.add_argument(
        "query",
        nargs="?",
        default="танец макарена",
        help="Text description of animation, for example: 'танец макарена'",
    )
    parser.add_argument(
        "--config",
        default="config/rag_config.yaml",
        help="Path to YAML config",
    )
    parser.add_argument("--output", default=None, help="Output GIF path")
    parser.add_argument("--top-k", type=int, default=None, help="Number of RAG poses")
    parser.add_argument(
        "--no-ollama",
        action="store_true",
        help="Disable Ollama planning and use deterministic retrieval fallback",
    )
    parser.add_argument(
        "--no-local-fallback",
        action="store_true",
        help="Fail if Pose API is unavailable instead of using local renderer",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)

    database = PoseDatabase(config.rag.database_path)
    top_k = args.top_k or config.rag.top_k
    retrieved = database.retrieve(args.query, top_k=top_k)

    ollama_client = None
    if not args.no_ollama:
        ollama_client = OllamaClient(
            base_url=config.services.ollama_url,
            model=config.services.ollama_model,
        )

    planner = PoseSequencePlanner(database, ollama_client=ollama_client)
    plan = planner.plan(args.query, retrieved)
    records = [database.get(pose_id) for pose_id in plan.pose_ids]

    pose_api = PoseApiClient(config.services.pose_api_url)
    if pose_api.is_available():
        renderer = pose_api
        renderer_source = "Pose API"
    elif config.animation.use_local_renderer_fallback and not args.no_local_fallback:
        renderer = LocalPoseRenderer()
        renderer_source = "local Pillow fallback"
    else:
        raise RuntimeError(
            "Pose API is unavailable. Start Step 2 service on port 8001 or "
            "enable local fallback in config."
        )

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = (
            Path(config.animation.output_dir) / config.animation.default_output
        )
    builder = GifAnimationBuilder(
        renderer=renderer,
        duration_ms=config.animation.frame_duration_ms,
        loop=config.animation.loop,
    )
    gif_path = builder.build(records, output_path)

    print("=== RAG Animation System ===")
    print(f"Query: {args.query}")
    print(f"Loaded poses: {len(database.records)}")
    print("Retrieved candidates:")
    for record in retrieved:
        print(f"  #{record.pose_id}: {record.description}")
    print(f"Plan source: {plan.source}")
    print(f"Plan explanation: {plan.explanation}")
    print(f"Pose sequence: {plan.pose_ids}")
    print(f"Renderer: {renderer_source}")
    print(f"GIF saved to: {gif_path}")


if __name__ == "__main__":
    main()
