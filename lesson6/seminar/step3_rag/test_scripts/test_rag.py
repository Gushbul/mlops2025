from pathlib import Path

from src.rag import PoseDatabase


def test_retrieve_macarena_poses():
    db = PoseDatabase(Path(__file__).resolve().parents[1] / "poses_database.json")
    retrieved = db.retrieve("танец макарена", top_k=5)
    assert retrieved
    assert any("Макарена" in record.description for record in retrieved)
