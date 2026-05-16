"""Проверка импортов"""


def test_imports():
    """Test that all modules can be imported"""
    try:
        from src.pose_agent import PoseAgent

        print("✅ PoseAgent imported")

        assert hasattr(PoseAgent, "chat")
        assert hasattr(PoseAgent, "reset_conversation")
        print("✅ PoseAgent methods OK")

        agent = PoseAgent()
        assert hasattr(agent, "tools")
        assert len(agent.tools) == 1
        print(f"✅ PoseAgent has {len(agent.tools)} tool")

        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    success = test_imports()
    exit(0 if success else 1)
