"""Simple import test to verify package structure"""


def test_imports():
    """Test that all modules can be imported"""
    try:
        from src.llm_client import VLLMClient

        print("✅ VLLMClient imported successfully")

        # Check that class has required methods
        assert hasattr(VLLMClient, "chat_completion")
        assert hasattr(VLLMClient, "chat_completion_stream")
        assert hasattr(VLLMClient, "get_models")
        assert hasattr(VLLMClient, "health_check")
        print("✅ VLLMClient has all required methods")

        return True
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_imports()
    exit(0 if success else 1)
