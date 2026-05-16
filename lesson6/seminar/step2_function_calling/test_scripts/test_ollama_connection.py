"""Quick Ollama connection test"""

import sys

sys.path.insert(0, ".")

from src.pose_agent import PoseAgent

print("Testing Ollama connection...")
print("-" * 50)

try:
    agent = PoseAgent(
        llm_base_url="http://localhost:11434/v1",
        pose_api_url="http://localhost:8001",
        model="qwen2.5:1.5b",
    )

    print("✅ Agent initialized")

    # Simple test
    result = agent.chat("Say 'Hello'")
    print(f"✅ Response: {result['text']}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
