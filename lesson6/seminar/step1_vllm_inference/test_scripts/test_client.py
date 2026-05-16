"""Test script for Ollama client"""

from src.llm_client import OllamaClient


def main():
    """Test Ollama client functionality"""
    print("🧪 Testing Ollama Client")
    print("=" * 50)

    client = OllamaClient()

    print("\n1️⃣ Health Check")
    print("-" * 50)
    is_healthy = client.health_check()
    print(f"Server health: {'✅ Healthy' if is_healthy else '❌ Unhealthy'}")

    if not is_healthy:
        print("\n⚠️  Server is not healthy. Start with: make start && make pull")
        return

    print("\n2️⃣ Available Models")
    print("-" * 50)
    try:
        models = client.get_models()
        for model in models:
            print(f"  - {model}")
    except Exception as e:
        print(f"Error getting models: {e}")

    print("\n3️⃣ Chat Completion Test")
    print("-" * 50)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a short joke about machine learning."},
    ]

    print("User: Tell me a short joke about machine learning.")
    print("\nAssistant: ", end="", flush=True)

    try:
        response = client.chat_completion(
            messages=messages, temperature=0.7, max_tokens=200
        )
        print(response)
    except Exception as e:
        print(f"\n❌ Error: {e}")

    print("\n4️⃣ Streaming Chat Completion Test")
    print("-" * 50)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain what is MLOps in 2 sentences."},
    ]

    print("User: Explain what is MLOps in 2 sentences.")
    print("\nAssistant: ", end="", flush=True)

    try:
        for chunk in client.chat_completion_stream(
            messages=messages, temperature=0.7, max_tokens=200
        ):
            print(chunk, end="", flush=True)
        print()
    except Exception as e:
        print(f"\n❌ Error: {e}")

    print("\n5️⃣ Russian Language Test")
    print("-" * 50)
    messages = [
        {"role": "system", "content": "Ты полезный ассистент."},
        {"role": "user", "content": "Расскажи короткую шутку про нейронные сети."},
    ]

    print("User: Расскажи короткую шутку про нейронные сети.")
    print("\nAssistant: ", end="", flush=True)

    try:
        response = client.chat_completion(
            messages=messages, temperature=0.8, max_tokens=200
        )
        print(response)
    except Exception as e:
        print(f"\n❌ Error: {e}")

    print("\n" + "=" * 50)
    print("✅ Testing completed!")


if __name__ == "__main__":
    main()
