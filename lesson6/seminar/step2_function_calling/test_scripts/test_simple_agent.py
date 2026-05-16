"""Simple agent test with function calling"""

import sys

sys.path.insert(0, ".")

from src.pose_agent import PoseAgent
import base64
from pathlib import Path

print("🤖 Testing Pose Agent with Function Calling")
print("=" * 60)

agent = PoseAgent()

output_dir = Path("test_output_simple")
output_dir.mkdir(exist_ok=True)

# Test 1: Simple pose
print("\n1. Testing: 'Создай T-позу'")
try:
    result = agent.chat("Создай T-позу")
    print(f"   Response: {result['text'][:100]}...")

    if result.get("image"):
        image_data = base64.b64decode(result["image"])
        with open(output_dir / "t_pose.png", "wb") as f:
            f.write(image_data)
        print(f"   ✅ Image saved to {output_dir}/t_pose.png")
    else:
        print("   ⚠️  No image returned")

except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "=" * 60)
print("✅ Test completed!")
