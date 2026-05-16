"""Debug agent test"""

import sys
import json

sys.path.insert(0, ".")

from src.pose_agent import PoseAgent

print("🤖 Debug Pose Agent")
print("=" * 60)

agent = PoseAgent()

# Manually test function call
print("\n1. Manual function test")
test_pose = {
    "Torso": [0, 0],
    "Head": [0, 60],
    "RH": [50, 35],
    "LH": [-50, 35],
    "RK": [15, -50],
    "LK": [-15, -50],
}

result = agent._call_function("visualize_pose", {"pose": test_pose})
print(f"Result keys: {result.keys()}")
print(f"Success: {result.get('success')}")
if result.get("image"):
    print(f"Image length: {len(result['image'])} chars")
else:
    print(f"Full result: {json.dumps(result, indent=2)}")

# Test with LLM
print("\n2. LLM Test with English")
try:
    result = agent.chat("Create a T-pose")
    print(f"Response: {result['text']}")
    print(f"Has image: {bool(result.get('image'))}")
    if result.get("image"):
        print(f"Image size: {len(result['image'])}")
except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
