import json
from typing import Any, Dict, List
import requests
from openai import OpenAI


class PoseAgent:
    def __init__(
        self,
        llm_base_url: str = "http://localhost:11434/v1",
        pose_api_url: str = "http://localhost:8001",
        model: str = "qwen2.5:1.5b",
    ):
        self.client = OpenAI(base_url=llm_base_url, api_key="ollama")
        self.pose_api_url = pose_api_url
        self.model = model
        self.conversation_history: List[Dict[str, Any]] = []

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "create_animation",
                    "description": "Создать анимацию из последовательности поз",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string"},
                            "poses": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "Torso": {
                                            "type": "array",
                                            "items": {"type": "number"},
                                        },
                                        "Head": {
                                            "type": "array",
                                            "items": {"type": "number"},
                                        },
                                        "RH": {
                                            "type": "array",
                                            "items": {"type": "number"},
                                        },
                                        "LH": {
                                            "type": "array",
                                            "items": {"type": "number"},
                                        },
                                        "RK": {
                                            "type": "array",
                                            "items": {"type": "number"},
                                        },
                                        "LK": {
                                            "type": "array",
                                            "items": {"type": "number"},
                                        },
                                    },
                                    "required": [
                                        "Torso",
                                        "Head",
                                        "RH",
                                        "LH",
                                        "RK",
                                        "LK",
                                    ],
                                },
                            },
                        },
                        "required": ["action", "poses"],
                    },
                },
            }
        ]

        self.system_message = """Create pose sequences for actions.

COORDINATES: Torso (0,0), Head (0,60), Hands Y=35, Knees Y=-50

EXAMPLES:
WAVE: [{"Torso":[0,0],"Head":[0,60],"RH":[20,40],"LH":[-40,30],"RK":[15,-50],"LK":[-15,-50]},
       {"Torso":[0,0],"Head":[0,60],"RH":[30,70],"LH":[-40,30],"RK":[15,-50],"LK":[-15,-50]}]

JUMP: [{"Torso":[0,0],"Head":[0,60],"RH":[25,35],"LH":[-25,35],"RK":[15,-50],"LK":[-15,-50]},
       {"Torso":[0,10],"Head":[0,70],"RH":[30,55],"LH":[-30,55],"RK":[10,-30],"LK":[-10,-30]}]"""

    def _call_function(self, function_name: str, arguments: Dict[str, Any]) -> Dict:
        import base64
        import io
        from PIL import Image

        if function_name == "create_animation":
            poses = arguments.get("poses", [])
            if not poses:
                return {"error": "No poses"}

            frames = []
            for pose in poses:
                response = requests.post(
                    f"{self.pose_api_url}/visualize",
                    json={"pose": pose},
                    timeout=10,
                )
                result = response.json()

                if result.get("success") and result.get("image"):
                    img_data = base64.b64decode(result["image"])
                    img = Image.open(io.BytesIO(img_data))
                    frames.append(img)

            if not frames:
                return {"error": "Failed to generate frames"}

            gif_buffer = io.BytesIO()
            frames[0].save(
                gif_buffer,
                format="GIF",
                save_all=True,
                append_images=frames[1:],
                duration=500,
                loop=0,
            )
            gif_buffer.seek(0)

            gif_base64 = base64.b64encode(gif_buffer.read()).decode("utf-8")

            return {
                "success": True,
                "animation": gif_base64,
                "format": "base64_gif",
                "frames": len(frames),
            }

        return {"error": f"Unknown function: {function_name}"}

    def chat(self, user_message: str, max_iterations: int = 5) -> Dict[str, Any]:
        self.conversation_history.append({"role": "user", "content": user_message})

        messages = [{"role": "system", "content": self.system_message}]
        messages.extend(self.conversation_history)

        iteration = 0
        last_image = None

        while iteration < max_iterations:
            iteration += 1

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=1024,
            )

            assistant_message = response.choices[0].message

            if assistant_message.tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_message.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in assistant_message.tool_calls
                        ],
                    }
                )

                for tool_call in assistant_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    function_result = self._call_function(function_name, function_args)

                    if "animation" in function_result:
                        last_image = function_result["animation"]
                    elif "image" in function_result:
                        last_image = function_result["image"]

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(function_result, ensure_ascii=False),
                        }
                    )

                continue

            else:
                final_response = assistant_message.content or ""
                self.conversation_history.append(
                    {"role": "assistant", "content": final_response}
                )

                return {"text": final_response, "image": last_image}

        return {"text": "Max iterations exceeded", "image": None}

    def reset_conversation(self):
        self.conversation_history = []
