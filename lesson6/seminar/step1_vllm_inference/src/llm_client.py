from typing import Dict, List
from openai import OpenAI


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434/v1"):
        self.client = OpenAI(base_url=base_url, api_key="ollama")
        self.base_url = base_url

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "qwen2.5:1.5b",
        temperature: float = 0.7,
        max_tokens: int = 512,
        stream: bool = False,
    ) -> str:
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
        )

        if stream:
            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
            return full_response
        else:
            return response.choices[0].message.content

    def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        model: str = "qwen2.5:1.5b",
        temperature: float = 0.7,
        max_tokens: int = 512,
    ):
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    def get_models(self) -> List[str]:
        models = self.client.models.list()
        return [model.id for model in models.data]

    def health_check(self) -> bool:
        import requests

        try:
            health_url = self.base_url.replace("/v1", "")
            response = requests.get(health_url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False
