# Ollama — runs models locally, no API key needed

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from triagent.backends.base import BackendResponse, ModelBackend


class OllamaBackend(ModelBackend):

    def __init__(self, api_key: str = "", model: str = "llama3", host: str = "", **kwargs):
        super().__init__(api_key=api_key, model=model, **kwargs)
        self.model = model
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self._client = None

    @property
    def name(self) -> str:
        return "Ollama (Local)"

    def initialize(self) -> None:
        import ollama
        self._client = ollama.Client(host=self.host)

    async def generate(self, prompt, system_prompt=None, temperature=0.7, max_tokens=4096, json_mode=False) -> BackendResponse:
        self._ensure_initialized()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self.model,
            "messages": messages,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if json_mode:
            kwargs["format"] = "json"

        response = self._client.chat(**kwargs)

        usage = {}
        if hasattr(response, "prompt_eval_count"):
            usage["prompt_tokens"] = response.get("prompt_eval_count", 0)
            usage["completion_tokens"] = response.get("eval_count", 0)
            usage["total_tokens"] = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)

        text = ""
        if isinstance(response, dict):
            msg = response.get("message", {})
            text = msg.get("content", "") if isinstance(msg, dict) else ""
        else:
            text = response.message.content if hasattr(response, "message") else str(response)

        return BackendResponse(text=text, model=self.model, backend="ollama", usage=usage, raw_response=response)

    async def generate_with_image(self, prompt, image_path, system_prompt=None, temperature=0.7, max_tokens=4096, json_mode=False) -> BackendResponse:
        self._ensure_initialized()

        b64_data, _ = self._load_image_as_base64(image_path)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt, "images": [b64_data]})

        # If model can't do vision, fallback to llava
        vision_model = self.model
        non_vision = {"llama3", "llama2", "mistral", "mixtral", "phi", "qwen"}
        if any(nv in vision_model.lower() for nv in non_vision):
            vision_model = "llava"

        kwargs = {
            "model": vision_model,
            "messages": messages,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if json_mode:
            kwargs["format"] = "json"

        response = self._client.chat(**kwargs)

        text = ""
        if isinstance(response, dict):
            msg = response.get("message", {})
            text = msg.get("content", "") if isinstance(msg, dict) else ""
        else:
            text = response.message.content if hasattr(response, "message") else str(response)

        return BackendResponse(text=text, model=vision_model, backend="ollama", usage={}, raw_response=response)

    async def is_available(self) -> bool:
        try:
            self._ensure_initialized()
            self._client.list()
            return True
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        try:
            self._ensure_initialized()
            response = self._client.list()
            if isinstance(response, dict):
                return [m.get("name", "") for m in response.get("models", [])]
            return [m.model for m in response.models] if hasattr(response, "models") else []
        except Exception:
            return []
