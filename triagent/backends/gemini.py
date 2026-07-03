# Google Gemini — supports both text and vision (image+text)

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from triagent.backends.base import BackendResponse, ModelBackend


class GeminiBackend(ModelBackend):

    def __init__(self, api_key: str = "", model: str = "gemini-2.5-flash", **kwargs):
        super().__init__(api_key=api_key, model=model, **kwargs)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model = model
        self._client = None

    @property
    def name(self) -> str:
        return "Google Gemini"

    def initialize(self) -> None:
        from google import genai

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set. Get one at https://aistudio.google.com/apikey")
        self._client = genai.Client(api_key=self.api_key)

    async def generate(self, prompt, system_prompt=None, temperature=0.7, max_tokens=4096, json_mode=False) -> BackendResponse:
        self._ensure_initialized()
        from google.genai import types

        config_kwargs = {"temperature": temperature, "max_output_tokens": max_tokens}
        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"

        config = types.GenerateContentConfig(**config_kwargs)

        response = self._client.models.generate_content(
            model=self.model, contents=prompt, config=config,
        )

        usage = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            um = response.usage_metadata
            usage = {
                "prompt_tokens": getattr(um, "prompt_token_count", 0),
                "completion_tokens": getattr(um, "candidates_token_count", 0),
                "total_tokens": getattr(um, "total_token_count", 0),
            }

        return BackendResponse(text=response.text or "", model=self.model, backend="gemini", usage=usage, raw_response=response)

    async def generate_with_image(self, prompt, image_path, system_prompt=None, temperature=0.7, max_tokens=4096, json_mode=False) -> BackendResponse:
        self._ensure_initialized()
        from google.genai import types

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        with open(path, "rb") as f:
            image_bytes = f.read()

        suffix = path.suffix.lower()
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"}
        mime_type = mime_map.get(suffix, "image/jpeg")

        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        contents = [image_part, prompt]

        config_kwargs = {"temperature": temperature, "max_output_tokens": max_tokens}
        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"

        config = types.GenerateContentConfig(**config_kwargs)

        response = self._client.models.generate_content(
            model=self.model, contents=contents, config=config,
        )

        usage = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            um = response.usage_metadata
            usage = {
                "prompt_tokens": getattr(um, "prompt_token_count", 0),
                "completion_tokens": getattr(um, "candidates_token_count", 0),
                "total_tokens": getattr(um, "total_token_count", 0),
            }

        return BackendResponse(text=response.text or "", model=self.model, backend="gemini", usage=usage, raw_response=response)

    async def is_available(self) -> bool:
        try:
            self._ensure_initialized()
            self._client.models.list()
            return True
        except Exception:
            return False
