# xAI Grok — uses OpenAI-compatible SDK since their API follows the same format

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from triagent.backends.base import BackendResponse, ModelBackend


class GrokBackend(ModelBackend):

    XAI_BASE_URL = "https://api.x.ai/v1"

    def __init__(self, api_key: str = "", model: str = "grok-3-mini", **kwargs):
        super().__init__(api_key=api_key, model=model, **kwargs)
        self.api_key = api_key or os.getenv("XAI_API_KEY", "")
        self.model = model
        self._client = None

    @property
    def name(self) -> str:
        return "xAI Grok"

    def initialize(self) -> None:
        from openai import OpenAI

        if not self.api_key:
            raise ValueError("XAI_API_KEY not set. Get one at https://console.x.ai/")
        self._client = OpenAI(api_key=self.api_key, base_url=self.XAI_BASE_URL)

    async def generate(self, prompt, system_prompt=None, temperature=0.7, max_tokens=4096, json_mode=False) -> BackendResponse:
        self._ensure_initialized()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(**kwargs)

        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return BackendResponse(text=response.choices[0].message.content or "", model=self.model, backend="grok", usage=usage, raw_response=response)

    async def generate_with_image(self, prompt, image_path, system_prompt=None, temperature=0.7, max_tokens=4096, json_mode=False) -> BackendResponse:
        self._ensure_initialized()

        b64_data, mime_type = self._load_image_as_base64(image_path)
        image_url = f"data:{mime_type};base64,{b64_data}"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Multimodal message — image + text together
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
                {"type": "text", "text": prompt},
            ],
        })

        # Auto-pick vision model if needed
        vision_model = self.model
        if "vision" not in vision_model and "grok-2" in vision_model:
            vision_model = "grok-2-vision-latest"

        kwargs = {"model": vision_model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(**kwargs)

        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return BackendResponse(text=response.choices[0].message.content or "", model=vision_model, backend="grok", usage=usage, raw_response=response)

    async def is_available(self) -> bool:
        try:
            self._ensure_initialized()
            self._client.models.list()
            return True
        except Exception:
            return False
