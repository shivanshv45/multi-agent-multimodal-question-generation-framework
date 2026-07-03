# Every backend (Gemini, Grok, Ollama) implements this same interface

from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class BackendResponse:
    text: str
    model: str
    backend: str
    usage: dict = field(default_factory=dict)
    raw_response: Optional[object] = None


class ModelBackend(ABC):

    def __init__(self, api_key: str = "", model: str = "", **kwargs):
        self.api_key = api_key
        self.model = model
        self._initialized = False

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def initialize(self) -> None: ...

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            self.initialize()
            self._initialized = True

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> BackendResponse: ...

    @abstractmethod
    async def generate_with_image(
        self,
        prompt: str,
        image_path: str | Path,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> BackendResponse: ...

    @abstractmethod
    async def is_available(self) -> bool: ...

    @staticmethod
    def _load_image_as_base64(image_path: str | Path) -> tuple[str, str]:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        suffix = path.suffix.lower()
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        mime_type = mime_map.get(suffix, "image/jpeg")

        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")

        return data, mime_type

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} model={self.model!r}>"
