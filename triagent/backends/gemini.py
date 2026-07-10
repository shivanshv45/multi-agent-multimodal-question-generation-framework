# Google Gemini — supports text, vision (image+text), video, and audio natively

from __future__ import annotations

import os
import time
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

    def _extract_usage(self, response) -> dict:
        """Extract token usage metadata from a Gemini response."""
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            um = response.usage_metadata
            return {
                "prompt_tokens": getattr(um, "prompt_token_count", 0),
                "completion_tokens": getattr(um, "candidates_token_count", 0),
                "total_tokens": getattr(um, "total_token_count", 0),
            }
        return {}

    def _build_config(self, system_prompt, temperature, max_tokens, json_mode):
        """Build a GenerateContentConfig from common parameters."""
        from google.genai import types

        config_kwargs = {"temperature": temperature, "max_output_tokens": max_tokens}
        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"

        return types.GenerateContentConfig(**config_kwargs)

    def _upload_and_wait(self, file_path: str | Path, poll_interval: float = 3.0, max_wait: float = 300.0) -> object:
        """Upload a file via the Gemini File API and poll until processing completes.

        This is the official Google-recommended approach for video and audio files.
        The File API uploads the file to Google's servers, processes it (extracting
        frames for video, decoding audio), and returns a file reference that can be
        passed directly into generate_content.

        Files are stored for 48 hours and support up to 2GB per file.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        uploaded_file = self._client.files.upload(file=str(path))

        elapsed = 0.0
        while uploaded_file.state and uploaded_file.state.name == "PROCESSING":
            if elapsed >= max_wait:
                raise TimeoutError(
                    f"File {path.name} still processing after {max_wait}s. "
                    "Try a smaller file or increase max_wait."
                )
            time.sleep(poll_interval)
            elapsed += poll_interval
            uploaded_file = self._client.files.get(name=uploaded_file.name)

        if uploaded_file.state and uploaded_file.state.name == "FAILED":
            raise RuntimeError(f"File processing failed for {path.name}")

        return uploaded_file

    # ── Text-only ──

    async def generate(self, prompt, system_prompt=None, temperature=0.7, max_tokens=4096, json_mode=False) -> BackendResponse:
        self._ensure_initialized()

        config = self._build_config(system_prompt, temperature, max_tokens, json_mode)
        response = self._client.models.generate_content(
            model=self.model, contents=prompt, config=config,
        )

        return BackendResponse(
            text=response.text or "", model=self.model, backend="gemini",
            usage=self._extract_usage(response), raw_response=response,
        )

    # ── Image ──

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

        config = self._build_config(system_prompt, temperature, max_tokens, json_mode)
        response = self._client.models.generate_content(
            model=self.model, contents=contents, config=config,
        )

        return BackendResponse(
            text=response.text or "", model=self.model, backend="gemini",
            usage=self._extract_usage(response), raw_response=response,
        )

    # ── Video (native via File API) ──

    async def generate_with_video(self, prompt, video_path, system_prompt=None, temperature=0.7, max_tokens=4096, json_mode=False) -> BackendResponse:
        """Upload a video via Gemini File API, wait for processing, then generate.

        Gemini natively understands video — it samples at 1 FPS for visual frames
        and also processes the audio track. This means it can answer questions about
        both what is seen AND what is heard in the video.

        Supported formats: mp4, mpeg, mov, avi, x-flv, mpg, webm, wmv, 3gpp
        """
        self._ensure_initialized()

        video_file = self._upload_and_wait(video_path)
        contents = [video_file, prompt]

        config = self._build_config(system_prompt, temperature, max_tokens, json_mode)
        response = self._client.models.generate_content(
            model=self.model, contents=contents, config=config,
        )

        return BackendResponse(
            text=response.text or "", model=self.model, backend="gemini",
            usage=self._extract_usage(response), raw_response=response,
        )

    # ── Audio (native via File API) ──

    async def generate_with_audio(self, prompt, audio_path, system_prompt=None, temperature=0.7, max_tokens=4096, json_mode=False) -> BackendResponse:
        """Upload an audio file via Gemini File API, wait for processing, then generate.

        Gemini natively understands audio — it can transcribe, translate, summarize,
        and answer questions about audio content including music, speech, and effects.

        Supported formats: wav, mp3, aiff, aac, ogg, flac
        """
        self._ensure_initialized()

        audio_file = self._upload_and_wait(audio_path)
        contents = [audio_file, prompt]

        config = self._build_config(system_prompt, temperature, max_tokens, json_mode)
        response = self._client.models.generate_content(
            model=self.model, contents=contents, config=config,
        )

        return BackendResponse(
            text=response.text or "", model=self.model, backend="gemini",
            usage=self._extract_usage(response), raw_response=response,
        )

    async def is_available(self) -> bool:
        try:
            self._ensure_initialized()
            self._client.models.list()
            return True
        except Exception:
            return False
