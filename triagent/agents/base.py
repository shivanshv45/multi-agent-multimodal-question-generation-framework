# Base class all 3 agents inherit from — handles logging, calling the model, parsing JSON

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel

from triagent.backends.base import ModelBackend, BackendResponse

console = Console()


class BaseAgent(ABC):

    def __init__(self, backend: ModelBackend, verbose: bool = True):
        self.backend = backend
        self.verbose = verbose
        self._execution_log: list[dict] = []

    @property
    @abstractmethod
    def agent_name(self) -> str: ...

    @property
    @abstractmethod
    def agent_role(self) -> str: ...

    @property
    @abstractmethod
    def phase(self) -> int: ...

    @property
    @abstractmethod
    def system_prompt(self) -> str: ...

    def log(self, message: str, style: str = "bold cyan") -> None:
        if self.verbose:
            console.print(f"  [{self.agent_name}] {message}", style=style)

    def log_phase_start(self, input_summary: str = "") -> None:
        if self.verbose:
            console.print()
            console.print(Panel(
                f"[bold]Phase {self.phase}: {self.agent_name}[/bold]\n"
                f"[dim]{self.agent_role}[/dim]\n"
                f"[dim]Backend: {self.backend.name} ({self.backend.model})[/dim]"
                + (f"\n[dim]Input: {input_summary}[/dim]" if input_summary else ""),
                title=f"🔄 Agent Phase {self.phase}",
                border_style="bright_blue",
                expand=False,
            ))

    def log_phase_complete(self, output_summary: str = "") -> None:
        if self.verbose:
            console.print(Panel(
                f"[bold green]✓ Phase {self.phase} Complete[/bold green]\n"
                f"[dim]{output_summary}[/dim]",
                border_style="green",
                expand=False,
            ))

    async def _call_backend(self, prompt, image_path=None, temperature=0.4, max_tokens=4096, json_mode=True, system_prompt_override: str | None = None) -> BackendResponse:
        start_time = time.time()
        active_system_prompt = system_prompt_override or self.system_prompt

        try:
            if image_path:
                self.log(f"Sending image + prompt to {self.backend.name}...")
                response = await self.backend.generate_with_image(
                    prompt=prompt, image_path=image_path, system_prompt=active_system_prompt,
                    temperature=temperature, max_tokens=max_tokens, json_mode=json_mode,
                )
            else:
                self.log(f"Sending prompt to {self.backend.name}...")
                response = await self.backend.generate(
                    prompt=prompt, system_prompt=active_system_prompt,
                    temperature=temperature, max_tokens=max_tokens, json_mode=json_mode,
                )

            elapsed = time.time() - start_time
            self.log(f"Response received in {elapsed:.1f}s ({response.usage.get('total_tokens', '?')} tokens)", style="dim green")

            self._execution_log.append({
                "agent": self.agent_name, "phase": self.phase, "backend": self.backend.name,
                "model": response.model, "elapsed_seconds": round(elapsed, 2),
                "tokens": response.usage, "success": True,
            })

            return response

        except Exception as e:
            elapsed = time.time() - start_time
            self.log(f"❌ Error after {elapsed:.1f}s: {e}", style="bold red")
            self._execution_log.append({
                "agent": self.agent_name, "phase": self.phase, "backend": self.backend.name,
                "elapsed_seconds": round(elapsed, 2), "success": False, "error": str(e),
            })
            raise

    def _parse_json_response(self, text: str) -> dict:
        # LLMs sometimes wrap JSON in ```json ... ``` — strip that
        cleaned = text.strip()

        if cleaned.startswith("```"):
            first_newline = cleaned.index("\n")
            cleaned = cleaned[first_newline + 1:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            self.log("⚠ JSON parse failed, trying to recover...", style="yellow")
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(cleaned[start:end])
                except json.JSONDecodeError:
                    pass
            raise ValueError(f"Failed to parse JSON from {self.agent_name}: {e}\nRaw: {text[:500]}...") from e

    def get_execution_log(self) -> list[dict]:
        return self._execution_log.copy()
