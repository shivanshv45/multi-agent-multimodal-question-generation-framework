# Loads settings from .env and wires up which backend goes to which agent

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def _find_env_file() -> Optional[Path]:
    current = Path.cwd()
    for parent in [current, *current.parents]:
        env_path = parent / ".env"
        if env_path.exists():
            return env_path
    return None


_env_file = _find_env_file()
if _env_file:
    load_dotenv(_env_file)


@dataclass
class BackendConfig:
    backend: str  # "gemini", "grok", "ollama"
    model: str

    @property
    def is_cloud(self) -> bool:
        return self.backend in ("gemini", "grok")

    @property
    def is_local(self) -> bool:
        return self.backend == "ollama"


@dataclass
class AgentConfig:
    # Each agent can use a different backend + model combo
    visual: BackendConfig = field(default_factory=lambda: BackendConfig(
        backend=os.getenv("VISUAL_AGENT_BACKEND", "gemini"),
        model=os.getenv("VISUAL_AGENT_MODEL", "gemini-2.5-flash"),
    ))
    reasoning: BackendConfig = field(default_factory=lambda: BackendConfig(
        backend=os.getenv("REASONING_AGENT_BACKEND", "gemini"),
        model=os.getenv("REASONING_AGENT_MODEL", "gemini-2.5-flash"),
    ))
    synthesis: BackendConfig = field(default_factory=lambda: BackendConfig(
        backend=os.getenv("SYNTHESIS_AGENT_BACKEND", "gemini"),
        model=os.getenv("SYNTHESIS_AGENT_MODEL", "gemini-2.5-flash"),
    ))


@dataclass
class APIKeys:
    gemini: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    xai: str = field(default_factory=lambda: os.getenv("XAI_API_KEY", ""))
    openai: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    groq: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    ollama_host: str = field(
        default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434")
    )

    def validate(self, backend: str) -> bool:
        if backend == "gemini":
            return bool(self.gemini) and self.gemini != "your_gemini_api_key_here"
        elif backend == "grok":
            return bool(self.xai) and self.xai != "your_xai_api_key_here"
        elif backend == "openai":
            return bool(self.openai) and self.openai != "your_openai_api_key_here"
        elif backend == "groq":
            return bool(self.groq) and self.groq != "your_groq_api_key_here"
        elif backend == "ollama":
            return True
        return False


@dataclass
class PipelineConfig:
    agents: AgentConfig = field(default_factory=AgentConfig)
    keys: APIKeys = field(default_factory=APIKeys)
    output_dir: Path = field(
        default_factory=lambda: Path(os.getenv("OUTPUT_DIR", "./output"))
    )
    verbose: bool = True

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_active_backends(self) -> set[str]:
        return {
            self.agents.visual.backend,
            self.agents.reasoning.backend,
            self.agents.synthesis.backend,
        }

    def validate_keys(self) -> dict[str, bool]:
        results = {}
        for backend in self.get_active_backends():
            results[backend] = self.keys.validate(backend)
        return results


def load_config(**overrides) -> PipelineConfig:
    config = PipelineConfig()

    if "visual_backend" in overrides:
        config.agents.visual.backend = overrides["visual_backend"]
    if "visual_model" in overrides:
        config.agents.visual.model = overrides["visual_model"]
    if "reasoning_backend" in overrides:
        config.agents.reasoning.backend = overrides["reasoning_backend"]
    if "reasoning_model" in overrides:
        config.agents.reasoning.model = overrides["reasoning_model"]
    if "synthesis_backend" in overrides:
        config.agents.synthesis.backend = overrides["synthesis_backend"]
    if "synthesis_model" in overrides:
        config.agents.synthesis.model = overrides["synthesis_model"]
    if "gemini_key" in overrides:
        config.keys.gemini = overrides["gemini_key"]
    if "xai_key" in overrides:
        config.keys.xai = overrides["xai_key"]
    if "openai_key" in overrides:
        config.keys.openai = overrides["openai_key"]
    if "groq_key" in overrides:
        config.keys.groq = overrides["groq_key"]
    if "verbose" in overrides:
        config.verbose = overrides["verbose"]

    return config
