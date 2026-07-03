# Backend clients — one per model provider
from triagent.backends.base import ModelBackend, BackendResponse
from triagent.backends.gemini import GeminiBackend
from triagent.backends.grok import GrokBackend
from triagent.backends.ollama import OllamaBackend

__all__ = [
    "ModelBackend",
    "BackendResponse",
    "GeminiBackend",
    "GrokBackend",
    "OllamaBackend",
]


def create_backend(backend_name: str, **kwargs) -> ModelBackend:
    # Pick the right backend class by name
    backends = {
        "gemini": GeminiBackend,
        "grok": GrokBackend,
        "ollama": OllamaBackend,
    }
    if backend_name not in backends:
        raise ValueError(
            f"Unknown backend '{backend_name}'. "
            f"Available: {list(backends.keys())}"
        )
    return backends[backend_name](**kwargs)
