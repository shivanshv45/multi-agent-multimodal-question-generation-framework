#!/usr/bin/env python3
"""
check_setup.py — standalone pre-flight check for the Tri-Agent Swarm project.

Verifies the environment is actually ready to run `triagent` pipelines,
without running any pipeline or agent itself:

  - required packages importable (from requirements.txt)
  - .env present and loaded
  - which backends are configured per agent (visual/reasoning/synthesis)
  - whether each configured backend's API key looks valid
  - whether each configured backend is actually reachable (is_available())
  - whether the output directory and dataset folders exist

This script only reads config and pings backends' `is_available()` checks
(a lightweight connectivity check the project already defines for each
backend) — it never generates content, calls the pipeline, or writes files.

Usage:
    python check_setup.py
"""

from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path

REQUIRED_PACKAGES = [
    "pydantic",
    "dotenv",
    "httpx",
    "rich",
    "click",
    "PIL",
    "google.genai",
    "ollama",
    "openai",
]


def check_packages() -> list[str]:
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append(pkg)
    return missing


def check_datasets(project_root: Path) -> dict[str, bool]:
    return {
        d.name: any(d.iterdir()) if d.exists() else False
        for d in project_root.glob("dataset_*")
    }


async def check_backend_live(backend_name: str, api_key: str, model: str) -> tuple[bool, str]:
    try:
        from triagent.backends import create_backend
        backend = create_backend(backend_name, api_key=api_key, model=model)
        ok = await backend.is_available()
        return ok, "reachable" if ok else "not reachable"
    except Exception as e:
        return False, f"error: {e}"


async def main() -> int:
    project_root = Path(__file__).resolve().parent
    print("=" * 60)
    print("Tri-Agent Swarm - Setup Check")
    print("=" * 60)

    # 1. Packages
    print("\n-- Dependencies --")
    missing = check_packages()
    if missing:
        print(f"  Missing packages: {', '.join(missing)}")
        print(f"  Run: pip install -r requirements.txt")
    else:
        print("  All required packages are importable.")

    # 2. .env / config
    env_path = project_root / ".env"
    print("\n-- Configuration --")
    if env_path.exists():
        print(f"  .env found at {env_path}")
    else:
        print("  .env not found — copy .env.example to .env and fill in API keys.")
        print("  (Continuing with defaults / environment variables, if any.)")

    try:
        from triagent.config import load_config
    except ImportError as e:
        print(f"\nCannot import triagent.config: {e}")
        print("Make sure dependencies are installed and you're running from the project root.")
        return 1

    config = load_config()

    print("\n-- Agent -> Backend wiring --")
    for agent_name, backend_cfg in (
        ("visual", config.agents.visual),
        ("reasoning", config.agents.reasoning),
        ("synthesis", config.agents.synthesis),
    ):
        print(f"  {agent_name:<10} -> backend={backend_cfg.backend:<8} model={backend_cfg.model}")

    print("\n-- API key validation (format check only) --")
    key_status = config.validate_keys()
    for backend, valid in key_status.items():
        mark = "OK" if valid else "MISSING/INVALID"
        print(f"  {backend:<10}: {mark}")

    # 3. Live reachability check for each active backend
    print("\n-- Live backend reachability --")
    key_map = {
        "gemini": config.keys.gemini,
        "grok": config.keys.xai,
        "openai": config.keys.openai,
        "groq": config.keys.groq,
        "ollama": "",
    }
    model_map = {
        config.agents.visual.backend: config.agents.visual.model,
        config.agents.reasoning.backend: config.agents.reasoning.model,
        config.agents.synthesis.backend: config.agents.synthesis.model,
    }
    for backend_name in config.get_active_backends():
        if not key_status.get(backend_name, False) and backend_name != "ollama":
            print(f"  {backend_name:<10}: skipped (no valid API key)")
            continue
        ok, detail = await check_backend_live(
            backend_name, key_map.get(backend_name, ""), model_map.get(backend_name, "")
        )
        mark = "OK" if ok else "FAIL"
        print(f"  {backend_name:<10}: {mark} ({detail})")

    # 4. Output dir + datasets
    print("\n-- Filesystem --")
    print(f"  output_dir: {config.output_dir} ({'exists' if config.output_dir.exists() else 'will be created on first run'})")
    datasets = check_datasets(project_root)
    if datasets:
        for name, has_files in datasets.items():
            print(f"  {name}: {'has files' if has_files else 'empty or missing files'}")
    else:
        print("  No dataset_* folders found.")

    print("\nDone. This script made no changes and generated no content.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
