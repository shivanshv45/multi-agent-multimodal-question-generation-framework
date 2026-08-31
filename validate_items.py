#!/usr/bin/env python3
"""
validate_items.py — standalone schema-conformance checker for Tri-Agent Swarm output.

Loads every JSON file under output/ (or a given path) and re-validates it
against the project's actual `BenchmarkItem` Pydantic model from
triagent/schemas.py. This is the ground-truth schema the pipeline itself
writes against, so this catches real drift: a backend that changed its
output shape, a hand-edited file, a QuestionType/DistractorStrategy enum
that no longer matches, etc.

This script only imports triagent.schemas (data models, no API calls) and
reads files under output/. It never runs the pipeline, calls a backend, or
writes anything.

Usage:
    python validate_items.py
    python validate_items.py --path output/tamil
    python validate_items.py --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from triagent.schemas import BenchmarkItem


def find_json_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.json") if p.name != ".gitkeep")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate output JSON files against the real BenchmarkItem schema.")
    parser.add_argument("--path", default="output", help="File or directory to validate (default: output)")
    parser.add_argument("--verbose", action="store_true", help="Show full validation error details")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    target = (project_root / args.path).resolve()

    if target.is_file():
        files = [target]
    elif target.is_dir():
        files = find_json_files(target)
    else:
        print(f"Path not found: {target}")
        return 1

    if not files:
        print(f"No JSON files found under {target}")
        return 0

    passed = 0
    failed = 0

    print("=" * 60)
    print("Tri-Agent Swarm - Schema Validation (against triagent.schemas.BenchmarkItem)")
    print("=" * 60)

    for path in files:
        rel = path.relative_to(project_root) if project_root in path.parents else path
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"[PARSE ERROR] {rel}: {e}")
            failed += 1
            continue

        try:
            BenchmarkItem.model_validate(raw)
            print(f"[OK]   {rel}")
            passed += 1
        except ValidationError as e:
            print(f"[FAIL] {rel}: {e.error_count()} validation error(s)")
            if args.verbose:
                for err in e.errors():
                    loc = ".".join(str(x) for x in err["loc"])
                    print(f"         - {loc}: {err['msg']}")
            failed += 1

    print("-" * 60)
    print(f"Total: {len(files)}  Passed: {passed}  Failed: {failed}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
