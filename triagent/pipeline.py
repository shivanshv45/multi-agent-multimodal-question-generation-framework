# Runs all 3 agents in sequence: Image → Visual IR → Reasoning → MCQ

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from triagent.config import PipelineConfig, load_config
from triagent.schemas import VisualIR, ReasoningOutput, BenchmarkItem
from triagent.backends import create_backend
from triagent.agents import VisualContextAgent, ReasoningAgent, SynthesisAgent

console = Console()


class TriAgentPipeline:

    def __init__(self, visual_agent, reasoning_agent, synthesis_agent, config):
        self.visual_agent = visual_agent
        self.reasoning_agent = reasoning_agent
        self.synthesis_agent = synthesis_agent
        self.config = config
        self._results: list[BenchmarkItem] = []

    @classmethod
    def from_config(cls, config: Optional[PipelineConfig] = None, **overrides):
        if config is None:
            config = load_config(**overrides)

        key_status = config.validate_keys()
        for backend, valid in key_status.items():
            if not valid:
                console.print(f"  ⚠ {backend}: API key not configured", style="yellow")

        # Wire each agent to its backend
        visual_backend = create_backend(
            config.agents.visual.backend,
            api_key=config.keys.gemini if config.agents.visual.backend == "gemini" else config.keys.xai,
            model=config.agents.visual.model,
        )
        reasoning_backend = create_backend(
            config.agents.reasoning.backend,
            api_key=config.keys.gemini if config.agents.reasoning.backend == "gemini" else config.keys.xai,
            model=config.agents.reasoning.model,
        )
        synthesis_backend = create_backend(
            config.agents.synthesis.backend,
            api_key=config.keys.gemini if config.agents.synthesis.backend == "gemini" else config.keys.xai,
            model=config.agents.synthesis.model,
        )

        visual_agent = VisualContextAgent(backend=visual_backend, verbose=config.verbose)
        reasoning_agent = ReasoningAgent(backend=reasoning_backend, verbose=config.verbose)
        synthesis_agent = SynthesisAgent(backend=synthesis_backend, verbose=config.verbose)

        return cls(visual_agent, reasoning_agent, synthesis_agent, config)

    async def run(self, image_path: str | Path, additional_context: str = "", save_output: bool = True) -> BenchmarkItem:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        start_time = time.time()

        console.print()
        console.print(Panel(
            "[bold bright_white]🧠 TRI-AGENT SWARM[/bold bright_white]\n"
            "[dim]Multi-Agent Multimodal Question Generation[/dim]\n"
            f"[dim]Image: {image_path.name}[/dim]",
            title="Pipeline Start",
            border_style="bright_magenta",
            expand=False,
        ))

        # Phase 1: look at the image, extract structured data
        visual_ir = await self.visual_agent.process(
            image_path=image_path, additional_context=additional_context,
        )

        # Phase 2: reason about the structured data (never sees image)
        reasoning_output = await self.reasoning_agent.process(visual_ir=visual_ir)

        # Phase 3: write the actual question in Tamil-English
        benchmark_item = await self.synthesis_agent.process(
            visual_ir=visual_ir, reasoning_output=reasoning_output, image_path=str(image_path),
        )

        elapsed = time.time() - start_time

        self._results.append(benchmark_item)

        if save_output:
            self._save_result(benchmark_item, image_path)

        self._print_summary(benchmark_item, elapsed)

        return benchmark_item

    async def run_batch(self, image_paths: list[str | Path], additional_context: str = "") -> list[BenchmarkItem]:
        results = []
        for i, path in enumerate(image_paths, 1):
            console.print(f"\n{'='*60}")
            console.print(f"[bold]Processing image {i}/{len(image_paths)}[/bold]")
            console.print(f"{'='*60}")
            try:
                result = await self.run(path, additional_context)
                results.append(result)
            except Exception as e:
                console.print(f"[red]✗ Failed on {path}: {e}[/red]")
        return results

    def _save_result(self, item: BenchmarkItem, image_path: Path) -> None:
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{item.question_id}.json"
        output_path = output_dir / filename

        data = item.model_dump(mode="json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        console.print(f"  💾 Saved to: {output_path}", style="dim")

    def _print_summary(self, item: BenchmarkItem, elapsed: float) -> None:
        table = Table(title="Generated Benchmark Item", border_style="bright_blue")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Question ID", item.question_id)
        table.add_row("Type", item.question_type.value)
        table.add_row("Difficulty", f"{'⭐' * item.difficulty} ({item.difficulty}/5)")
        table.add_row("Question (EN)", item.question_stem_english[:100] + "..." if len(item.question_stem_english) > 100 else item.question_stem_english)
        table.add_row("Question (TN-EN)", item.question_stem[:100] + "..." if len(item.question_stem) > 100 else item.question_stem)

        for key in ("A", "B", "C", "D"):
            marker = "✓" if key == item.correct_answer else " "
            table.add_row(f"  [{marker}] {key}", item.choices.get(key, ""))

        table.add_row("Correct Answer", f"[bold green]{item.correct_answer}[/bold green]")
        table.add_row("Cultural Tags", ", ".join(item.cultural_tags))
        table.add_row("Pipeline Time", f"{elapsed:.1f}s")

        console.print()
        console.print(table)

    def get_all_results(self) -> list[BenchmarkItem]:
        return self._results.copy()

    def export_dataset(self, output_path: str | Path) -> None:
        path = Path(output_path)
        with open(path, "w", encoding="utf-8") as f:
            for item in self._results:
                data = item.model_dump(mode="json", exclude={"visual_ir", "reasoning_output"})
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        console.print(f"[green]✓ Exported {len(self._results)} items to {path}[/green]")
