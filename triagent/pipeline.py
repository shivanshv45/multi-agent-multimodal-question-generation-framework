# Runs all 3 agents in sequence: Media → IR → Reasoning → MCQ
# Supports image, video, and audio inputs

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from triagent.config import PipelineConfig, load_config
from triagent.schemas import VisualIR, VideoIR, AudioIR, ReasoningOutput, BenchmarkItem
from triagent.backends import create_backend
from triagent.agents import (
    VisualContextAgent,
    VideoContextAgent,
    AudioContextAgent,
    ReasoningAgent,
    SynthesisAgent,
)

console = Console()

# File extensions for each media type
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mpeg", ".mov", ".avi", ".flv", ".mpg", ".webm", ".wmv", ".3gp"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".aac", ".ogg", ".aiff", ".m4a", ".wma"}


def detect_media_type(path: Path) -> str:
    """Auto-detect media type from file extension."""
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    elif suffix in VIDEO_EXTENSIONS:
        return "video"
    elif suffix in AUDIO_EXTENSIONS:
        return "audio"
    else:
        raise ValueError(
            f"Unknown file type '{suffix}' for {path.name}. "
            f"Supported: images ({', '.join(IMAGE_EXTENSIONS)}), "
            f"videos ({', '.join(VIDEO_EXTENSIONS)}), "
            f"audio ({', '.join(AUDIO_EXTENSIONS)})"
        )


class TriAgentPipeline:

    def __init__(self, visual_agent, video_agent, audio_agent, reasoning_agent, synthesis_agent, config):
        self.visual_agent = visual_agent
        self.video_agent = video_agent
        self.audio_agent = audio_agent
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

        # Video and audio agents use the same backend as the visual agent
        # (since they need multimodal capabilities — Gemini is recommended)
        visual_agent = VisualContextAgent(backend=visual_backend, verbose=config.verbose)
        video_agent = VideoContextAgent(backend=visual_backend, verbose=config.verbose)
        audio_agent = AudioContextAgent(backend=visual_backend, verbose=config.verbose)
        reasoning_agent = ReasoningAgent(backend=reasoning_backend, verbose=config.verbose)
        synthesis_agent = SynthesisAgent(backend=synthesis_backend, verbose=config.verbose)

        return cls(visual_agent, video_agent, audio_agent, reasoning_agent, synthesis_agent, config)

    async def run(
        self,
        media_path: str | Path,
        additional_context: str = "",
        base_language: str = "Tamil",
        save_output: bool = True,
        media_type: str | None = None,
    ) -> BenchmarkItem:
        media_path = Path(media_path)
        if not media_path.exists():
            raise FileNotFoundError(f"Media not found: {media_path}")

        # Auto-detect or use provided media type
        if media_type is None:
            media_type = detect_media_type(media_path)

        start_time = time.time()

        console.print()
        console.print(Panel(
            "[bold bright_white]🧠 TRI-AGENT SWARM[/bold bright_white]\n"
            "[dim]Multi-Agent Multimodal Question Generation[/dim]\n"
            f"[dim]{media_type.upper()}: {media_path.name}[/dim]\n"
            f"[dim]Base Language: {base_language}[/dim]",
            title="Pipeline Start",
            border_style="bright_magenta",
            expand=False,
        ))

        visual_ir = None
        video_ir = None
        audio_ir = None

        # Phase 1: Extract structured IR from the media
        if media_type == "image":
            visual_ir = await self.visual_agent.process(
                image_path=media_path, additional_context=additional_context,
            )
        elif media_type == "video":
            video_ir = await self.video_agent.process(
                video_path=media_path, additional_context=additional_context,
            )
        elif media_type == "audio":
            audio_ir = await self.audio_agent.process(
                audio_path=media_path, additional_context=additional_context,
            )
        else:
            raise ValueError(f"Unknown media type: {media_type}")

        # Phase 2: reason about the structured data (never sees raw media)
        reasoning_output = await self.reasoning_agent.process(
            visual_ir=visual_ir, video_ir=video_ir, audio_ir=audio_ir,
        )

        # Phase 3: write the actual question in the target code-mixed language
        benchmark_item = await self.synthesis_agent.process(
            reasoning_output=reasoning_output,
            visual_ir=visual_ir, video_ir=video_ir, audio_ir=audio_ir,
            media_path=str(media_path),
            base_language=base_language,
        )

        elapsed = time.time() - start_time

        self._results.append(benchmark_item)

        if save_output:
            self._save_result(benchmark_item, media_path)

        self._print_summary(benchmark_item, elapsed)

        return benchmark_item

    async def run_batch(
        self,
        media_paths: list[str | Path],
        additional_context: str = "",
        base_language: str = "Tamil",
    ) -> list[BenchmarkItem]:
        results = []
        for i, path in enumerate(media_paths, 1):
            console.print(f"\n{'='*60}")
            console.print(f"[bold]Processing media {i}/{len(media_paths)}[/bold]")
            console.print(f"{'='*60}")
            try:
                result = await self.run(path, additional_context, base_language=base_language)
                results.append(result)
            except Exception as e:
                console.print(f"[red]✗ Failed on {path}: {e}[/red]")
        return results

    def _save_result(self, item: BenchmarkItem, media_path: Path) -> None:
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        # Include the original media filename (without extension) in the output filename
        filename = f"{media_path.stem}_{item.question_id}.json"
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
        table.add_row("Source", f"{item.source_media_type.upper()}")
        table.add_row("Type", item.question_type.value)
        table.add_row("Difficulty", f"{'⭐' * item.difficulty} ({item.difficulty}/5)")
        table.add_row("Question (EN)", item.question_stem_english[:100] + "...")
        table.add_row(f"Question (Pure {item.base_language})", item.question_stem_pure[:100] + "...")
        table.add_row(f"Question (Mixed)", item.question_stem_mixed[:100] + "...")

        for key in ("A", "B", "C", "D"):
            marker = "✓" if key == item.correct_answer else " "
            table.add_row(f"  [{marker}] {key} (Pure)", item.choices_pure.get(key, ""))
            table.add_row(f"  [{marker}] {key} (Mixed)", item.choices_mixed.get(key, ""))

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
                data = item.model_dump(mode="json", exclude={"visual_ir", "video_ir", "audio_ir", "reasoning_output"})
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        console.print(f"[green]✓ Exported {len(self._results)} items to {path}[/green]")
