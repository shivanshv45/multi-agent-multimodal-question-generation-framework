# Demo — runs the full pipeline with pretty output for any media type

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
import json

from triagent.config import load_config
from triagent.pipeline import TriAgentPipeline, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, AUDIO_EXTENSIONS

console = Console()

ALL_MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS | AUDIO_EXTENSIONS


DEMO_BANNER = """
[bold bright_magenta]╔══════════════════════════════════════════════════════════╗
║         TRI-AGENT SWARM — LIVE DEMONSTRATION            ║
║   Multi-Agent Multimodal Question Generation Framework   ║
║                                                          ║
║   NIT Tiruchirappalli | CSE Department                   ║
║   Supervisor: Dr. A. Santhanavijayan                     ║
╚══════════════════════════════════════════════════════════╝[/bold bright_magenta]
"""


async def run_demo(media_path=None, backend="gemini", model=None, language="Tanglish (Tamil-English)"):
    console.print(DEMO_BANNER)

    # Find a media file to use
    if media_path:
        media = Path(media_path)
        if not media.exists():
            console.print(f"[red]Media not found: {media}[/red]")
            sys.exit(1)
    else:
        # Try to find sample media
        sample_dir = Path("data/sample_images")
        if sample_dir.exists():
            media_files = [f for f in sample_dir.iterdir() if f.suffix.lower() in ALL_MEDIA_EXTENSIONS]
            if media_files:
                media = media_files[0]
                console.print(f"[cyan]Using sample media: {media}[/cyan]")
            else:
                console.print("[red]No sample media found in data/sample_images/[/red]")
                console.print("[yellow]Please provide a media file:[/yellow]")
                console.print("  python -m triagent.demo --media path/to/file")
                sys.exit(1)
        else:
            console.print("[cyan]Downloading sample image...[/cyan]")
            media = await _download_sample_image()
            if not media:
                console.print("[red]Could not download sample image.[/red]")
                console.print("[yellow]Please provide a media file manually:[/yellow]")
                console.print("  python -m triagent.demo --media path/to/file")
                sys.exit(1)

    # Detect and display media type
    suffix = media.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        media_emoji, media_label = "📸", "Image"
    elif suffix in VIDEO_EXTENSIONS:
        media_emoji, media_label = "🎬", "Video"
    elif suffix in AUDIO_EXTENSIONS:
        media_emoji, media_label = "🎵", "Audio"
    else:
        media_emoji, media_label = "📄", "Unknown"

    console.print(f"\n[bold]{media_emoji} {media_label}: {media.name}[/bold]")

    # Set up the pipeline
    console.print(Panel(
        "[bold]Initializing Tri-Agent Swarm...[/bold]\n"
        f"[dim]Backend: {backend}[/dim]\n"
        f"[dim]Media: {media_label}[/dim]\n"
        f"[dim]Language: {language}[/dim]",
        border_style="cyan",
        expand=False,
    ))

    overrides = {"visual_backend": backend, "reasoning_backend": backend, "synthesis_backend": backend}
    if model:
        overrides["visual_model"] = model
        overrides["reasoning_model"] = model
        overrides["synthesis_model"] = model

    config = load_config(**overrides)

    # Check API keys before running
    key_status = config.validate_keys()
    if not all(key_status.values()):
        for name, ok in key_status.items():
            if not ok:
                console.print(f"  [red]✗ {name}: API key missing![/red]")
        console.print("\n[yellow]Set your API keys in .env file:[/yellow]")
        console.print("  copy .env.example .env")
        console.print("  # Then edit .env with your keys")
        sys.exit(1)

    pipeline = TriAgentPipeline.from_config(config)

    # Run it
    console.print("\n[bold bright_green]▶ Running Pipeline...[/bold bright_green]\n")

    result = await pipeline.run(
        media_path=media,
        additional_context="South Indian cultural context, Tamil Nadu traditions",
        target_language=language,
        save_output=True,
    )

    # Show detailed results from each phase
    console.print("\n" + "=" * 60)
    console.print("[bold bright_green]✓ PIPELINE COMPLETE — FULL RESULTS[/bold bright_green]")
    console.print("=" * 60)

    # Display the appropriate IR based on media type
    if result.visual_ir:
        ir_json = json.dumps(result.visual_ir.model_dump(mode="json"), indent=2, ensure_ascii=False)
        console.print(Panel(Syntax(ir_json, "json", theme="monokai"), title="Phase 1: Visual IR", border_style="blue"))

    if result.video_ir:
        ir_json = json.dumps(result.video_ir.model_dump(mode="json"), indent=2, ensure_ascii=False)
        console.print(Panel(Syntax(ir_json, "json", theme="monokai"), title="Phase 1: Video IR", border_style="blue"))

    if result.audio_ir:
        ir_json = json.dumps(result.audio_ir.model_dump(mode="json"), indent=2, ensure_ascii=False)
        console.print(Panel(Syntax(ir_json, "json", theme="monokai"), title="Phase 1: Audio IR", border_style="blue"))

    if result.reasoning_output:
        ro_json = json.dumps(result.reasoning_output.model_dump(mode="json"), indent=2, ensure_ascii=False)
        console.print(Panel(Syntax(ro_json, "json", theme="monokai"), title="Phase 2: Reasoning Output", border_style="yellow"))

    mcq_data = result.model_dump(mode="json", exclude={"visual_ir", "video_ir", "audio_ir", "reasoning_output"})
    mcq_json = json.dumps(mcq_data, indent=2, ensure_ascii=False)
    console.print(Panel(Syntax(mcq_json, "json", theme="monokai"), title="Phase 3: Final Benchmark Item (MCQ)", border_style="green"))

    console.print("\n[bold bright_green]🎉 Demo complete![/bold bright_green]")
    return result


async def _download_sample_image() -> Path | None:
    import httpx

    sample_dir = Path("data/sample_images")
    sample_dir.mkdir(parents=True, exist_ok=True)

    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/Kolam3.JPG/640px-Kolam3.JPG"
    target = sample_dir / "kolam_sample.jpg"

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                target.write_bytes(resp.content)
                console.print(f"[green]✓ Downloaded sample: {target}[/green]")
                return target
            else:
                console.print(f"[yellow]Download returned {resp.status_code}[/yellow]")
                return None
    except Exception as e:
        console.print(f"[yellow]Download failed: {e}[/yellow]")
        return None


@click.command()
@click.option("--image", "-i", type=click.Path(), default=None, help="Image file (legacy alias)")
@click.option("--media", type=click.Path(), default=None, help="Media file: image, video, or audio")
@click.option("--backend", "-b", type=click.Choice(["gemini", "grok", "ollama"]), default="gemini")
@click.option("--model", "-m", type=str, default=None)
@click.option("--language", "-l", type=str, default="Tanglish (Tamil-English)", help="Target code-mixed language")
def main(image, media, backend, model, language):
    path = media or image
    asyncio.run(run_demo(media_path=path, backend=backend, model=model, language=language))


if __name__ == "__main__":
    main()
