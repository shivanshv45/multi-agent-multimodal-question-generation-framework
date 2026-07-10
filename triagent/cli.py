# CLI — run `python -m triagent run --media photo.jpg` or `python -m triagent check`

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from triagent.config import load_config
from triagent.pipeline import TriAgentPipeline, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, AUDIO_EXTENSIONS

console = Console()

ALL_MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS | AUDIO_EXTENSIONS


def _print_banner():
    console.print(Panel(
        "[bold bright_magenta]"
        "╔╦╗╦═╗╦  ╔═╗╔═╗╔═╗╔╗╔╔╦╗  ╔═╗╦ ╦╔═╗╦═╗╔╦╗\n"
        " ║ ╠╦╝║  ╠═╣║ ╦║╣ ║║║ ║   ╚═╗║║║╠═╣╠╦╝║║║\n"
        " ╩ ╩╚═╩  ╩ ╩╚═╝╚═╝╝╚╝ ╩   ╚═╝╚╩╝╩ ╩╩╚═╩ ╩\n"
        "[/bold bright_magenta]"
        "[dim]Multi-Agent Multimodal Question Generation Framework[/dim]\n"
        "[dim]NIT Tiruchirappalli | Dr. A. Santhanavijayan[/dim]",
        border_style="bright_magenta",
        expand=False,
    ))


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    if ctx.invoked_subcommand is None:
        _print_banner()
        click.echo(ctx.get_help())


@cli.command()
@click.option("--image", "-i", type=click.Path(exists=True), default=None, help="Image file (legacy alias for --media)")
@click.option("--media", type=click.Path(exists=True), default=None, help="Media file: image, video, or audio")
@click.option("--backend", "-b", type=click.Choice(["gemini", "grok", "ollama"]), default=None)
@click.option("--model", "-m", type=str, default=None)
@click.option("--context", "-c", type=str, default="")
@click.option("--language", "-l", type=str, default="Tanglish (Tamil-English)", help="Target code-mixed language")
@click.option("--output", "-o", type=click.Path(), default="./output")
def run(image, media, backend, model, context, language, output):
    """Process a single image, video, or audio file into a benchmark MCQ."""
    _print_banner()

    # Support both --image (legacy) and --media (new)
    media_path = media or image
    if not media_path:
        console.print("[red]Please provide a media file via --media or --image[/red]")
        sys.exit(1)

    overrides = {"output_dir": output}
    if backend:
        overrides["visual_backend"] = backend
        overrides["reasoning_backend"] = backend
        overrides["synthesis_backend"] = backend
    if model:
        overrides["visual_model"] = model
        overrides["reasoning_model"] = model
        overrides["synthesis_model"] = model

    config = load_config(**overrides)
    pipeline = TriAgentPipeline.from_config(config)

    result = asyncio.run(pipeline.run(media_path, additional_context=context, target_language=language))
    console.print("\n[bold green]✓ Pipeline completed successfully![/bold green]")


@cli.command()
@click.option("--media-dir", "-d", type=click.Path(exists=True), required=True, help="Directory of media files")
@click.option("--backend", "-b", type=click.Choice(["gemini", "grok", "ollama"]), default=None)
@click.option("--language", "-l", type=str, default="Tanglish (Tamil-English)", help="Target code-mixed language")
@click.option("--output", "-o", type=click.Path(), default="./output")
@click.option("--export", "-e", type=click.Path(), default=None)
def batch(media_dir, backend, language, output, export):
    """Batch process all media files (images, videos, audio) in a directory."""
    _print_banner()

    dir_path = Path(media_dir)
    media_files = [f for f in dir_path.iterdir() if f.suffix.lower() in ALL_MEDIA_EXTENSIONS]

    if not media_files:
        console.print(f"[red]No supported media files found in {dir_path}[/red]")
        sys.exit(1)

    # Categorize for the user
    images = [f for f in media_files if f.suffix.lower() in IMAGE_EXTENSIONS]
    videos = [f for f in media_files if f.suffix.lower() in VIDEO_EXTENSIONS]
    audios = [f for f in media_files if f.suffix.lower() in AUDIO_EXTENSIONS]
    console.print(
        f"[cyan]Found {len(media_files)} media files: "
        f"{len(images)} images, {len(videos)} videos, {len(audios)} audio[/cyan]"
    )

    overrides = {"output_dir": output}
    if backend:
        overrides["visual_backend"] = backend
        overrides["reasoning_backend"] = backend
        overrides["synthesis_backend"] = backend

    config = load_config(**overrides)
    pipeline = TriAgentPipeline.from_config(config)
    results = asyncio.run(pipeline.run_batch(media_files, target_language=language))

    if export:
        pipeline.export_dataset(export)

    console.print(f"\n[bold green]✓ Generated {len(results)} benchmark items![/bold green]")


@cli.command()
def check():
    """Check backend configuration and connectivity."""
    _print_banner()

    config = load_config()
    table = Table(title="Backend Status", border_style="cyan")
    table.add_column("Backend", style="cyan")
    table.add_column("API Key", style="white")
    table.add_column("Agent Assignment", style="white")

    key_status = config.validate_keys()

    assignments = {
        config.agents.visual.backend: [],
        config.agents.reasoning.backend: [],
        config.agents.synthesis.backend: [],
    }
    assignments[config.agents.visual.backend].append(f"Visual+Video+Audio ({config.agents.visual.model})")
    assignments[config.agents.reasoning.backend].append(f"Reasoning ({config.agents.reasoning.model})")
    assignments[config.agents.synthesis.backend].append(f"Synthesis ({config.agents.synthesis.model})")

    for backend in ("gemini", "grok", "ollama"):
        key_ok = key_status.get(backend, config.keys.validate(backend))
        key_str = "[green]✓ Configured[/green]" if key_ok else "[red]✗ Missing[/red]"
        agents = ", ".join(assignments.get(backend, [])) or "[dim]Not assigned[/dim]"
        table.add_row(backend.title(), key_str, agents)

    console.print(table)

    # Show supported media types
    console.print("\n[cyan]Supported media types:[/cyan]")
    console.print(f"  Images: {', '.join(sorted(IMAGE_EXTENSIONS))}")
    console.print(f"  Videos: {', '.join(sorted(VIDEO_EXTENSIONS))}")
    console.print(f"  Audio:  {', '.join(sorted(AUDIO_EXTENSIONS))}")

    console.print("\n[cyan]Testing connectivity...[/cyan]")
    for backend_name in config.get_active_backends():
        try:
            from triagent.backends import create_backend
            key = config.keys.gemini if backend_name == "gemini" else config.keys.xai
            b = create_backend(backend_name, api_key=key)
            available = asyncio.run(b.is_available())
            status = "[green]✓ Reachable[/green]" if available else "[red]✗ Unreachable[/red]"
        except Exception as e:
            status = f"[red]✗ Error: {e}[/red]"
        console.print(f"  {backend_name}: {status}")


if __name__ == "__main__":
    cli()
