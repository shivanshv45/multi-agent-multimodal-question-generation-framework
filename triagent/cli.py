# CLI — run `python -m triagent run --image photo.jpg` or `python -m triagent check`

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from triagent.config import load_config
from triagent.pipeline import TriAgentPipeline

console = Console()


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
@click.option("--image", "-i", type=click.Path(exists=True), required=True)
@click.option("--backend", "-b", type=click.Choice(["gemini", "grok", "ollama", "openai", "groq"]), default=None)
@click.option("--model", "-m", type=str, default=None)
@click.option("--context", "-c", type=str, default="")
@click.option("--output", "-o", type=click.Path(), default="./output")
def run(image, backend, model, context, output):
    _print_banner()

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

    result = asyncio.run(pipeline.run(image, additional_context=context))
    console.print("\n[bold green]✓ Pipeline completed successfully![/bold green]")


@cli.command()
@click.option("--image-dir", "-d", type=click.Path(exists=True), required=True)
@click.option("--backend", "-b", type=click.Choice(["gemini", "grok", "ollama", "openai", "groq"]), default=None)
@click.option("--output", "-o", type=click.Path(), default="./output")
@click.option("--export", "-e", type=click.Path(), default=None)
def batch(image_dir, backend, output, export):
    _print_banner()

    img_dir = Path(image_dir)
    extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
    images = [f for f in img_dir.iterdir() if f.suffix.lower() in extensions]

    if not images:
        console.print(f"[red]No images found in {img_dir}[/red]")
        sys.exit(1)

    console.print(f"[cyan]Found {len(images)} images in {img_dir}[/cyan]")

    overrides = {"output_dir": output}
    if backend:
        overrides["visual_backend"] = backend
        overrides["reasoning_backend"] = backend
        overrides["synthesis_backend"] = backend

    config = load_config(**overrides)
    pipeline = TriAgentPipeline.from_config(config)
    results = asyncio.run(pipeline.run_batch(images))

    if export:
        pipeline.export_dataset(export)

    console.print(f"\n[bold green]✓ Generated {len(results)} benchmark items![/bold green]")


@cli.command()
def check():
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
    assignments[config.agents.visual.backend].append(f"Visual ({config.agents.visual.model})")
    assignments[config.agents.reasoning.backend].append(f"Reasoning ({config.agents.reasoning.model})")
    assignments[config.agents.synthesis.backend].append(f"Synthesis ({config.agents.synthesis.model})")

    for backend in ("gemini", "grok", "ollama", "openai", "groq"):
        key_ok = key_status.get(backend, config.keys.validate(backend))
        key_str = "[green]✓ Configured[/green]" if key_ok else "[red]✗ Missing[/red]"
        agents = ", ".join(assignments.get(backend, [])) or "[dim]Not assigned[/dim]"
        table.add_row(backend.title(), key_str, agents)

    console.print(table)

    console.print("\n[cyan]Testing connectivity...[/cyan]")
    for backend_name in config.get_active_backends():
        try:
            from triagent.backends import create_backend
            key = ""
            if backend_name == "gemini": key = config.keys.gemini
            elif backend_name == "grok": key = config.keys.xai
            elif backend_name == "openai": key = config.keys.openai
            elif backend_name == "groq": key = config.keys.groq
            
            b = create_backend(backend_name, api_key=key)
            available = asyncio.run(b.is_available())
            status = "[green]✓ Reachable[/green]" if available else "[red]✗ Unreachable[/red]"
        except Exception as e:
            status = f"[red]✗ Error: {e}[/red]"
        console.print(f"  {backend_name}: {status}")


if __name__ == "__main__":
    cli()
