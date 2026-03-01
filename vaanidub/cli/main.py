"""VaaniDub CLI — command-line interface for dubbing, jobs, and model management."""

import asyncio
import uuid
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from vaanidub.constants import LANGUAGES, STAGE_NAMES
from vaanidub.logging_config import setup_logging

app = typer.Typer(
    name="vaanidub",
    help="AI-powered regional dubbing tool for Indian languages",
    no_args_is_help=True,
    callback=lambda: setup_logging(),
)
console = Console()


# ─── Dub Command ───


@app.command()
def dub(
    input_file: Path = typer.Argument(..., help="Input audio/video file path"),
    target: list[str] = typer.Option(
        ..., "--target", "-t", help="Target language codes (hi, ta, te, bn, etc.)"
    ),
    output_dir: Path = typer.Option(
        "./vaanidub_output", "--output", "-o", help="Output directory"
    ),
    source_lang: Optional[str] = typer.Option(
        None, "--source", "-s", help="Source language (auto-detect if omitted)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
):
    """Dub an audio/video file into target Indian languages."""
    # Validate input
    if not input_file.exists():
        console.print(f"[red]Error:[/red] Input file not found: {input_file}")
        raise typer.Exit(1)

    # Validate target languages
    for lang in target:
        if lang not in LANGUAGES:
            console.print(f"[red]Error:[/red] Unsupported language: {lang}")
            console.print(f"Supported: {', '.join(LANGUAGES.keys())}")
            raise typer.Exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    console.print("\n[bold]VaaniDub[/bold] — AI Regional Dubbing")
    console.print(f"  Input:   {input_file}")
    lang_names = ", ".join(LANGUAGES[code].name for code in target)
    console.print(f"  Target:  {lang_names}")
    console.print(f"  Output:  {output_dir}\n")

    # Run the pipeline
    asyncio.run(_run_pipeline(input_file, target, output_dir, source_lang, verbose))


async def _run_pipeline(
    input_file: Path,
    target_languages: list[str],
    output_dir: Path,
    source_lang: str | None,
    verbose: bool,
):
    """Execute the dubbing pipeline with progress display."""
    from vaanidub.config import AppConfig
    from vaanidub.pipeline.context import PipelineContext
    from vaanidub.pipeline.orchestrator import PipelineOrchestrator

    config = AppConfig()
    config.resolve_secrets()
    config.ensure_directories()

    job_id = str(uuid.uuid4())[:8]
    job_dir = config.storage.base_path / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    ctx = PipelineContext(
        job_id=job_id,
        job_dir=job_dir,
        target_languages=target_languages,
        source_language=source_lang,
        input_file_path=input_file,
    )

    # Progress display
    stage_labels = {
        "ingest": "Ingesting media",
        "separate": "Separating vocals from background",
        "diarize": "Identifying speakers",
        "transcribe": "Transcribing speech",
        "prosody": "Analyzing prosody & emotions",
        "translate": "Translating text",
        "synthesize": "Synthesizing dubbed voices",
        "mixdown": "Mixing final output",
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        tasks = {}
        for i, stage_name in enumerate(STAGE_NAMES):
            label = f"[{i+1}/8] {stage_labels[stage_name]}"
            tasks[stage_name] = progress.add_task(label, total=100, visible=True)

        def on_progress(stage: str, percent: int, message: str = ""):
            if stage in tasks:
                progress.update(tasks[stage], completed=percent)
                if message and verbose:
                    progress.console.print(f"      {message}")

        ctx.on_progress = on_progress

        orchestrator = PipelineOrchestrator(config)

        try:
            ctx = await orchestrator.run(ctx)
        except Exception as e:
            console.print(f"\n[red]Error:[/red] {e}")
            raise typer.Exit(1)

    # Copy outputs to output_dir and display results
    console.print("\n[bold green]Done![/bold green] Output files:")
    for lang, path in ctx.final_output_paths.items():
        lang_info = LANGUAGES[lang]
        dest = output_dir / f"{input_file.stem}_{lang}{path.suffix}"

        import shutil
        shutil.copy2(path, dest)
        console.print(f"  {dest}  ({lang_info.name})")

    # Display timing summary
    console.print("\n[dim]Stage Timings:[/dim]")
    for stage_name, elapsed in ctx.stage_timings.items():
        console.print(f"  {stage_name}: {elapsed:.1f}s")


# ─── Jobs Commands ───

jobs_app = typer.Typer(help="Manage dubbing jobs")
app.add_typer(jobs_app, name="jobs")


@jobs_app.command("list")
def jobs_list(
    status: Optional[str] = typer.Option(None, help="Filter by status"),
    api_url: str = typer.Option("http://localhost:8000", "--api-url", envvar="VAANIDUB_API_URL"),
):
    """List recent dubbing jobs."""
    import httpx

    params = {}
    if status:
        params["status"] = status

    try:
        resp = httpx.get(f"{api_url}/api/v1/jobs", params=params, timeout=10)
        resp.raise_for_status()
    except httpx.ConnectError:
        console.print("[red]Error:[/red] Cannot connect to API server.")
        console.print("[dim]Start the server first with: vaanidub serve[/dim]")
        raise typer.Exit(1)
    except httpx.HTTPStatusError as e:
        console.print(f"[red]Error:[/red] API returned {e.response.status_code}")
        raise typer.Exit(1)

    data = resp.json()
    jobs = data if isinstance(data, list) else data.get("jobs", [])

    if not jobs:
        console.print("[dim]No jobs found.[/dim]")
        return

    table = Table(title="Dubbing Jobs")
    table.add_column("ID", style="cyan")
    table.add_column("Status")
    table.add_column("Progress")
    table.add_column("Languages")
    table.add_column("Created")

    for job in jobs:
        status_style = {
            "completed": "[green]completed[/green]",
            "failed": "[red]failed[/red]",
            "processing": "[yellow]processing[/yellow]",
        }.get(job.get("status", ""), job.get("status", ""))

        progress_pct = f"{job.get('progress', 0):.0f}%"
        langs = job.get("target_languages", "")
        if isinstance(langs, list):
            langs = ", ".join(langs)
        created = job.get("created_at", "")[:19] if job.get("created_at") else ""

        table.add_row(job.get("id", "")[:8], status_style, progress_pct, langs, created)

    console.print(table)


@jobs_app.command("status")
def jobs_status(
    job_id: str = typer.Argument(..., help="Job ID"),
    api_url: str = typer.Option("http://localhost:8000", "--api-url", envvar="VAANIDUB_API_URL"),
):
    """Check status of a dubbing job."""
    import httpx

    try:
        resp = httpx.get(f"{api_url}/api/v1/jobs/{job_id}", timeout=10)
    except httpx.ConnectError:
        console.print("[red]Error:[/red] Cannot connect to API server.")
        console.print("[dim]Start the server first with: vaanidub serve[/dim]")
        raise typer.Exit(1)

    if resp.status_code == 404:
        console.print(f"[red]Error:[/red] Job '{job_id}' not found.")
        raise typer.Exit(1)

    resp.raise_for_status()
    job = resp.json()

    console.print(f"\n[bold]Job {job.get('id', job_id)[:8]}[/bold]")
    console.print(f"  Status:   {job.get('status', 'unknown')}")
    console.print(f"  Progress: {job.get('progress', 0):.0f}%")

    if job.get("current_stage"):
        console.print(f"  Stage:    {job['current_stage']}")
    if job.get("source_language"):
        console.print(f"  Source:   {job['source_language']}")

    langs = job.get("target_languages", "")
    if isinstance(langs, list):
        langs = ", ".join(langs)
    console.print(f"  Targets:  {langs}")

    if job.get("error_message"):
        console.print(f"\n  [red]Error:[/red] {job['error_message']}")
        if job.get("error_stage"):
            console.print(f"  [red]Stage:[/red] {job['error_stage']}")

    stage_logs = job.get("stage_logs", [])
    if stage_logs:
        console.print("\n[dim]Stage Logs:[/dim]")
        for log in stage_logs:
            dur = f" ({log['duration_sec']:.1f}s)" if log.get("duration_sec") else ""
            status_icon = {"completed": "[green]OK[/green]", "failed": "[red]FAIL[/red]"}.get(
                log.get("status", ""), log.get("status", "")
            )
            console.print(f"  {log.get('stage_name', '')}: {status_icon}{dur}")

    outputs = job.get("output_paths")
    if outputs and isinstance(outputs, dict):
        console.print("\n[bold green]Outputs:[/bold green]")
        for lang, path in outputs.items():
            console.print(f"  {lang}: {path}")


# ─── Models Commands ───

models_app = typer.Typer(help="Manage AI models")
app.add_typer(models_app, name="models")


@models_app.command("list")
def models_list():
    """List all AI models and their status."""
    from vaanidub.models.model_manager import ModelManager

    manager = ModelManager()
    models = manager.list_models()

    table = Table(title="VaaniDub AI Models")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("VRAM (MB)")
    table.add_column("Download (GB)")
    table.add_column("Description")

    for m in models:
        table.add_row(
            m["name"],
            m["type"],
            str(m["gpu_vram_mb"]),
            str(m["download_size_gb"]),
            m["description"],
        )

    console.print(table)

    # GPU info
    gpu_info = manager.check_gpu()
    if gpu_info.get("available"):
        console.print(f"\n[green]GPU:[/green] {gpu_info['device_name']} "
                       f"({gpu_info['free_vram_mb']}MB free / {gpu_info['total_vram_mb']}MB total)")
    else:
        reason = gpu_info.get("reason", "unknown")
        console.print(f"\n[yellow]GPU:[/yellow] Not available — {reason}")

    reqs = manager.get_gpu_requirements()
    console.print(f"[dim]Minimum VRAM needed: {reqs['min_vram_mb']}MB (sequential execution)[/dim]")


@models_app.command("download")
def models_download(
    all_models: bool = typer.Option(False, "--all", help="Download all models"),
    model: Optional[str] = typer.Option(None, "--model", help="Download specific model"),
):
    """Download AI models."""
    from vaanidub.config import AppConfig
    from vaanidub.models.model_manager import ModelManager

    config = AppConfig()
    manager = ModelManager()

    if all_models:
        console.print("[bold]Downloading all models...[/bold]")
        total_gb = manager.get_total_download_size()
        console.print(f"Total download size: ~{total_gb:.1f} GB")
        asyncio.run(manager.download_all(hf_token=config.hf_token))
        console.print("[green]All models downloaded.[/green]")
    elif model:
        console.print(f"Downloading {model}...")
        asyncio.run(manager.download_model(model, hf_token=config.hf_token))
        console.print(f"[green]{model} downloaded.[/green]")
    else:
        console.print("[yellow]Specify --all or --model <name>[/yellow]")


@models_app.command("check")
def models_check():
    """Check GPU availability and model readiness."""
    from vaanidub.models.model_manager import ModelManager

    manager = ModelManager()
    gpu = manager.check_gpu()

    if gpu.get("available"):
        console.print(f"[green]GPU OK:[/green] {gpu['device_name']}")
        console.print(f"  VRAM: {gpu['free_vram_mb']}MB free / {gpu['total_vram_mb']}MB total")
        console.print(f"  CUDA: {gpu.get('cuda_version', 'N/A')}")
    else:
        console.print(f"[red]GPU not available:[/red] {gpu.get('reason')}")


# ─── Detect Command ───

@app.command()
def detect(input_file: Path = typer.Argument(..., help="Input audio/video file")):
    """Detect the language of an audio/video file."""
    if not input_file.exists():
        console.print(f"[red]Error:[/red] File not found: {input_file}")
        raise typer.Exit(1)

    console.print(f"Detecting language of {input_file}...")
    # Quick detection using first 30 seconds
    try:
        import whisperx
        model = whisperx.load_model("base", "cpu", compute_type="int8")
        audio = whisperx.load_audio(str(input_file))
        # Use only first 30 seconds
        audio = audio[:30 * 16000]
        result = model.transcribe(audio, batch_size=16)

        lang = result.get("language", "unknown")
        conf = result.get("language_probability", 0)
        lang_info = LANGUAGES.get(lang)
        name = lang_info.name if lang_info else lang

        console.print(f"\n[bold]Detected:[/bold] {name} ({lang}) — {conf:.0%} confidence")
    except Exception as e:
        console.print(f"[red]Detection failed:[/red] {e}")
        raise typer.Exit(1)


# ─── Serve Command ───

@app.command()
def serve(
    port: int = typer.Option(8000, "--port", "-p", help="API server port"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of Celery workers"),
):
    """Start the API server and Celery worker."""
    console.print(f"[bold]Starting VaaniDub server on port {port}...[/bold]")

    import uvicorn

    from vaanidub.api.app import create_app

    api_app = create_app()
    uvicorn.run(api_app, host="0.0.0.0", port=port)


# ─── Languages Command ───

@app.command()
def languages():
    """List all supported languages."""
    table = Table(title="Supported Indian Languages")
    table.add_column("Code", style="cyan")
    table.add_column("Name")
    table.add_column("Native Name")
    table.add_column("Script")
    table.add_column("TTS Providers")

    for code, info in LANGUAGES.items():
        table.add_row(
            code,
            info.name,
            info.native_name,
            info.script,
            ", ".join(info.tts_providers),
        )

    console.print(table)


if __name__ == "__main__":
    app()
