"""M1 CLI. Later stages remain gated by the handoff's milestones."""

import json
import os
from pathlib import Path
from typing import Annotated

import typer

from crosstalk.artifacts import CallLimitReached, RunStore, new_run
from crosstalk.budget import INITIAL_TEST_LIMIT_USD, Budget, BudgetExceeded
from crosstalk.config import KEYS, load_config, load_environment
from crosstalk.decisions.jev import JevClient
from crosstalk.http import ProviderError
from crosstalk.llm.base import RefusalError
from crosstalk.llm.factory import create_client
from crosstalk.research.dedupe import EmbeddingClient
from crosstalk.research.dossier import ResearchPipeline, load_dossier
from crosstalk.research.fetch import PageFetcher
from crosstalk.research.search import TavilySearch

app = typer.Typer(no_args_is_help=True, help="CROSSTALK: evidence-grounded podcast pipeline.")


@app.command()
def doctor():
    """Show credential presence without exposing values or making API calls."""
    load_environment()
    for vendor, names in KEYS.items():
        configured = any(os.getenv(name) for name in names)
        typer.echo(f"{vendor}: {'configured' if configured else 'missing'}")
    typer.echo(
        "Implemented: research, discussion, transcript evidence review, optional guarded polish, and draft audio rendering."
    )


@app.command()
def inspect(run_dir: Path):
    """Summarize a saved dossier and its acceptance report."""
    dossier = load_dossier(run_dir / "dossier.json")
    typer.echo(f"{dossier.topic}: {len(dossier.sources)} sources, {len(dossier.claims)} claims")
    if (run_dir / "acceptance.json").exists():
        typer.echo((run_dir / "acceptance.json").read_text())


@app.command()
def research(
    topic: Annotated[str | None, typer.Option(help="Policy question/topic.")] = None,
    sources_file: Annotated[Path | None, typer.Option(help="One source URL per line.")] = None,
    run_dir: Annotated[Path | None, typer.Option(help="Resume an existing run directory.")] = None,
    config: Annotated[Path, typer.Option()] = Path("config/show.yaml"),
    live: Annotated[bool, typer.Option(help="Enable paid vendor calls.")] = False,
    budget_usd: Annotated[
        float,
        typer.Option(
            min=0.01, max=INITIAL_TEST_LIMIT_USD, help="Shared initial-test dollar ceiling."
        ),
    ] = INITIAL_TEST_LIMIT_USD,
):
    """Build and categorize a cited dossier, checkpointing every source."""
    load_environment()
    if run_dir and (run_dir / "run.json").exists():
        saved_topic = json.loads((run_dir / "run.json").read_text())["topic"]
        if topic is not None and topic != saved_topic:
            raise typer.BadParameter("--topic must match the saved run")
        topic = saved_topic
    if not topic:
        raise typer.BadParameter("Provide --topic or --run-dir")
    if not live:
        typer.echo(
            "Live calls disabled. Add --live to execute research; run doctor to check credentials."
        )
        raise typer.Exit(0)
    settings = load_config(config).research
    urls = None
    if sources_file is not None:
        urls = [
            line.strip()
            for line in sources_file.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if not urls:
            raise typer.BadParameter("Source file is empty")
    path = run_dir or new_run(topic)
    typer.echo(f"Research artifacts: {path}")
    # A fixed project-wide ledger prevents budget reset by creating another run.
    store = RunStore(
        path,
        max_calls=settings.max_calls,
        budget=Budget(Path("runs/initial_test_budget.jsonl"), budget_usd),
    )
    llm = create_client(settings.vendor, settings.model, store)
    decisions = JevClient(settings.decision_model, store)
    search = TavilySearch(store) if os.getenv("TAVILY_API_KEY") else None
    try:
        dossier = ResearchPipeline(
            store,
            llm,
            decisions,
            PageFetcher(),
            search,
            max_pages=settings.max_pages,
            threshold=settings.decision_threshold,
            embeddings=EmbeddingClient(store),
        ).run(topic, urls)
    except (ValueError, ProviderError, RefusalError, CallLimitReached, BudgetExceeded) as error:
        typer.echo(f"Research stopped: {error}. Saved checkpoints can be resumed.", err=True)
        raise typer.Exit(1) from None
    typer.echo(f"Saved {len(dossier.sources)} sources and {len(dossier.claims)} supported claims.")
    report = json.loads((path / "acceptance.json").read_text())
    if not report["passed"]:
        typer.echo("M1 acceptance incomplete; see acceptance.json.", err=True)
        raise typer.Exit(2)
    typer.echo(
        "Dossier acceptance passed. Live adapter and categorization evaluation also gate M1."
    )


@app.command()
def discuss(
    run_dir: Path,
    mode: Annotated[str, typer.Option(help="two (M2) or full (M3)")] = "two",
    minutes: Annotated[int, typer.Option(min=1, max=10)] = 5,
    config: Annotated[Path, typer.Option()] = Path("config/show.yaml"),
    live: Annotated[bool, typer.Option()] = False,
):
    """Generate a draft discussion from an existing dossier, without researching again."""
    import yaml

    from crosstalk.agents.host import HostAgent
    from crosstalk.agents.persona import PersonaAgent
    from crosstalk.models import Persona
    from crosstalk.orchestrator.director import Director

    if mode not in {"two", "full"}:
        raise typer.BadParameter("--mode must be two or full")
    if not live:
        typer.echo("Live calls disabled; add --live to generate discussion.")
        raise typer.Exit(0)
    load_environment()
    settings = load_config(config)
    dossier = load_dossier(run_dir / "dossier.json")
    report_path = run_dir / "acceptance.json"
    if not report_path.exists() or not json.loads(report_path.read_text())["passed"]:
        raise typer.BadParameter("Research acceptance must pass before discussion")
    personas = [
        Persona.model_validate(yaml.safe_load(Path(name).read_text())) for name in settings.roster
    ]
    if mode == "two":
        personas = [persona for persona in personas if persona.id in {"nia", "sam"}]
    elif (
        not (run_dir / "discussion_two" / "discussion_acceptance.json").exists()
        or not json.loads((run_dir / "discussion_two" / "discussion_acceptance.json").read_text())[
            "automated_checks_passed"
        ]
    ):
        raise typer.BadParameter("Complete the M2 two-persona discussion before running M3")
    path = run_dir / f"discussion_{mode}"
    store = RunStore(
        path,
        max_calls=300,
        budget=Budget(Path("runs/initial_test_budget.jsonl"), INITIAL_TEST_LIMIT_USD),
    )
    agents = [
        (HostAgent if persona.role == "host" else PersonaAgent)(
            persona,
            create_client(persona.vendor, persona.model, store),
            settings.discussion.tone,
            path / "refusals.jsonl",
        )
        for persona in personas
    ]
    bid_llm = create_client(settings.discussion.bid_vendor, settings.discussion.bid_model, store)
    decisions = JevClient(settings.research.decision_model, store)
    try:
        report = Director(
            store,
            dossier,
            agents,
            bid_llm,
            decisions,
            settings.discussion,
            minutes=minutes,
            mode=mode,
        ).run()
    except (ValueError, ProviderError, RefusalError, CallLimitReached, BudgetExceeded) as error:
        typer.echo(f"Discussion stopped: {error}. Resume with the same command.", err=True)
        raise typer.Exit(1) from None
    typer.echo(json.dumps({"transcript": str(path / "transcript.md"), **report}, indent=2))
    if not report["automated_checks_passed"]:
        raise typer.Exit(2)
    typer.echo("Human transcript review is required before M3 can pass and audio work can begin.")


@app.command()
def factcheck(
    run_dir: Path,
    config: Annotated[Path, typer.Option()] = Path("config/show.yaml"),
    live: Annotated[bool, typer.Option()] = False,
):
    """Review the approved full transcript against saved source passages; resume safely."""
    from crosstalk.editor.artifacts import stage_paths
    from crosstalk.editor.factcheck import FactChecker

    if not live:
        typer.echo("Live calls disabled; add --live to check transcript evidence.")
        raise typer.Exit(0)
    load_environment()
    settings = load_config(config)
    path, _ = stage_paths(run_dir)
    store = RunStore(
        path,
        max_calls=500,
        budget=Budget(Path("runs/initial_test_budget.jsonl"), INITIAL_TEST_LIMIT_USD),
    )
    try:
        report = FactChecker(
            store,
            create_client(settings.research.vendor, settings.research.model, store),
            JevClient(settings.research.decision_model, store),
            settings.research.decision_threshold,
        ).run(run_dir, run_dir / "discussion_full", load_dossier(run_dir / "dossier.json"))
    except (ValueError, ProviderError, RefusalError, CallLimitReached, BudgetExceeded) as error:
        typer.echo(f"Fact checking stopped: {error}. Saved checkpoints can be resumed.", err=True)
        raise typer.Exit(1) from None
    typer.echo(
        json.dumps(
            {
                "factcheck": str(path / "factcheck.json"),
                "turn_counts": report["turn_counts"],
                "review_required": report["review_required"],
            },
            indent=2,
        )
    )


@app.command()
def editorial(
    run_dir: Path,
    config: Annotated[Path, typer.Option()] = Path("config/show.yaml"),
    live: Annotated[bool, typer.Option()] = False,
):
    """Source-bound factual corrections before the single speech-polish pass."""
    import yaml

    from crosstalk.editor.editorial import EditorialPass
    from crosstalk.editor.quality import discussion_approved
    from crosstalk.models import Persona

    if not live:
        typer.echo("Live calls disabled; add --live for editorial review.")
        return
    if not discussion_approved(run_dir):
        raise typer.BadParameter("Current transcript must pass M3 quality review first")
    load_environment()
    settings = load_config(config)
    store = RunStore(
        run_dir / "editorial",
        max_calls=200,
        budget=Budget(Path("runs/initial_test_budget.jsonl"), INITIAL_TEST_LIMIT_USD),
    )
    try:
        report = EditorialPass(
            store,
            create_client(settings.editorial.vendor, settings.editorial.model, store),
            JevClient(settings.research.decision_model, store),
        ).run(
            run_dir,
            {
                persona.id: persona.model_dump()
                for name in settings.roster
                if (persona := Persona.model_validate(yaml.safe_load(Path(name).read_text())))
            },
            minutes=settings.episode_minutes,
        )
    except (ValueError, ProviderError, RefusalError, CallLimitReached, BudgetExceeded) as error:
        typer.echo(f"Editorial stopped: {error}. Saved checkpoints can be resumed.", err=True)
        raise typer.Exit(1) from None
    typer.echo(json.dumps({k: v for k, v in report.items() if k != "checks"}, indent=2))
    if not report["passed"]:
        typer.echo(
            "Editorial requires review; inspect editorial/editorial_review.json before polish or speech.",
            err=True,
        )
        raise typer.Exit(2)


@app.command()
def polish(
    run_dir: Path,
    config: Annotated[Path, typer.Option()] = Path("config/show.yaml"),
    live: Annotated[bool, typer.Option()] = False,
):
    """One light spoken-flow pass preserving each persona; keeps originals and an edit diff."""
    import yaml

    from crosstalk.editor.artifacts import active_transcript, stage_paths
    from crosstalk.editor.polish import Polisher
    from crosstalk.editor.quality import discussion_approved
    from crosstalk.models import Persona

    if not live:
        typer.echo("Live calls disabled; add --live for optional polish.")
        raise typer.Exit(0)
    load_environment()
    editorial(run_dir, config=config, live=True)
    turns = active_transcript(run_dir)
    if not discussion_approved(run_dir):
        raise typer.BadParameter("Current transcript must pass M3 quality review first")
    settings = load_config(config)
    store = RunStore(
        stage_paths(run_dir)[1],
        max_calls=200,
        budget=Budget(Path("runs/initial_test_budget.jsonl"), INITIAL_TEST_LIMIT_USD),
    )
    try:
        Polisher(
            store,
            create_client(settings.research.vendor, settings.research.model, store),
            JevClient(settings.research.decision_model, store),
        ).run(
            turns,
            {
                persona.id: persona.model_dump()
                for name in settings.roster
                if (persona := Persona.model_validate(yaml.safe_load(Path(name).read_text())))
            },
        )
    except (ValueError, ProviderError, RefusalError, CallLimitReached, BudgetExceeded) as error:
        typer.echo(f"Polish stopped: {error}. Saved checkpoints can be resumed.", err=True)
        raise typer.Exit(1) from None
    typer.echo(
        "Polish saved separately. Preservation checks are provisional; edited assertions still require evidence review."
    )


@app.command()
def audio(
    run_dir: Path,
    voices_file: Annotated[Path, typer.Option()] = Path("config/voices.yaml"),
    config: Annotated[Path, typer.Option()] = Path("config/show.yaml"),
    live: Annotated[bool, typer.Option()] = False,
    preview: Annotated[
        bool,
        typer.Option(
            help="Draft preview permits unresolved evidence flags; never grants publication approval."
        ),
    ] = True,
):
    """Render cached voice clips and a timed mix from an evidence-checked transcript."""
    import subprocess

    import yaml

    from crosstalk.audio.render import render_audio
    from crosstalk.audio.tts import OpenAITTS

    if not live:
        typer.echo("Live calls disabled; add --live to render speech.")
        raise typer.Exit(0)
    load_environment()
    polish(run_dir, config=config, live=True)
    factcheck(run_dir, config=config, live=True)
    store = RunStore(
        run_dir / "audio",
        max_calls=150,
        budget=Budget(Path("runs/initial_test_budget.jsonl"), INITIAL_TEST_LIMIT_USD),
    )
    try:
        result = render_audio(
            store,
            OpenAITTS(store),
            run_dir,
            yaml.safe_load(voices_file.read_text()),
            preview=preview,
        )
    except (
        ValueError,
        ProviderError,
        CallLimitReached,
        BudgetExceeded,
        subprocess.CalledProcessError,
    ) as error:
        typer.echo(f"Audio stopped: {error}. Cached clips can be resumed.", err=True)
        raise typer.Exit(1) from None
    typer.echo(json.dumps(result, indent=2))


@app.command()
def topics(
    sources_file: Annotated[Path, typer.Option()] = Path(
        "config/research/congestion_pricing_urls.txt"
    ),
    top: Annotated[int, typer.Option(min=1, max=10)] = 5,
    config: Annotated[Path, typer.Option()] = Path("config/show.yaml"),
    live: Annotated[bool, typer.Option()] = False,
):
    """Rank policy questions from curated sources; does not measure social popularity."""
    if not live:
        typer.echo("Live calls disabled; add --live for curated-source topic ranking.")
        return None
    from datetime import UTC, datetime

    from crosstalk.research.discover import TopicDiscovery, evaluate_debatability

    load_environment()
    settings = load_config(config)
    urls = [
        s.strip()
        for s in sources_file.read_text().splitlines()
        if s.strip() and not s.strip().startswith("#")
    ]
    path = Path("runs/discovery") / datetime.now(UTC).strftime("%Y-%m-%d")
    store = RunStore(
        path,
        max_calls=50,
        budget=Budget(Path("runs/initial_test_budget.jsonl"), INITIAL_TEST_LIMIT_USD),
    )
    decisions = JevClient(settings.research.decision_model, store)
    try:
        evaluate_debatability(store, decisions)
        report = TopicDiscovery(
            store,
            create_client(settings.research.vendor, settings.research.model, store),
            decisions,
            PageFetcher(),
        ).run(urls)
    except (ValueError, ProviderError, RefusalError, CallLimitReached, BudgetExceeded) as error:
        typer.echo(f"Topic discovery stopped: {error}", err=True)
        raise typer.Exit(1) from None
    typer.echo(
        json.dumps(
            {
                "mode": report["mode"],
                "popularity_measured": False,
                "candidates": report["candidates"][:top],
            },
            indent=2,
        )
    )
    return report


@app.command("run")
def run_pipeline(
    run_dir: Annotated[Path | None, typer.Argument()] = None,
    from_stage: Annotated[
        str, typer.Option(help="research, discussion, editorial, polish, factcheck, or audio")
    ] = "research",
    topic: Annotated[str | None, typer.Option()] = None,
    sources_file: Annotated[Path | None, typer.Option()] = None,
    auto: Annotated[
        bool,
        typer.Option(
            help="Unattended drafting after a matching prototype has passed human M3 review."
        ),
    ] = False,
    prototype_run: Annotated[Path | None, typer.Option()] = None,
    config: Annotated[Path, typer.Option()] = Path("config/show.yaml"),
    live: Annotated[bool, typer.Option()] = False,
):
    """Run or resume a draft. Explicit auto mode inherits a validated prototype quality gate."""
    from crosstalk.artifacts import write_json

    stages = ["research", "discussion", "editorial", "polish", "factcheck", "audio"]
    if from_stage not in stages:
        raise typer.BadParameter("Unknown --from-stage")
    selected = stages[stages.index(from_stage) :]
    if not live:
        typer.echo("Live calls disabled. Planned stages: " + " -> ".join(selected))
        return
    from crosstalk.editor.quality import (
        discussion_approved,
        permit_automatic_draft,
        prototype_proof,
    )

    if auto:
        if prototype_run is None:
            status = json.loads(Path("project_status.json").read_text())
            prototype_run = Path(status["research_run"])
        prototype_proof(prototype_run, config)
    if run_dir is not None and (run_dir / "run.json").exists():
        saved = json.loads((run_dir / "run.json").read_text())["topic"]
        if topic and topic != saved:
            raise typer.BadParameter("Topic must match the saved run")
        topic = saved
    if not topic and auto and from_stage == "research":
        labels_path = Path("config/discovery_labels.json")
        labels = json.loads(labels_path.read_text()) if labels_path.exists() else {}
        if not labels.get("confirmed_by_user"):
            raise typer.BadParameter(
                "Automatic topic selection needs human calibration labels; use --topic for unattended drafting with an explicit topic."
            )
        sources_file = sources_file or Path("config/research/congestion_pricing_urls.txt")
        candidates = topics(sources_file=sources_file, config=config, live=True)["candidates"]
        candidate = next((c for c in candidates if c["eligible"]), None)
        if candidate is None:
            raise typer.BadParameter("No defensibly opposing topic passed the discovery threshold")
        topic = candidate["topic"]
    if run_dir is None:
        if not topic:
            raise typer.BadParameter("Provide a run directory, --topic, or calibrated --auto")
        run_dir = new_run(topic)
    progress = {
        "from_stage": from_stage,
        "completed": [],
        "status": "running",
        "automatic_draft": auto,
    }

    for stage in selected:
        progress["current_stage"] = stage
        write_json(run_dir / "pipeline.json", progress)
        if stage in {"editorial", "polish", "factcheck", "audio"}:
            discussion = run_dir / "discussion_full"
            if auto and not (run_dir / "automation_quality.json").exists():
                permit_automatic_draft(run_dir, prototype_run, config)
            try:
                approved = discussion_approved(run_dir)
            except FileNotFoundError:
                approved = False
            if not approved:
                progress["status"] = "awaiting_human_review"
                progress["review_artifact"] = str(discussion / "transcript.md")
                write_json(run_dir / "pipeline.json", progress)
                typer.echo(
                    f"M3 requires human reading approval in handoff.md section 6. Review {discussion / 'transcript.md'} before continuing."
                )
                raise typer.Exit(3)
        try:
            if stage == "research":
                research(
                    topic=topic,
                    sources_file=sources_file,
                    run_dir=run_dir,
                    config=config,
                    live=True,
                    budget_usd=INITIAL_TEST_LIMIT_USD,
                )
            elif stage == "discussion":
                discuss(run_dir, mode="two", minutes=5, config=config, live=True)
                discuss(run_dir, mode="full", minutes=10, config=config, live=True)
            elif stage == "editorial":
                editorial(run_dir, config=config, live=True)
            elif stage == "polish":
                polish(run_dir, config=config, live=True)
            elif stage == "factcheck":
                factcheck(run_dir, config=config, live=True)
            else:
                audio(run_dir, config=config, live=True, preview=True)
        except (typer.Exit, ValueError, OSError):
            progress["status"] = "stopped"
            write_json(run_dir / "pipeline.json", progress)
            raise
        progress["completed"].append(stage)
        write_json(run_dir / "pipeline.json", progress)
    progress["status"] = "draft_complete"
    progress["publication_approved"] = False
    write_json(run_dir / "pipeline.json", progress)
    typer.echo(
        "Draft pipeline complete. See evidence flags and audio listening review before publication."
    )


@app.command("review")
def review_episode(run_dir: Path):
    """Build a self-contained local factual-review screen; no API calls."""
    from crosstalk.review.screen import build_screen
    from crosstalk.review.workflow import review_status, snapshot

    typer.echo(str(build_screen(run_dir).resolve()))
    typer.echo(json.dumps(review_status(snapshot(run_dir)), indent=2))


@app.command("review-import")
def review_import(run_dir: Path, decisions_file: Path):
    """Record explicit version-bound decisions downloaded from the local screen."""
    from crosstalk.review.workflow import import_decisions

    try:
        result = import_decisions(run_dir, json.loads(decisions_file.read_text()))
    except (ValueError, OSError) as error:
        raise typer.BadParameter(str(error)) from None
    typer.echo(json.dumps(result, indent=2))


@app.command("export")
def export_bundle(
    run_dir: Path,
    output: Annotated[Path | None, typer.Option()] = None,
    require_reviewed: Annotated[bool, typer.Option()] = False,
):
    """Export current audio, transcript, sources, summary, and review provenance offline."""
    from crosstalk.review.export import export_episode

    try:
        path = export_episode(run_dir, output, require_reviewed)
    except (ValueError, OSError) as error:
        raise typer.BadParameter(str(error)) from None
    typer.echo(str(path.resolve()))


if __name__ == "__main__":
    app()
