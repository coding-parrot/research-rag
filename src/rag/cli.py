"""Command-line interface.

rag ingest   fetch PDFs, OCR, detect headers, chunk; writes the ingest report
rag index    embed chunks and build the vector + BM25 indexes
rag ask      one question through the full pipeline
rag eval     run the golden set; --judge adds the LLM-graded metrics
rag headers  score header detection against the hand-labelled truth
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from rag.config import Config
from rag.observability import configure_logging

app = typer.Typer(add_completion=False, no_args_is_help=True, pretty_exceptions_enable=False)
console = Console(stderr=False)

ConfigOption = Annotated[
    Path | None,
    typer.Option("--config", "-c", help="Path to a YAML config; defaults apply otherwise."),
]


def _load(config_path: Path | None, verbose: bool = False) -> Config:
    import logging

    from rag.errors import RagError

    configure_logging(logging.DEBUG if verbose else logging.INFO)
    try:
        return Config.load(config_path)
    except RagError as exc:
        # A typed config failure is a usage error, not a crash: exit code 2 with
        # the message, no traceback.
        console.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(code=2) from exc


@app.command()
def ingest(
    config_path: ConfigOption = None,
    papers: Annotated[
        list[str] | None, typer.Option("--paper", "-p", help="Ingest only these ids.")
    ] = None,
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Fetch, OCR, detect headers, and chunk the corpus."""
    config = _load(config_path, verbose)
    from rag.app import build_ocr_engine
    from rag.ingest.service import run_ingest

    chunks, report = run_ingest(config, build_ocr_engine(config), paper_ids=papers)

    table = Table(title="Ingest")
    table.add_column("paper")
    table.add_column("pages", justify="right")
    table.add_column("sections", justify="right")
    table.add_column("chunks", justify="right")
    table.add_column("split/merged", justify="right")
    table.add_column("health")
    for doc in report.documents:
        sections = str(doc.headers.accepted) if doc.headers else "-"
        chunk_count = str(doc.chunks.chunks_emitted) if doc.chunks else "-"
        adjusted = (
            f"{doc.chunks.sections_split}/{doc.chunks.sections_merged}" if doc.chunks else "-"
        )
        health = (
            "[green]ok[/green]" if doc.healthy else f"[red]{doc.error or 'check structure'}[/red]"
        )
        table.add_row(doc.doc_id, str(doc.pages), sections, chunk_count, adjusted, health)
    console.print(table)
    console.print(f"{len(chunks)} chunks from {len(report.documents)} documents")

    # Chunks are staged to disk here and turned into an index by `rag index`.
    # Only healthy documents are staged: an unhealthy document's chunks are
    # typically one mislabeled whole-paper blob, and indexing them is exactly the
    # silent retrieval poisoning the ingest report exists to prevent.
    healthy_ids = {d.doc_id for d in report.documents if d.healthy}
    staged_chunks = [c for c in chunks if c.doc_id in healthy_ids]

    staging = config.paths.index / "staged"
    from rag.index.base import ChunkStore

    if papers:
        # A partial ingest refreshes the selected papers inside the staged store.
        # Saving only the subset would truncate the previously staged corpus, so
        # `rag index` would silently rebuild from just these papers.
        existing = (
            ChunkStore.load(staging).chunks if (staging / ChunkStore.FILENAME).exists() else ()
        )
        reingested = {d.doc_id for d in report.documents if d.ok}
        store = ChunkStore(c for c in existing if c.doc_id not in reingested)
        store.add(staged_chunks)
    else:
        store = ChunkStore(staged_chunks)
    store.save(staging)
    console.print(f"{len(store)} chunk(s) staged to {staging}")

    if report.unhealthy:
        excluded = len(chunks) - len(staged_chunks)
        console.print(
            f"[yellow]{len(report.unhealthy)} document(s) look unhealthy; "
            f"{excluded} chunk(s) from them were not staged. See the ingest report.[/yellow]"
        )
        raise typer.Exit(code=1)


@app.command()
def index(
    config_path: ConfigOption = None,
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Embed staged chunks and build the searchable indexes."""
    config = _load(config_path, verbose)
    from rag.app import build_index, save_index
    from rag.index.base import ChunkStore

    # The staged store is written by `rag ingest`; the hint must not send the user
    # back into the command that just failed.
    staged = ChunkStore.load(config.paths.index / "staged", hint="run `rag ingest` first")
    bundle = build_index(config, list(staged.chunks))
    save_index(bundle, config.paths.index)
    console.print(f"indexed {len(staged)} chunks -> {config.paths.index}")


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="The question to answer.")],
    config_path: ConfigOption = None,
    show_chunks: bool = typer.Option(False, "--show-chunks", help="Print retrieved chunks."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Ask one question through the full pipeline."""
    config = _load(config_path, verbose)
    from rag.app import build_pipeline, load_index
    from rag.domain import AnswerStatus
    from rag.guardrails.output_guard import format_citation_markers

    bundle = load_index(config)
    pipeline = build_pipeline(config, bundle)
    answer = pipeline.ask(question)

    if show_chunks:
        for scored in answer.retrieved:
            console.print(
                f"[dim]{scored.rank}. {scored.score:.3f}  {scored.chunk.citation_label}[/dim]"
            )
        console.print()

    if answer.status is AnswerStatus.OK:
        console.print(format_citation_markers(answer.text, list(answer.citations)))
    else:
        console.print(f"[yellow]({answer.status.value})[/yellow] {answer.text}")

    console.print(
        f"\n[dim]trace={answer.trace_id} tokens_in={answer.usage.input_tokens} "
        f"tokens_out={answer.usage.output_tokens} llm_calls={answer.usage.llm_calls}[/dim]"
    )


@app.command("eval")
def run_eval(
    config_path: ConfigOption = None,
    golden_path: Annotated[Path | None, typer.Option("--golden")] = None,
    judge: bool = typer.Option(
        False, "--judge", help="Also run LLM-judged metrics (spends tokens)."
    ),
    gate_unreviewed: bool = typer.Option(
        False, "--gate-unreviewed", help="Let unreviewed items gate."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the golden set and check thresholds. Exit code 1 on failure."""
    config = _load(config_path, verbose)
    from rag.app import build_pipeline, load_index
    from rag.eval.datasets import load_golden
    from rag.eval.judge import Judge
    from rag.eval.runner import EvalRunner, save_report
    from rag.generate.client import build_client

    golden = load_golden(golden_path or config.paths.evals / "golden" / "golden.yaml")
    bundle = load_index(config)
    pipeline = build_pipeline(config, bundle)

    judge_obj = None
    if judge:
        from rag.config import Secrets

        # Mirror build_pipeline: the key lives in Secrets (.env via pydantic-settings,
        # never exported to os.environ), so the SDK's env-var fallback cannot see it
        # and it must be passed to the client explicitly.
        secrets = Secrets()
        api_key = (
            secrets.anthropic_api_key.get_secret_value() if secrets.anthropic_api_key else None
        )
        judge_client = build_client(
            config.eval.judge_provider,
            model=config.eval.judge_model,
            ollama_model=config.generate.ollama_model,
            ollama_host=config.generate.ollama_host,
            api_key=api_key,
            openai_api_key=(
                secrets.openai_api_key.get_secret_value() if secrets.openai_api_key else None
            ),
        )
        judge_obj = Judge(
            judge_client, model=config.eval.judge_model, effort=config.eval.judge_effort
        )

    runner = EvalRunner(config, judge=judge_obj)
    report = runner.run(pipeline, golden, gate_on_unreviewed=gate_unreviewed, with_judge=judge)
    save_report(report, config.paths.evals / "results")

    table = Table(title=f"Eval {report.run_id} (config {report.config_hash})")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_column("threshold", justify="right")
    table.add_column("status")
    for check in report.checks:
        direction = ">=" if check.higher_is_better else "<="
        status = "[green]pass[/green]" if check.passed else "[red]FAIL[/red]"
        table.add_row(
            check.name, f"{check.value:.3f}", f"{direction} {check.threshold:.3f}", status
        )
    console.print(table)

    for key, value in sorted(report.aggregates.items()):
        console.print(f"  [dim]{key}: {value:.4f}[/dim]")
    for note in report.notes:
        console.print(f"[yellow]note: {note}[/yellow]")

    if not report.passed:
        raise typer.Exit(code=1)


@app.command()
def headers(
    config_path: ConfigOption = None,
    labels_path: Annotated[Path | None, typer.Option("--labels")] = None,
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Score header detection against hand-labelled section lists."""
    config = _load(config_path, verbose)
    from rag.app import build_ocr_engine
    from rag.eval.datasets import load_header_labels
    from rag.eval.metrics import mean, score_headers
    from rag.ingest.headers import HeaderDetector, read_outline
    from rag.ingest.manifest import load_manifest
    from rag.ingest.normalize import normalize

    labels = load_header_labels(labels_path or config.paths.evals / "headers" / "labels.yaml")
    manifest = load_manifest(config.paths.corpus_manifest)
    ocr_engine = build_ocr_engine(config)
    detector = HeaderDetector(config.headers)

    scores = []
    table = Table(title="Header detection")
    table.add_column("paper")
    table.add_column("precision", justify="right")
    table.add_column("recall", justify="right")
    table.add_column("f1", justify="right")

    for label in labels:
        paper = manifest.get(label.doc_id)
        pdf_path = config.paths.pdfs / paper.filename
        if not pdf_path.exists():
            console.print(f"[yellow]skipping {label.doc_id}: PDF not fetched[/yellow]")
            continue
        ocr = ocr_engine.read(pdf_path, label.doc_id)
        result = normalize(ocr, title=paper.title)
        outline = read_outline(pdf_path) if config.headers.use_outline else ()
        headings, _ = detector.detect(result, outline=outline)
        detected = [h.label for h in headings if h.level == 1]
        score = score_headers(detected, label)
        scores.append(score)
        table.add_row(
            label.doc_id, f"{score.precision:.2f}", f"{score.recall:.2f}", f"{score.f1:.2f}"
        )

    console.print(table)
    # Zero scored documents must fail, not pass: a CI gate built on this command
    # would otherwise go green with the F1 threshold never evaluated.
    if not scores:
        console.print(
            f"[red]no documents were scored: no labelled PDFs found under "
            f"{config.paths.pdfs}; run `rag ingest` first[/red]"
        )
        raise typer.Exit(code=1)
    mean_f1 = mean([s.f1 for s in scores])
    console.print(f"mean F1: {mean_f1:.3f} (threshold {config.eval.min_header_boundary_f1})")
    if mean_f1 < config.eval.min_header_boundary_f1:
        raise typer.Exit(code=1)


@app.command()
def info(config_path: ConfigOption = None) -> None:
    """Print resolved config hashes and index status."""
    config = _load(config_path)
    payload = {
        "config_hash": config.hash(),
        "ingest_hash": config.ingest_hash(),
        "index_exists": (config.paths.index / "chunks.jsonl").exists(),
        "corpus_manifest": str(config.paths.corpus_manifest),
    }
    console.print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    sys.exit(app())
