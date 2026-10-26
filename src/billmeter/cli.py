"""Command line interface: `billmeter --help`."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from .documents import SUPPORTED_SUFFIXES, UnreadableDocument
from .evals import html_report
from .evals.compare import compare as compare_results
from .evals.compare import load_results
from .evals.graders import hmac_value
from .evals.report import print_comparison, print_results
from .evals.runner import run_eval, save_results
from .extractor import extract
from .labeling import (
    BILLS_DIR,
    DRAFTS_DIR,
    InvalidValue,
    add_case,
    build_expected,
    default_tags,
    drafts_by_doubt,
    parse_answer,
    pending_drafts,
    save_draft,
    unlabeled_files,
)
from .providers import (
    MODEL_LISTS,
    PROVIDER_ENV,
    GenerationSettings,
    ProviderConfigError,
    available_providers,
    get_provider,
)
from .schema import EXTRACTED_FIELDS

load_dotenv()  # read API keys from .env in the current folder
app = typer.Typer(help="Extract data from Italian utility bills with any AI model, and evaluate it.",
                  no_args_is_help=True, add_completion=False)
console = Console()

# Listed from documents.py so the help never drifts from what is actually accepted.
SUPPORTED_HELP = ", ".join(suffix.lstrip(".").upper() for suffix in SUPPORTED_SUFFIXES)

ModelOpt = typer.Option(..., "--model", "-m", help="provider:model, e.g. ollama:<model> (see `billmeter providers`)")
PromptOpt = typer.Option("v2", "--prompt", "-p", help="Prompt version in prompts/ (v1 = naive baseline, v2 = tuned)")
TempOpt = typer.Option("0", "--temperature", "-t", help="Number, or 'none' to not send it")
EffortOpt = typer.Option(None, "--effort", help="Reasoning depth where the model supports it: low, medium, high")


def _settings(temperature: str, max_tokens: int, timeout: float,
              effort: str | None = None) -> GenerationSettings:
    temp = None if temperature.lower() == "none" else float(temperature)
    return GenerationSettings(temperature=temp, max_tokens=max_tokens, timeout_s=timeout,
                              effort=effort)


@app.command("extract")
def extract_cmd(
    file: Path = typer.Argument(..., exists=True, help=f"Bill to read: {SUPPORTED_HELP}"),
    model: str = ModelOpt,
    prompt: str = PromptOpt,
    temperature: str = TempOpt,
    max_tokens: int = typer.Option(2048),
    effort: str = EffortOpt,
    timeout: float = typer.Option(120.0),
    raw: bool = typer.Option(False, "--raw", help="Also print the model's raw answer"),
):
    """Extract the data from one bill and print it as JSON."""
    try:
        provider = get_provider(model, _settings(temperature, max_tokens, timeout, effort))
        result = extract(file, provider, prompt_version=prompt)
    except (ProviderConfigError, ValueError, FileNotFoundError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc

    if raw:
        console.rule("raw model answer")
        console.print(result.raw_text)
    console.rule(f"{provider.spec} · {result.latency_s:.2f}s · attempts {result.attempts}")
    if result.error:
        console.print(f"[red]Extraction failed: {result.error}[/]")
        raise typer.Exit(1)
    console.print_json(json.dumps(result.bill.model_dump(), ensure_ascii=False))
    for warning in result.warnings:
        console.print(f"[yellow]⚠ {warning}[/]")
    console.print(f"tokens in/out: {result.input_tokens} / {result.output_tokens}")


@app.command("eval")
def eval_cmd(
    model: str = ModelOpt,
    dataset: Path = typer.Option(Path("dataset/synthetic"), "--dataset", "-d", exists=True),
    prompt: str = PromptOpt,
    runs: int = typer.Option(1, help="Repeat each document N times to measure stability"),
    concurrency: Optional[int] = typer.Option(
        None, help="Parallel requests (default: 4, or 1 for a model running on this machine)"),
    limit: Optional[int] = typer.Option(None, help="Only the first N documents (quick checks)"),
    tag: Optional[list[str]] = typer.Option(None, help="Only documents with this tag (repeatable)"),
    temperature: str = TempOpt,
    max_tokens: int = typer.Option(2048),
    effort: str = EffortOpt,
    timeout: float = typer.Option(120.0),
    out: Optional[Path] = typer.Option(
        None, help="Where to save the results JSON (default: eval/results, or eval/results/private for real data)"),
    no_save: bool = typer.Option(False, "--no-save", help="Print only, don't write a results file"),
):
    """Run a model over a dataset, grade every field and save the results."""
    try:
        provider = get_provider(model, _settings(temperature, max_tokens, timeout, effort))
        done = {"n": 0}

        def progress(record):
            done["n"] += 1
            mark = "[red]✗[/]" if record["error"] else "·"
            console.print(f"{mark} {record['id']} (run {record['run']}) {record['latency_s']:.1f}s", highlight=False)

        # A local model serves one request at a time anyway: asking for four only queues
        # them up and inflates the per-document latency it reports.
        workers = concurrency if concurrency is not None else (1 if provider.local else 4)
        results = run_eval(dataset, provider, prompt_version=prompt, runs=runs, concurrency=workers,
                           limit=limit, tags=tag, on_progress=progress)
    except (ProviderConfigError, ValueError, FileNotFoundError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc

    print_results(results)
    if not no_save:
        if out is None:  # results of real bills go to a git-ignored folder
            out = Path("eval/results") if results["meta"]["values_stored"] else Path("eval/results/private")
        path = save_results(results, out)
        console.print(f"\n[green]Saved:[/] {path}")


@app.command("label")
def label_cmd(
    model: str = ModelOpt,
    dataset: Path = typer.Option(Path("dataset/private"), "--dataset", "-d", help="Folder with a bills/ subfolder"),
    prompt: str = PromptOpt,
    limit: Optional[int] = typer.Option(None, help="Only the first N new bills"),
    second_model: Optional[str] = typer.Option(
        None, "--second-model", help="provider:<model> to extract with too: fields the two disagree on become doubts"),
    temperature: str = TempOpt,
    max_tokens: int = typer.Option(2048),
    effort: str = EffortOpt,
    timeout: float = typer.Option(120.0),
):
    """Step 1 of labeling real bills: a model proposes the answers (drafts) for every new bill."""
    files = unlabeled_files(dataset)
    if not files:
        console.print(f"No new bills in {dataset / BILLS_DIR} (already labeled or with a draft).")
        raise typer.Exit(0)
    if limit:
        files = files[:limit]
    settings = _settings(temperature, max_tokens, timeout, effort)
    try:
        provider = get_provider(model, settings)
        second = get_provider(second_model, settings) if second_model else None
    except (ProviderConfigError, ValueError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc

    with_second = f" and {second.spec}" if second else ""
    console.print(f"Proposing answers for {len(files)} bills with {provider.spec}{with_second}")
    skipped = []
    for file in files:
        try:
            result = extract(file, provider, prompt_version=prompt)
            # A failed second opinion is no opinion: comparing against an empty answer
            # would mark every field as disputed, which is noise, not a doubt.
            second_result = extract(file, second, prompt_version=prompt) if second else None
        except (ProviderConfigError, FileNotFoundError) as exc:  # wrong key, missing package or prompt
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(1) from exc
        except UnreadableDocument as exc:
            # No draft for a file nothing can open: it stays in the list of bills still to
            # label, so the next run reminds you about it instead of hiding the problem.
            skipped.append(file.name)
            console.print(f"[red]✗ {exc}[/]")
            continue
        second_values = second_result.bill.model_dump() if second_result and not second_result.error else None
        path = save_draft(dataset, file, provider.spec, result.bill.model_dump(), result.error,
                          dropped=result.dropped, second=second_values,
                          second_model=second.spec if second else None)
        doubts = json.loads(path.read_text(encoding="utf-8"))["doubts"]
        mark = "[red]✗[/]" if result.error else "·"
        console.print(f"{mark} {file.name} {result.latency_s:.1f}s · {len(doubts)} fields to check",
                      highlight=False)
    if skipped:
        files_word = "file" if len(skipped) == 1 else "files"
        console.print(f"\n[yellow]No draft for {len(skipped)} unreadable {files_word}, "
                      f"still to be labeled:[/] " + ", ".join(skipped))
    console.print(f"\n[green]Drafts saved in {dataset / DRAFTS_DIR}.[/] Now check them: billmeter review -d {dataset}")


@app.command("review")
def review_cmd(
    dataset: Path = typer.Option(Path("dataset/private"), "--dataset", "-d"),
    open_files: bool = typer.Option(True, "--open/--no-open", help="Open each bill in your default viewer"),
    hash_values: bool = typer.Option(True, "--hash/--plain",
                                     help="Store codes and names as HMAC (needs EVAL_HMAC_KEY) or in clear text"),
    keep_address: bool = typer.Option(False, "--keep-address",
                                      help="Also store the supply address (in clear text) so it can be graded"),
):
    """Step 2 of labeling real bills: check every draft and save the confirmed answers in expected.json."""
    drafts = drafts_by_doubt(dataset)  # most doubtful first: spend your attention where it pays
    if not drafts:
        console.print(f"No drafts to review in {dataset / DRAFTS_DIR}. Create them with: billmeter label -m <model>")
        raise typer.Exit(0)
    hmac_key = os.environ.get("EVAL_HMAC_KEY")
    if hash_values and not hmac_key:
        console.print("[red]Set EVAL_HMAC_KEY in .env to hash sensitive values (or use --plain).[/]\n"
                      "Create one with: python -c \"import secrets; print(secrets.token_hex(32))\"")
        raise typer.Exit(1)

    console.print("For each field: [bold]Enter[/] = keep the proposal · [bold]-[/] = not on the bill · "
                  "[bold]?[/] = don't grade · or type the right value.")
    console.print(f"[yellow]⚠[/] marks a field to check carefully. {len(drafts)} drafts, most doubtful first.")
    for draft_file, draft in drafts:
        bill = dataset / draft["file"]
        doubts = draft.get("doubts") or {}  # drafts written before doubts existed simply have none
        console.rule(f"[bold]{draft['file']}[/] (proposed by {draft['model']})")
        if draft.get("error"):
            console.print(f"[yellow]The model failed on this bill ({draft['error']}): fill in the fields yourself.[/]")
        elif doubts:
            console.print(f"[yellow]{len(doubts)} fields to check carefully:[/]")
            for name, reasons in doubts.items():
                console.print(f"  [yellow]⚠ {name}[/]: {' · '.join(reasons)}", highlight=False)
        if open_files and bill.exists():
            typer.launch(str(bill))

        reviewed = {}
        for name in EXTRACTED_FIELDS:
            proposed = draft["proposed"].get(name)
            label = f"⚠ {name}" if name in doubts else name  # click prompts are plain text, so no colour here
            reviewed[name] = parse_answer(
                typer.prompt(f"{label} [{proposed}]", default="", show_default=False), proposed)
        while True:
            try:
                expected = build_expected(reviewed, hash_values, keep_address, hmac_key)
                break
            except InvalidValue as exc:
                console.print(f"[red]{exc}[/]")
                reviewed[exc.field] = parse_answer(typer.prompt(f"{exc.field} (- = not on the bill, ? = skip)",
                                                                default="", show_default=False), None)

        action = typer.prompt("Save this bill? [y]es / [s]kip for now / [q]uit", default="y").strip().lower()
        if action.startswith("q"):
            break
        if not action.startswith("y"):
            continue
        tags = [t.strip() for t in typer.prompt("Tags, comma separated", default=",".join(default_tags(draft["file"])))
                .split(",") if t.strip()]
        case_id = add_case(dataset, draft["file"], expected, tags)
        draft_file.unlink()
        console.print(f"[green]Saved as case {case_id}[/] ({len(expected)} graded fields)")

    left = len(pending_drafts(dataset))
    console.print(f"\n{left} drafts left." if left else "\nAll drafts reviewed. Measure a model: "
                  f"billmeter eval -m <provider:model> -d {dataset}")


@app.command("compare")
def compare_cmd(
    baseline: Path = typer.Argument(..., exists=True, help="Results JSON of the reference run"),
    candidate: Path = typer.Argument(..., exists=True, help="Results JSON of the new run"),
):
    """Compare two eval results: per-field deltas, regressions and improvements."""
    print_comparison(compare_results(load_results(baseline), load_results(candidate)))


@app.command("report")
def report_cmd(
    results: Path = typer.Argument(..., exists=True, help="Results JSON of the run to report on"),
    baseline: Optional[Path] = typer.Argument(
        None, exists=True, help="Results JSON of an earlier run: the report also compares the two"),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Where to write the HTML file"),
):
    """Write a self-contained HTML report of a run, and of its difference from a baseline."""
    candidate = load_results(results)
    comparison = compare_results(load_results(baseline), candidate) if baseline else None
    if out is None:
        # Reports of real bills can quote extracted values, so they follow the same rule
        # as the results they come from: a git-ignored folder, never the published one.
        real = not candidate["meta"].get("values_stored")
        folder = Path("eval/reports/private") if real else Path("eval/reports")
        name = results.stem if baseline is None else f"{results.stem}_vs_{baseline.stem}"
        out = folder / f"{name}.html"
    path = html_report.write(candidate, out, comparison)
    console.print(f"[green]Report written:[/] {path}")


@app.command("generate")
def generate_cmd(
    out: Path = typer.Option(Path("dataset/synthetic"), help="Output folder"),
    n: int = typer.Option(24, help="Number of synthetic bills"),
    photos: int = typer.Option(6, help="How many of them also get a phone-photo version"),
    seed: int = typer.Option(42, help="Same seed = same dataset"),
):
    """Generate the synthetic dataset (fake bills + expected.json)."""
    try:
        from .synthetic import generate_dataset
    except ImportError as exc:
        console.print("[red]Missing packages. Run: uv sync --extra dataset[/]")
        raise typer.Exit(1) from exc
    path = generate_dataset(out, n=n, photos=photos, seed=seed)
    console.print(f"[green]Dataset written:[/] {path}")


@app.command("providers")
def providers_cmd():
    """List supported providers, whether they are configured, and where to find their models."""
    table = Table("provider", "env variable", "configured")
    table.add_column("where to find models", overflow="fold")
    for name in available_providers():
        env = PROVIDER_ENV.get(name, "")
        var = env.split(" ")[0]
        configured = "local" if "optional" in env else ("yes" if os.environ.get(var) else "no")
        table.add_row(name, env, configured, MODEL_LISTS.get(name, ""))
    console.print(table)
    console.print("Use --model provider:<model>. BillMeter does not pick a model for you:")
    console.print("choose any model with vision (image input) support from these lists.")


@app.command("hash")
def hash_cmd(
    field: str = typer.Argument(..., help="Field name, e.g. holder_tax_code"),
    value: str = typer.Argument(..., help="The real value to hash"),
):
    """HMAC a sensitive expected value for a private dataset (needs EVAL_HMAC_KEY)."""
    try:
        console.print(hmac_value(field, value))
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc


if __name__ == "__main__":
    app()
