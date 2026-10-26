"""Console rendering of eval results and comparisons (rich tables)."""

from __future__ import annotations

from typing import Any, Optional

from rich.console import Console
from rich.table import Table

console = Console()

# For these metrics a LOWER value is better.
LOWER_IS_BETTER = {"dangerous_rate", "error_rate", "latency_p50_s", "latency_p95_s", "cost_per_doc_usd"}


def pct(value: Optional[float]) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def fmt(key: str, value: Any) -> str:
    if value is None:
        return "-"
    if key.endswith("_s"):
        return f"{value:.2f}s"
    if key.endswith("_usd"):
        return f"${value:.4f}"
    return pct(value)


def print_results(results: dict[str, Any]) -> None:
    meta, summary = results["meta"], results["summary"]
    console.rule(f"[bold]{meta['model']}[/] · prompt {meta['prompt_version']} · dataset {meta['dataset']}")

    table = Table("field", "accuracy", "on filled", "on empty", "dangerous", "missing", "correct",
                  "correct_null", "missing#", "wrong", "halluc.", title="Per-field results")
    for field, s in summary["fields"].items():
        danger = pct(s["dangerous_rate"])
        if s["dangerous_rate"] > 0:
            danger = f"[red]{danger}[/]"
        table.add_row(field, pct(s["accuracy"]), pct(s["filled_accuracy"]), pct(s["empty_accuracy"]),
                      danger,
                      pct(s["missing_rate"]), str(s["correct"]), str(s["correct_null"]),
                      str(s["missing"]), str(s["wrong"]), str(s["hallucinated"]))
    console.print(table)

    lines = [
        f"documents: {summary['n_docs']}   field accuracy: [bold]{pct(summary['field_accuracy'])}[/]"
        f"   right where the bill has a value: [bold]{pct(summary['filled_accuracy'])}[/]"
        f"   quiet where it has none: [bold]{pct(summary['empty_accuracy'])}[/]"
        f"   dangerous: [bold red]{pct(summary['dangerous_rate'])}[/]"
        f"   fully correct docs: {pct(summary['docs_fully_correct'])}",
        f"errors: {pct(summary['error_rate'])}   latency p50/p95: "
        f"{fmt('x_s', summary['latency_p50_s'])} / {fmt('x_s', summary['latency_p95_s'])}"
        f"   tokens in/out: {summary['input_tokens']:,} / {summary['output_tokens']:,}"
        f"   cost: {fmt('x_usd', summary.get('cost_usd'))}",
    ]
    if summary.get("stability") is not None:
        lines.append(f"stability across runs: {pct(summary['stability'])}")
    for line in lines:
        console.print(line)

    failures = [r for r in results["cases"] if r["error"] or
                any(o in ("wrong", "hallucinated", "missing") for o in r["outcomes"].values())]
    if failures:
        console.print("\n[bold]Failures[/] (the most useful part: read these before changing the prompt)")
        for r in failures[:40]:
            if r["error"]:
                console.print(f"  {r['id']} run {r['run']}: [red]ERROR[/] {r['error']}")
                continue
            for field, outcome in r["outcomes"].items():
                if outcome in ("wrong", "hallucinated", "missing"):
                    console.print(f"  {r['id']} run {r['run']}: {field} [yellow]{outcome}[/]{_values(r, field)}")
        if len(failures) > 40:
            console.print(f"  ... and {len(failures) - 40} more documents (see the JSON file)")


def _values(record: dict[str, Any], field: str) -> str:
    """expected / raw / normalized for one failed field, when the run stored values.

    The raw answer is the model's own words before normalize touched them, so the
    three together say whether the model misread the bill (raw is already wrong) or
    our validation dropped a good value (raw is right, normalized is empty).
    """
    if "values" not in record:  # real data: nothing is stored, only the outcome
        return ""
    v = record["values"][field]
    raw = f" raw={v['raw']!r}" if "raw" in v else ""
    dropped = " [dim](dropped by validation)[/]" if field in record.get("dropped", {}) else ""
    return f"  expected={v['expected']!r}{raw} normalized={v['actual']!r}{dropped}"


def _delta(key: str, before: Any, after: Any) -> str:
    if before is None or after is None:
        return ""
    diff = after - before
    if abs(diff) < 1e-9:
        return "="
    better = diff < 0 if key in LOWER_IS_BETTER else diff > 0
    color = "green" if better else "red"
    shown = f"{diff:+.2f}s" if key.endswith("_s") else f"{diff * 100:+.1f}pt"
    if key.endswith("_usd"):
        shown = f"{diff:+.4f}$"
    return f"[{color}]{shown}[/]"


def print_comparison(cmp: dict[str, Any]) -> None:
    b, c = cmp["baseline"], cmp["candidate"]
    console.rule("[bold]Comparison[/]")
    console.print(f"baseline : {b['model']} · prompt {b['prompt_version']} ({b['prompt_sha256']}) · {b['timestamp']}")
    console.print(f"candidate: {c['model']} · prompt {c['prompt_version']} ({c['prompt_sha256']}) · {c['timestamp']}")
    for warning in cmp["warnings"]:
        console.print(f"[bold red]⚠ {warning}[/]")

    table = Table("metric", "baseline", "candidate", "delta", title="Overall")
    for key, (before, after) in cmp["overall"].items():
        table.add_row(key, fmt(key, before), fmt(key, after), _delta(key, before, after))
    console.print(table)

    table = Table("field", "accuracy before", "after", "delta", "dangerous before", "after", "delta",
                  title="Per field")
    for field, values in cmp["fields"].items():
        (acc_b, acc_a), (dan_b, dan_a) = values["accuracy"], values["dangerous_rate"]
        table.add_row(field, pct(acc_b), pct(acc_a), _delta("accuracy", acc_b, acc_a),
                      pct(dan_b), pct(dan_a), _delta("dangerous_rate", dan_b, dan_a))
    console.print(table)

    console.print(f"\n[bold red]Regressions ({len(cmp['regressions'])})[/]: correct before, not anymore")
    for r in cmp["regressions"][:30]:
        console.print(f"  {r['id']}  {r['field']}  {pct(r['before'])} → {pct(r['after'])}")
    console.print(f"[bold green]Improvements ({len(cmp['improvements'])})[/]")
    for r in cmp["improvements"][:30]:
        console.print(f"  {r['id']}  {r['field']}  {pct(r['before'])} → {pct(r['after'])}")
