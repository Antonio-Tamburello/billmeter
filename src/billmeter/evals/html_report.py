"""Turn a results file (or two of them) into one self-contained HTML page.

The terminal report is for the moment you run an eval; this one is for later:
a file you can keep next to the results JSON, reopen in a month, or send to
someone who does not have the repository. So everything lives in that single
file: the CSS is inline, there is no JavaScript, no font, no image and no link
to anything outside it. It opens from the file system, offline, forever.

No template engine either: the standard library escapes and joins strings well
enough for a page this size, and one dependency less is one dependency less.
Values always go through `esc`, because a model can answer anything at all and
a results file must never be able to inject markup into the page.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Optional

from .. import __version__
from .graders import DANGEROUS_OUTCOMES
from .report import LOWER_IS_BETTER, fmt, pct

# Metrics shown as cards at the top, in reading order: how good, how dangerous, how expensive.
SUMMARY_CARDS = [
    ("field_accuracy", "field accuracy"),
    ("filled_accuracy", "right where the bill has a value"),
    ("empty_accuracy", "quiet where it has none"),
    ("dangerous_rate", "dangerous errors"),
    ("docs_fully_correct", "fully correct documents"),
    ("error_rate", "extraction errors"),
    ("stability", "stability across runs"),
    ("latency_p50_s", "latency p50"),
    ("latency_p95_s", "latency p95"),
    ("cost_per_doc_usd", "cost per document"),
]
META_ROWS = [
    ("model", "model"),
    ("prompt_version", "prompt"),
    ("prompt_sha256", "prompt hash"),
    ("dataset", "dataset"),
    ("dataset_sha256", "dataset hash"),
    ("n_cases", "documents"),
    ("runs", "runs per document"),
    ("temperature", "temperature"),
    ("timestamp", "run date (UTC)"),
    ("git_commit", "git commit"),
    ("billmeter_version", "billmeter version"),
]
FAILING_OUTCOMES = ("wrong", "hallucinated", "missing")
MAX_FAILED_DOCS = 200

# Colours are defined once per theme; the dark variant follows the reader's system setting.
STYLE = """
:root {
  --bg: #fbfaf8; --card: #ffffff; --ink: #1c1b19; --muted: #6a6762; --line: #e4e1db;
  --good: #1d7a4c; --bad: #b3261e; --warn: #9a6400; --accent: #2f5fd0;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #17181a; --card: #1f2023; --ink: #eceae6; --muted: #9b968e; --line: #32343a;
    --good: #5bc48c; --bad: #ff8a80; --warn: #e0b250; --accent: #8fb0ff;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 15px/1.5 -apple-system, BlinkMacSystemFont,
       "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
main { max-width: 68rem; margin: 0 auto; padding: 1.5rem 1rem 4rem; }
h1 { font-size: 1.5rem; margin: 0 0 .25rem; }
h2 { font-size: 1.1rem; margin: 2.5rem 0 .75rem; padding-bottom: .35rem; border-bottom: 1px solid var(--line); }
p.sub { color: var(--muted); margin: 0 0 1.5rem; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(10rem, 1fr)); gap: .75rem; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: .6rem; padding: .8rem .9rem; }
.card .label { color: var(--muted); font-size: .78rem; text-transform: uppercase; letter-spacing: .04em; }
.card .value { font-size: 1.5rem; font-variant-numeric: tabular-nums; margin-top: .2rem; }
.card .delta { font-size: .85rem; margin-top: .15rem; }
.scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: .4rem .55rem; border-bottom: 1px solid var(--line); white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
thead th { color: var(--muted); font-weight: 600; font-size: .8rem; position: sticky; top: 0; background: var(--bg); }
tbody tr:hover { background: var(--card); }
dl.meta { display: grid; grid-template-columns: max-content 1fr; gap: .3rem 1rem; margin: 0; }
dl.meta dt { color: var(--muted); }
dl.meta dd { margin: 0; overflow-wrap: anywhere; }
.doc { background: var(--card); border: 1px solid var(--line); border-radius: .6rem;
       padding: .7rem .9rem; margin-bottom: .6rem; }
.doc h3 { font-size: .95rem; margin: 0 0 .5rem; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.doc ul { margin: 0; padding-left: 1.1rem; }
.doc li { margin-bottom: .35rem; overflow-wrap: anywhere; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .88em;
       background: var(--bg); border: 1px solid var(--line); border-radius: .25rem; padding: 0 .25rem; }
.good { color: var(--good); } .bad { color: var(--bad); } .warn { color: var(--warn); }
.muted { color: var(--muted); }
.tag { font-size: .75rem; text-transform: uppercase; letter-spacing: .04em; }
footer { color: var(--muted); font-size: .85rem; margin-top: 3rem;
         border-top: 1px solid var(--line); padding-top: .75rem; }
@media (max-width: 30rem) {
  main { padding: 1rem .75rem 3rem; }
  dl.meta { grid-template-columns: 1fr; gap: 0 0; }
  dl.meta dd { margin-bottom: .5rem; }
}
"""


def esc(value: Any) -> str:
    """Everything that reaches the page goes through here. None becomes a dash."""
    if value is None:
        return "—"
    return html.escape(str(value), quote=True)


def _outcome_class(outcome: str) -> str:
    return "bad" if outcome in DANGEROUS_OUTCOMES else "warn"


def _delta(key: str, before: Any, after: Any) -> str:
    """The change between two runs, already coloured by whether it is good news."""
    if before is None or after is None:
        return ""
    diff = after - before
    if abs(diff) < 1e-9:
        return '<span class="muted">no change</span>'
    if key.endswith("_usd"):
        shown = f"{diff:+.4f}$"
    elif key.endswith("_s"):
        shown = f"{diff:+.2f}s"
    else:
        shown = f"{diff * 100:+.1f}pt"
    better = diff < 0 if key in LOWER_IS_BETTER else diff > 0
    return f'<span class="{"good" if better else "bad"}">{esc(shown)}</span>'


def _meta_block(meta: dict[str, Any]) -> str:
    rows = "".join(f"<dt>{esc(label)}</dt><dd>{esc(meta.get(key))}</dd>" for key, label in META_ROWS)
    return f'<dl class="meta">{rows}</dl>'


def _cards(summary: dict[str, Any], baseline: Optional[dict[str, Any]] = None) -> str:
    cards = []
    for key, label in SUMMARY_CARDS:
        value = summary.get(key)
        if value is None and key == "stability":
            continue  # only measured with --runs > 1: an empty card would just be noise
        tone = ' class="bad"' if key == "dangerous_rate" and value else ""
        delta = _delta(key, baseline.get(key), value) if baseline else ""
        delta_line = f'<div class="delta">{delta}</div>' if delta else ""
        cards.append(
            f'<div class="card"><div class="label">{esc(label)}</div>'
            f'<div class="value"{tone}>{esc(fmt(key, value))}</div>{delta_line}</div>'
        )
    tokens = (f'<div class="card"><div class="label">tokens in / out</div>'
              f'<div class="value">{summary["input_tokens"]:,} / {summary["output_tokens"]:,}</div></div>')
    return f'<div class="cards">{"".join(cards)}{tokens}</div>'


def _field_table(summary: dict[str, Any], comparison: Optional[dict[str, Any]] = None) -> str:
    head = ["field", "accuracy", "dangerous", "missing rate", "correct", "correct null", "missing", "wrong",
            "hallucinated"]
    if comparison:
        head[1:3] = ["accuracy", "Δ accuracy", "dangerous", "Δ dangerous"]
    rows = []
    for field, s in summary["fields"].items():
        danger = f'<span class="bad">{esc(pct(s["dangerous_rate"]))}</span>' if s["dangerous_rate"] else \
            esc(pct(s["dangerous_rate"]))
        cells = [f"<code>{esc(field)}</code>", esc(pct(s["accuracy"]))]
        if comparison:
            before = (comparison["fields"].get(field) or {}).get("accuracy", (None, None))[0]
            dan_before = (comparison["fields"].get(field) or {}).get("dangerous_rate", (None, None))[0]
            cells += [_delta("accuracy", before, s["accuracy"]) or "—", danger,
                      _delta("dangerous_rate", dan_before, s["dangerous_rate"]) or "—"]
        else:
            cells.append(danger)
        cells += [esc(pct(s["missing_rate"])), esc(s["correct"]), esc(s["correct_null"]), esc(s["missing"]),
                  esc(s["wrong"]), esc(s["hallucinated"])]
        rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    header = "".join(f"<th>{esc(h)}</th>" for h in head)
    return f'<div class="scroll"><table><thead><tr>{header}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def _values_line(record: dict[str, Any], field: str) -> str:
    """expected / raw / normalized, so a wrong field says who got it wrong.

    `raw` is the model's own answer before normalize touched it: when raw is right
    and normalized is empty, the extraction was fine and validation dropped it.
    """
    if "values" not in record:  # a run on real data stores outcomes only
        return ""
    v = record["values"][field]
    parts = [f"expected <code>{esc(v['expected'])}</code>"]
    if "raw" in v:
        parts.append(f"raw <code>{esc(v['raw'])}</code>")
    parts.append(f"normalized <code>{esc(v['actual'])}</code>")
    if field in record.get("dropped", {}):
        parts.append('<span class="muted">dropped by validation</span>')
    return " · " + " · ".join(parts)


def _failures(results: dict[str, Any]) -> str:
    failed = [r for r in results["cases"]
              if r["error"] or any(o in FAILING_OUTCOMES for o in r["outcomes"].values())]
    if not failed:
        return '<p class="good">Every field of every document was correct.</p>'

    blocks = []
    for record in failed[:MAX_FAILED_DOCS]:
        title = f"{record['id']} · run {record['run']}"
        if record["error"]:
            blocks.append(f'<div class="doc"><h3>{esc(title)}</h3>'
                          f'<p class="bad">extraction failed: {esc(record["error"])}</p></div>')
            continue
        items = [f'<li><code>{esc(field)}</code> '
                 f'<span class="tag {_outcome_class(outcome)}">{esc(outcome)}</span>{_values_line(record, field)}</li>'
                 for field, outcome in record["outcomes"].items() if outcome in FAILING_OUTCOMES]
        blocks.append(f'<div class="doc"><h3>{esc(title)}</h3><ul>{"".join(items)}</ul></div>')

    extra = ""
    if len(failed) > MAX_FAILED_DOCS:
        extra = f'<p class="muted">… and {len(failed) - MAX_FAILED_DOCS} more documents (see the results JSON).</p>'
    note = ""
    if not results["meta"].get("values_stored"):
        note = ('<p class="muted">This run is on data marked as real, so no extracted value is stored: '
                'the outcomes are all this file can show.</p>')
    return note + "".join(blocks) + extra


def _changed_list(entries: list[dict[str, Any]], tone: str) -> str:
    if not entries:
        return '<p class="muted">none</p>'
    items = "".join(f'<li><code>{esc(e["id"])}</code> <code>{esc(e["field"])}</code> '
                    f'{esc(pct(e["before"]))} → <span class="{tone}">{esc(pct(e["after"]))}</span></li>'
                    for e in entries[:MAX_FAILED_DOCS])
    rest = ""
    if len(entries) > MAX_FAILED_DOCS:
        rest = f'<p class="muted">… and {len(entries) - MAX_FAILED_DOCS} more.</p>'
    return f'<div class="doc"><ul>{items}</ul></div>{rest}'


def _comparison_block(comparison: dict[str, Any]) -> str:
    b, c = comparison["baseline"], comparison["candidate"]
    warnings = "".join(f'<p class="bad">⚠ {esc(w)}</p>' for w in comparison["warnings"])
    rows = "".join(
        f"<tr><td><code>{esc(key)}</code></td><td>{esc(fmt(key, before))}</td>"
        f"<td>{esc(fmt(key, after))}</td><td>{_delta(key, before, after) or '—'}</td></tr>"
        for key, (before, after) in comparison["overall"].items()
    )
    return (
        f"<h2>Compared with the baseline</h2>{warnings}"
        f'<dl class="meta">'
        f"<dt>baseline</dt><dd>{esc(b['model'])} · prompt {esc(b['prompt_version'])} "
        f"({esc(b['prompt_sha256'])}) · {esc(b['timestamp'])}</dd>"
        f"<dt>candidate</dt><dd>{esc(c['model'])} · prompt {esc(c['prompt_version'])} "
        f"({esc(c['prompt_sha256'])}) · {esc(c['timestamp'])}</dd></dl>"
        f'<div class="scroll"><table><thead><tr><th>metric</th><th>baseline</th><th>candidate</th>'
        f"<th>delta</th></tr></thead><tbody>{rows}</tbody></table></div>"
        f"<h2>Regressions ({len(comparison['regressions'])})</h2>"
        f'<p class="muted">Correct in the baseline, not anymore.</p>'
        f'{_changed_list(comparison["regressions"], "bad")}'
        f"<h2>Improvements ({len(comparison['improvements'])})</h2>"
        f'{_changed_list(comparison["improvements"], "good")}'
    )


def render(results: dict[str, Any], comparison: Optional[dict[str, Any]] = None) -> str:
    """The whole page, as one string."""
    meta, summary = results["meta"], results["summary"]
    title = f"BillMeter · {meta['model']} · {meta['dataset']}"
    baseline_summary = comparison["overall"] if comparison else None
    before = {k: v[0] for k, v in baseline_summary.items()} if baseline_summary else None

    body = (
        f"<h1>{esc(title)}</h1>"
        f'<p class="sub">prompt {esc(meta["prompt_version"])} ({esc(meta["prompt_sha256"])}) · '
        f"dataset {esc(meta['dataset'])} ({esc(meta['dataset_sha256'])}) · {esc(meta['timestamp'])}</p>"
        f"<h2>Summary</h2>{_cards(summary, before)}"
        f"<h2>Per field</h2>{_field_table(summary, comparison)}"
        f"{_comparison_block(comparison) if comparison else ''}"
        f"<h2>Documents with errors</h2>{_failures(results)}"
        f"<h2>Run</h2>{_meta_block(meta)}"
        f"<footer>Generated by billmeter {esc(__version__)}. "
        f"Self-contained file: no script, no network, nothing outside it.</footer>"
    )
    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{esc(title)}</title><style>{STYLE}</style></head>"
        f"<body><main>{body}</main></body></html>\n"
    )


def write(results: dict[str, Any], path: str | Path, comparison: Optional[dict[str, Any]] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(results, comparison), encoding="utf-8")
    return path
