"""Compare two result files: what improved, what regressed, and where."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .graders import OK_OUTCOMES


def load_results(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _ok_share(results: dict[str, Any]) -> dict[tuple[str, str], float]:
    """(case id, field) -> share of runs where the field was correct."""
    counts: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for record in results["cases"]:
        for field, outcome in record["outcomes"].items():
            counts[(record["id"], field)].append(outcome in OK_OUTCOMES)
    return {key: sum(v) / len(v) for key, v in counts.items()}


def compare(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    warnings = []
    if baseline["meta"]["dataset_sha256"] != candidate["meta"]["dataset_sha256"]:
        warnings.append("Different dataset versions: the numbers are NOT directly comparable.")

    a, b = _ok_share(baseline), _ok_share(candidate)
    common = sorted(set(a) & set(b))
    regressions = [{"id": k[0], "field": k[1], "before": a[k], "after": b[k]} for k in common if b[k] < a[k]]
    improvements = [{"id": k[0], "field": k[1], "before": a[k], "after": b[k]} for k in common if b[k] > a[k]]

    fields = {}
    for field in baseline["summary"]["fields"]:
        before = baseline["summary"]["fields"][field]
        after = candidate["summary"]["fields"].get(field)
        if after:
            fields[field] = {
                "accuracy": (before["accuracy"], after["accuracy"]),
                "dangerous_rate": (before["dangerous_rate"], after["dangerous_rate"]),
            }

    keys = ["field_accuracy", "filled_accuracy", "empty_accuracy", "dangerous_rate", "docs_fully_correct", "error_rate",
            "latency_p50_s", "latency_p95_s", "cost_per_doc_usd", "stability"]
    overall = {k: (baseline["summary"].get(k), candidate["summary"].get(k)) for k in keys}

    return {
        "baseline": baseline["meta"],
        "candidate": candidate["meta"],
        "warnings": warnings,
        "overall": overall,
        "fields": fields,
        "regressions": regressions,
        "improvements": improvements,
    }
