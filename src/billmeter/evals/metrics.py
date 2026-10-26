"""Aggregate per-field outcomes into the numbers you compare between runs."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import median
from typing import Any, Optional

from .graders import (
    CORRECT,
    CORRECT_NULL,
    DANGEROUS_OUTCOMES,
    HALLUCINATED,
    MISSING,
    OK_OUTCOMES,
    OUTCOMES,
    WRONG,
)

#: Outcomes of a field the document carries a value for, and of one it leaves empty.
FILLED_OUTCOMES = (CORRECT, MISSING, WRONG)
EMPTY_OUTCOMES = (CORRECT_NULL, HALLUCINATED)


def percentile(values: list[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1)))
    return round(ordered[index], 3)


def _share(counts: Counter, hits: str, over: tuple[str, ...]) -> Optional[float]:
    total = sum(counts[o] for o in over)
    return round(counts[hits] / total, 4) if total else None


def _filled_accuracy(counts: Counter) -> Optional[float]:
    """Did we get the right answer where the bill has one? (`correct` / fields with a value)

    `field_accuracy` also rewards `correct_null`, which here is about a third of
    every grade: a model answering null to everything already scores 31%. These
    two numbers have no such floor, and they fail in opposite directions.
    """
    return _share(counts, CORRECT, FILLED_OUTCOMES)


def _empty_accuracy(counts: Counter) -> Optional[float]:
    """Did we keep quiet where the bill has nothing? (`correct_null` / fields left empty)

    A model can read well and still be useless by inventing a value for every
    field the document does not carry, which `filled_accuracy` cannot see.
    """
    return _share(counts, CORRECT_NULL, EMPTY_OUTCOMES)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    per_field: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        for field, outcome in record["outcomes"].items():
            per_field[field][outcome] += 1

    fields: dict[str, Any] = {}
    for field, counts in per_field.items():
        total = sum(counts.values())
        fields[field] = {
            "n": total,
            "accuracy": round(sum(counts[o] for o in OK_OUTCOMES) / total, 4),
            "filled_accuracy": _filled_accuracy(counts),
            "empty_accuracy": _empty_accuracy(counts),
            "dangerous_rate": round(sum(counts[o] for o in DANGEROUS_OUTCOMES) / total, 4),
            "missing_rate": round(counts[MISSING] / total, 4),
            **{o: counts[o] for o in OUTCOMES},
        }

    all_outcomes = [o for r in records for o in r["outcomes"].values()]
    totals = Counter(all_outcomes)
    latencies = [r["latency_s"] for r in records if r["error"] is None]
    n = len(records)

    return {
        "n_docs": n,
        "field_accuracy": round(sum(o in OK_OUTCOMES for o in all_outcomes) / max(1, len(all_outcomes)), 4),
        "filled_accuracy": _filled_accuracy(totals),
        "empty_accuracy": _empty_accuracy(totals),
        "dangerous_rate": round(sum(o in DANGEROUS_OUTCOMES for o in all_outcomes) / max(1, len(all_outcomes)), 4),
        "docs_fully_correct": round(
            sum(all(o in OK_OUTCOMES for o in r["outcomes"].values()) for r in records) / max(1, n), 4
        ),
        "error_rate": round(sum(r["error"] is not None for r in records) / max(1, n), 4),
        "latency_p50_s": round(median(latencies), 3) if latencies else None,
        "latency_p95_s": percentile(latencies, 95),
        "input_tokens": sum(r["input_tokens"] or 0 for r in records),
        "output_tokens": sum(r["output_tokens"] or 0 for r in records),
        "outcomes": {o: totals[o] for o in OUTCOMES},
        "fields": fields,
        "stability": _stability(records),
    }


def _stability(records: list[dict[str, Any]]) -> Optional[float]:
    """With runs > 1: share of (document, field) pairs whose outcome never changed between runs."""
    by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        by_case[record["id"]].append(record["outcomes"])
    pairs = stable = 0
    for outcomes_per_run in by_case.values():
        if len(outcomes_per_run) < 2:
            continue
        for field in outcomes_per_run[0]:
            pairs += 1
            stable += len({run.get(field) for run in outcomes_per_run}) == 1
    return round(stable / pairs, 4) if pairs else None
