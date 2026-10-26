"""Run a model over a dataset, grade every field, save the results to JSON.

A dataset is a folder with an `expected.json`:

    {
      "name": "synthetic",
      "contains_real_data": false,
      "cases": [
        {"id": "syn-001", "file": "bills/syn-001.pdf", "tags": ["power"], "expected": {...}}
      ]
    }

Fields missing from "expected" are not graded, so a private dataset can grade
only what you verified by hand.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .. import __version__
from ..documents import UnreadableDocument
from ..extractor import ExtractionResult, extract
from ..prompts import load_prompt, prompt_hash
from ..providers import Provider
from ..schema import NormalizedBill
from .graders import grade
from .metrics import summarize
from .pricing import cost_usd


def load_dataset(dataset_dir: str | Path) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    manifest = dataset_dir / "expected.json"
    if not manifest.exists():
        raise FileNotFoundError(f"{manifest} not found")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["_dir"] = dataset_dir
    data["_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()[:12]
    return data


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5)
        if out.returncode != 0:
            return None
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return None


def run_eval(
    dataset_dir: str | Path,
    provider: Provider,
    prompt_version: str = "v2",
    runs: int = 1,
    concurrency: int = 4,
    limit: Optional[int] = None,
    tags: Optional[list[str]] = None,
    store_values: Optional[bool] = None,
    on_progress=None,
) -> dict[str, Any]:
    dataset = load_dataset(dataset_dir)
    cases = dataset["cases"]
    if tags:
        cases = [c for c in cases if set(tags) & set(c.get("tags", []))]
    if limit:
        cases = cases[:limit]
    if store_values is None:  # never write real personal data into results by default
        store_values = not dataset.get("contains_real_data", True)

    prompt_text = load_prompt(prompt_version)
    jobs = [(case, run) for run in range(runs) for case in cases]

    def work(job: tuple[dict[str, Any], int]) -> dict[str, Any]:
        case, run = job
        file = dataset["_dir"] / case["file"]
        try:
            result = extract(file, provider, prompt=prompt_text)
        except UnreadableDocument as exc:
            # A corrupt or unsupported file is a failed extraction, not a reason to throw
            # away every call already paid for in this run: record it and carry on.
            result = ExtractionResult(file=str(file), model=provider.spec, bill=NormalizedBill(),
                                      error=f"{type(exc).__name__}: {exc}")
        actual = result.bill.model_dump()
        record: dict[str, Any] = {
            "id": case["id"],
            "run": run,
            "tags": case.get("tags", []),
            "outcomes": grade(case["expected"], actual),
            "latency_s": result.latency_s,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "attempts": result.attempts,
            "error": result.error,
        }
        if store_values:
            # Only on datasets that hold no real personal data. "raw" is what the model
            # answered before normalize touched it: with expected / raw / normalized side
            # by side, a wrong field tells you whether the model misread the bill or our
            # own validation threw a good value away.
            raw = result.raw or {}
            record["values"] = {
                name: {"expected": case["expected"][name], "raw": raw.get(name), "actual": actual.get(name)}
                for name in case["expected"]
            }
            record["raw_text"] = result.raw_text
            record["dropped"] = result.dropped
        if on_progress:
            on_progress(record)
        return record

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        records = list(pool.map(work, jobs))
    records.sort(key=lambda r: (r["id"], r["run"]))

    summary = summarize(records)
    summary["cost_usd"] = cost_usd(provider.spec, summary["input_tokens"], summary["output_tokens"])
    if summary["cost_usd"] is not None and summary["n_docs"]:
        summary["cost_per_doc_usd"] = round(summary["cost_usd"] / summary["n_docs"], 5)

    return {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": provider.spec,
            "prompt_version": prompt_version,
            "prompt_sha256": prompt_hash(prompt_text),
            "dataset": dataset.get("name", str(dataset_dir)),
            "dataset_sha256": dataset["_sha256"],
            "n_cases": len(cases),
            "runs": runs,
            "temperature": provider.settings.temperature,
            "pdf_native": provider.pdf_native,
            "git_commit": _git_commit(),
            "billmeter_version": __version__,
            "values_stored": store_values,
        },
        "summary": summary,
        "cases": records,
    }


def save_results(results: dict[str, Any], out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = results["meta"]
    stamp = meta["timestamp"].replace(":", "").replace("-", "")[:15]
    model = meta["model"].replace(":", "_").replace("/", "_")
    path = out_dir / f"{stamp}_{model}_{meta['prompt_version']}.json"
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
