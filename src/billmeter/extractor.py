"""The extraction pipeline: file -> provider -> JSON -> normalized bill.

    load_document  ->  to_parts (PDF or page images)  ->  provider.generate
                   ->  parse_json  ->  normalize  ->  ExtractionResult
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .documents import MAX_PAGES, load_document
from .normalize import normalize
from .prompts import load_prompt
from .providers import Provider, ProviderConfigError
from .schema import NormalizedBill, json_schema

RETRY_DELAYS_S = (2.0, 5.0)


@dataclass
class ExtractionResult:
    file: str
    model: str
    bill: NormalizedBill
    raw: Optional[dict[str, Any]] = None
    raw_text: str = ""
    latency_s: float = 0.0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    attempts: int = 0
    error: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    #: field -> raw value the model returned and validation discarded
    dropped: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None


def parse_json(text: str) -> dict[str, Any]:
    """Accept bare JSON, JSON inside ```json fences, or JSON surrounded by text."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in the model answer")
    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("the model answer is not a JSON object")
    return value


def extract(
    path: str | Path,
    provider: Provider,
    prompt_version: str = "v2",
    prompt: Optional[str] = None,
    max_pages: int = MAX_PAGES,
) -> ExtractionResult:
    """Extract one bill.

    Never raises for model errors: network failures, rate limits and invalid JSON
    end up in `result.error` after the retries. A file that cannot be read at all
    raises UnreadableDocument instead, because no retry and no model can fix it.
    """
    document = load_document(path)
    prompt_text = prompt if prompt is not None else load_prompt(prompt_version)
    parts = document.to_parts(pdf_native=provider.pdf_native, max_pages=max_pages)
    schema = json_schema()

    result = ExtractionResult(file=str(document.path), model=provider.spec, bill=NormalizedBill())
    started = time.perf_counter()

    for attempt in range(len(RETRY_DELAYS_S) + 1):
        result.attempts = attempt + 1
        try:
            response = provider.generate(parts, prompt_text, schema, source=document.path)
            result.raw_text = response.text
            result.input_tokens = response.input_tokens
            result.output_tokens = response.output_tokens
            result.raw = parse_json(response.text)
            result.bill = normalize(result.raw)
            result.error = None
            break
        except ProviderConfigError:
            raise  # wrong key / missing package: stop immediately, retrying is pointless
        except Exception as exc:  # network errors, rate limits, invalid JSON...
            result.error = f"{type(exc).__name__}: {exc}"
            if attempt < len(RETRY_DELAYS_S):
                time.sleep(RETRY_DELAYS_S[attempt])

    result.latency_s = round(time.perf_counter() - started, 3)
    if result.raw is not None:
        result.dropped = _dropped_values(result.raw, result.bill)
        result.warnings = [f"{name}: '{value}' discarded by validation" for name, value in result.dropped.items()]
    return result


def _dropped_values(raw: dict[str, Any], bill: NormalizedBill) -> dict[str, Any]:
    """Fields the model filled but validation discarded.

    Kept per field, not only as a message: it is the difference between "the model
    did not read it" and "the model read it and we threw it away", which is what you
    need to know when a prompt or a validation rule is under suspicion.
    """
    return {
        name: value for name, value in raw.items()
        if value not in (None, "") and getattr(bill, name, "missing") is None
    }
