"""Smoke test against REAL APIs. Skipped unless you ask for it, because it costs money.

    BILLMETER_LIVE_MODELS="ollama:<model>,anthropic:<model>" uv run pytest tests/test_live.py -s

It extracts one synthetic bill per model and checks the most important codes.
"""

import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from billmeter.extractor import extract
from billmeter.providers import get_provider

load_dotenv()
MODELS = [m.strip() for m in os.environ.get("BILLMETER_LIVE_MODELS", "").split(",") if m.strip()]
DATASET = Path(__file__).resolve().parents[1] / "dataset" / "synthetic"

pytestmark = pytest.mark.skipif(not MODELS, reason="set BILLMETER_LIVE_MODELS to run live tests")


@pytest.mark.parametrize("model", MODELS)
def test_live_extraction(model):
    case = json.loads((DATASET / "expected.json").read_text())["cases"][0]
    result = extract(DATASET / case["file"], get_provider(model))
    print(f"\n{model}: {result.latency_s:.1f}s, tokens {result.input_tokens}/{result.output_tokens}")
    print(json.dumps(result.bill.model_dump(), indent=2, ensure_ascii=False))
    assert result.ok, result.error
    assert result.bill.pod == case["expected"]["pod"]
