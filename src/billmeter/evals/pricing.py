"""Cost estimate from token counts and the prices YOU put in pricing.json.

Prices change often, so none are hard-coded: copy them from the provider's
pricing page into pricing.json (USD per million tokens). Models without a
price show tokens only. "provider:*" entries act as a wildcard.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

#: Copy shipped inside the wheel, so costs still work outside the repository.
PACKAGED_PRICING = Path(__file__).resolve().parents[1] / "pricing.json"


def pricing_file() -> Path:
    """`pricing.json` in the folder you run from, else the copy shipped with the package."""
    local = Path.cwd() / "pricing.json"
    return local if local.is_file() else PACKAGED_PRICING


def _prices() -> dict[str, dict[str, float]]:
    path = pricing_file()
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def cost_usd(model_spec: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    prices = _prices()
    entry = prices.get(model_spec) or prices.get(model_spec.split(":", 1)[0] + ":*")
    if not entry:
        return None
    total = input_tokens / 1e6 * entry["input_per_mtok"] + output_tokens / 1e6 * entry["output_per_mtok"]
    return round(total, 5)
