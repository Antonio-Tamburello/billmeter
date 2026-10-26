"""Test double: a fake model used ONLY by the automated tests (never by the CLI).

It answers by reading the expected values of the dataset, so the whole pipeline
(extraction, normalization, eval, compare) can be tested offline, for free and
deterministically.

- FakeProvider("perfect") -> returns exactly the expected values
- FakeProvider("noisy")   -> deterministically corrupts ~15% of the fields (wrong value,
                             missing value or invented value), so tests can check that
                             the eval detects every kind of error
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from billmeter.documents import Part
from billmeter.providers import Provider, ProviderConfigError, ProviderResponse
from billmeter.schema import EXTRACTED_FIELDS

_FAKE_VALUES = {
    "pod": "IT001E99999999",
    "pdr": "99999999999999",
    "holder_tax_code": "RSSMRA85T10A562S",
    "supply_address": "Via Inventata 1, 00100 Roma RM",
    "annual_consumption_kwh": 9999,
    "period_consumption_kwh": 999,
    "contracted_power_kw": 6.0,
    "gas_period_consumption_smc": 999,
    "gas_annual_consumption_smc": 9999,
    "gas_period_start": "2026-01-01",
    "gas_period_end": "2026-01-31",
    "is_resident": True,
}


class FakeProvider(Provider):
    name = "fake"
    pdf_native = True

    def generate(
        self, parts: list[Part], prompt: str, schema: dict[str, Any], source: Path | None = None
    ) -> ProviderResponse:
        if source is None:
            raise ProviderConfigError("The fake provider needs the source file path")
        expected = self._lookup_expected(source)
        answer = {field: expected.get(field) for field in EXTRACTED_FIELDS}
        if self.model == "noisy":
            answer = self._corrupt(answer, source.name)
        return ProviderResponse(json.dumps(answer), input_tokens=1500, output_tokens=200)

    @staticmethod
    def _lookup_expected(file: Path) -> dict[str, Any]:
        for folder in (file.parent, file.parent.parent):
            manifest = folder / "expected.json"
            if manifest.exists():
                cases = json.loads(manifest.read_text(encoding="utf-8"))["cases"]
                for case in cases:
                    if Path(case["file"]).name == file.name:
                        return case["expected"]
        raise ProviderConfigError(f"No expected.json entry for {file.name} (fake provider)")

    @staticmethod
    def _corrupt(answer: dict[str, Any], file_name: str) -> dict[str, Any]:
        corrupted = dict(answer)
        for field in EXTRACTED_FIELDS:
            roll = int(hashlib.md5(f"{file_name}:{field}".encode()).hexdigest(), 16) % 100
            if roll >= 15:
                continue
            value = corrupted[field]
            if value is None:
                corrupted[field] = _FAKE_VALUES.get(field)  # invent something
            elif roll < 8:
                corrupted[field] = None  # miss it
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                corrupted[field] = value * 2  # wrong number
            elif isinstance(value, str):
                corrupted[field] = _FAKE_VALUES.get(field, value[::-1])  # wrong text
        return corrupted
