"""Code graders: compare one extracted field with its expected value.

Every field gets one of five outcomes. The split matters: for a form that is
auto-filled, a WRONG or HALLUCINATED value is far worse than a MISSING one,
because the user may not notice it.

    correct       expected X, got X
    correct_null  expected nothing, got nothing
    missing       expected X, got nothing
    wrong         expected X, got Y
    hallucinated  expected nothing, got something

Private datasets can store sensitive expected values as "hmac:<hex>" instead of
plain text (see `hmac_value`); they are compared without ever being revealed.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import unicodedata
from collections.abc import Callable
from difflib import SequenceMatcher
from typing import Any

CORRECT = "correct"
CORRECT_NULL = "correct_null"
MISSING = "missing"
WRONG = "wrong"
HALLUCINATED = "hallucinated"
OUTCOMES = [CORRECT, CORRECT_NULL, MISSING, WRONG, HALLUCINATED]
OK_OUTCOMES = {CORRECT, CORRECT_NULL}
DANGEROUS_OUTCOMES = {WRONG, HALLUCINATED}

HMAC_PREFIX = "hmac:"


# ---------------------------------------------------------------- text helpers

def fold(text: str) -> str:
    """Lowercase, strip accents and punctuation, collapse spaces."""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


_ABBREVIATIONS = {"v": "via", "p": "piazza", "pza": "piazza", "p za": "piazza", "c so": "corso",
                  "cso": "corso", "v le": "viale", "vle": "viale", "loc": "localita"}


def fold_address(text: str) -> str:
    folded = fold(text)
    for short in sorted(_ABBREVIATIONS, key=len, reverse=True):  # longest first: "p za" before "p"
        full = _ABBREVIATIONS[short]
        folded = re.sub(rf"^{short}\b", full, folded)
    return folded


def canonical(field: str, value: Any) -> str:
    """The normalized string an HMAC is computed on (same for expected and actual)."""
    if field == "holder_name":
        return " ".join(sorted(fold(value).split()))
    if field == "supply_address":
        return fold_address(value)
    return str(value).strip().upper().replace(" ", "")


def hmac_value(field: str, value: Any, key: str | None = None) -> str:
    key = key or os.environ.get("EVAL_HMAC_KEY")
    if not key:
        raise RuntimeError("Set EVAL_HMAC_KEY in .env to use hashed expected values")
    digest = hmac.new(key.encode(), canonical(field, value).encode(), hashlib.sha256).hexdigest()
    return HMAC_PREFIX + digest


# ---------------------------------------------------------------- comparators

def exact(expected: Any, actual: Any) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected is actual
    return str(expected).strip().upper() == str(actual).strip().upper()


def numeric(tolerance: float) -> Callable[[Any, Any], bool]:
    """Relative tolerance, e.g. 0.05 = within 5%."""
    def compare(expected: Any, actual: Any) -> bool:
        expected, actual = float(expected), float(actual)
        return abs(expected - actual) <= abs(expected) * tolerance + 1e-9
    return compare


def same_tokens(expected: Any, actual: Any) -> bool:
    """'ROSSI MARIO' == 'Mario Rossi'."""
    return sorted(fold(expected).split()) == sorted(fold(actual).split())


def similar_text(threshold: float = 0.85) -> Callable[[Any, Any], bool]:
    noise = r"\b(s p a|spa|s r l|srl|s a s|sas|societa per azioni)\b"
    def compare(expected: Any, actual: Any) -> bool:
        a = re.sub(noise, "", fold(expected)).strip()
        b = re.sub(noise, "", fold(actual)).strip()
        return SequenceMatcher(None, a, b).ratio() >= threshold
    return compare


def address_match(expected: Any, actual: Any) -> bool:
    """CAP and street number must match exactly; the rest only needs to be similar.

    'Via G. Garibaldi 12, 50053 Empoli FI' == 'VIA GIUSEPPE GARIBALDI 12, 50053 EMPOLI (FI)'
    """
    e, a = fold_address(expected), fold_address(actual)
    cap_e, cap_a = re.findall(r"\b\d{5}\b", e), re.findall(r"\b\d{5}\b", a)
    if cap_e != cap_a:
        return False
    num_e = re.findall(r"\b\d{1,4}[a-z]?\b", e.split(cap_e[0])[0] if cap_e else e)
    num_a = re.findall(r"\b\d{1,4}[a-z]?\b", a.split(cap_a[0])[0] if cap_a else a)
    if num_e[-1:] != num_a[-1:]:
        return False
    return SequenceMatcher(None, e, a).ratio() >= 0.75


COMPARATORS: dict[str, Callable[[Any, Any], bool]] = {
    "supplier_name": similar_text(),
    "bill_type": exact,
    "pod": exact,
    "pdr": exact,
    "holder_name": same_tokens,
    "holder_tax_code": exact,
    "supply_address": address_match,
    "is_resident": exact,
    "contracted_power_kw": numeric(0.001),
    "annual_consumption_kwh": numeric(0.01),
    "period_start": exact,
    "period_end": exact,
    "period_consumption_kwh": numeric(0.01),
    "gas_period_start": exact,
    "gas_period_end": exact,
    "gas_period_consumption_smc": numeric(0.01),
    "gas_annual_consumption_smc": numeric(0.01),
    "estimated_annual_consumption_kwh": numeric(0.05),
    "estimated_annual_gas_consumption_smc": numeric(0.05),
}


def is_empty(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def grade_field(field: str, expected: Any, actual: Any) -> str:
    if is_empty(expected):
        return CORRECT_NULL if is_empty(actual) else HALLUCINATED
    if is_empty(actual):
        return MISSING
    if isinstance(expected, str) and expected.startswith(HMAC_PREFIX):
        return CORRECT if hmac.compare_digest(expected, hmac_value(field, actual)) else WRONG
    try:
        return CORRECT if COMPARATORS.get(field, exact)(expected, actual) else WRONG
    except (TypeError, ValueError):
        return WRONG


def grade(expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, str]:
    """Grade every field present in `expected` (fields not listed there are skipped)."""
    return {name: grade_field(name, value, actual.get(name)) for name, value in expected.items()}
