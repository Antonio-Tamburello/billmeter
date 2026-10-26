"""Turn the raw JSON returned by a model into a validated NormalizedBill.

Models return slightly different shapes ("3,0" instead of 3.0, spaces inside
codes, dates as 01/03/2026...). Everything is cleaned here, so the graders and
the callers always see one consistent format. Values that fail validation
become None: for a form that is auto-filled, an empty field is better than a
wrong one.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Optional, cast

from . import fiscal_code
from .schema import BillData, BillType, NormalizedBill

POD_RE = re.compile(r"^IT\d{3}E\d{8,9}$")
PDR_RE = re.compile(r"^\d{14}$")
_NULL_STRINGS = {"", "null", "none", "n/a", "na", "-", "--", "not found", "non presente"}


def clean_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return None if text.lower() in _NULL_STRINGS else text


def parse_number(value: Any) -> Optional[float]:
    """Parse numbers written the Italian way too: '2.700' -> 2700, '3,5' -> 3.5."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = clean_str(value)
    if text is None:
        return None
    text = re.sub(r"[^\d.,-]", "", text)
    if "," in text and "." in text:
        text = text.replace(".", "").replace(",", ".")  # 1.234,5
    elif "," in text:
        text = text.replace(",", ".")  # 3,5
    elif re.fullmatch(r"\d{1,3}(\.\d{3})+", text):
        text = text.replace(".", "")  # 2.700
    try:
        return float(text)
    except ValueError:
        return None


def parse_int(value: Any) -> Optional[int]:
    number = parse_number(value)
    # 0 is a real reading (a bill with the meter untouched for the whole period), so only
    # negative values are dropped.
    return None if number is None or number < 0 else int(round(number))


def parse_date(value: Any) -> Optional[str]:
    text = clean_str(value)
    if text is None:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d.%m.%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def parse_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    text = (clean_str(value) or "").lower()
    if text in ("true", "residente", "si", "sì", "yes"):
        return True
    if text in ("false", "non residente", "no"):
        return False
    return None


def normalize_pod(value: Any) -> Optional[str]:
    text = (clean_str(value) or "").replace(" ", "").upper()
    return text if POD_RE.match(text) else None


def normalize_pdr(value: Any) -> Optional[str]:
    text = re.sub(r"\s", "", clean_str(value) or "")
    return text if PDR_RE.match(text) else None


def normalize_tax_code(value: Any) -> Optional[str]:
    text = (clean_str(value) or "").replace(" ", "").upper()
    return text if fiscal_code.is_valid(text) else None


def compute_bill_type(pod: Optional[str], pdr: Optional[str], declared: Optional[str]) -> BillType:
    """The codes found are stronger evidence than the type the model declared."""
    if pod and pdr:
        return "dual"
    if pod:
        return "power"
    if pdr:
        return "gas"
    if declared in ("power", "gas", "dual"):
        return cast(BillType, declared)
    return "unknown"


def estimate_annual(start: Optional[str], end: Optional[str], amount: Optional[int]) -> Optional[int]:
    """Scale the consumption of a billing period to 365 days (inclusive day count).

    Works for electricity (kWh) and gas (Smc) alike.
    """
    if start is None or end is None or amount is None:
        return None
    days = (date.fromisoformat(end) - date.fromisoformat(start)).days + 1
    if not 20 <= days <= 400:  # implausible period: better no estimate than a wrong one
        return None
    return round(amount * 365 / days)


def _period(start_value: Any, end_value: Any) -> tuple[Optional[str], Optional[str]]:
    """Parse a period; an end before the start means the model mixed something up: drop both."""
    start, end = parse_date(start_value), parse_date(end_value)
    if start and end and end < start:
        return None, None
    return start, end


def normalize(raw: dict[str, Any]) -> NormalizedBill:
    pod = normalize_pod(raw.get("pod"))
    pdr = normalize_pdr(raw.get("pdr"))
    bill_type = compute_bill_type(pod, pdr, clean_str(raw.get("bill_type")))
    has_power = bill_type in ("power", "dual")

    power_kw = parse_number(raw.get("contracted_power_kw"))
    if power_kw is not None and not 0 < power_kw <= 1000:
        power_kw = None

    has_gas = bill_type in ("gas", "dual")

    period_start, period_end = _period(raw.get("period_start"), raw.get("period_end"))
    period_kwh = parse_int(raw.get("period_consumption_kwh")) if has_power else None

    # Gas fields exist only on gas and dual bills. The gas period is filled only when the bill
    # prints a different one for gas; otherwise the gas consumption refers to the bill period.
    gas_start, gas_end = _period(raw.get("gas_period_start"), raw.get("gas_period_end")) if has_gas else (None, None)
    if (gas_start, gas_end) == (period_start, period_end):
        gas_start = gas_end = None  # same as the bill period: nothing different to report
    gas_smc = parse_int(raw.get("gas_period_consumption_smc")) if has_gas else None
    gas_annual_smc = parse_int(raw.get("gas_annual_consumption_smc")) if has_gas else None

    data = BillData(
        supplier_name=clean_str(raw.get("supplier_name")),
        bill_type=bill_type,
        pod=pod,
        pdr=pdr,
        holder_name=clean_str(raw.get("holder_name")),
        holder_tax_code=normalize_tax_code(raw.get("holder_tax_code")),
        supply_address=clean_str(raw.get("supply_address")),
        is_resident=parse_bool(raw.get("is_resident")),
        contracted_power_kw=power_kw if has_power else None,
        annual_consumption_kwh=parse_int(raw.get("annual_consumption_kwh")) if has_power else None,
        period_start=period_start,
        period_end=period_end,
        period_consumption_kwh=period_kwh,
        gas_period_start=gas_start,
        gas_period_end=gas_end,
        gas_period_consumption_smc=gas_smc,
        gas_annual_consumption_smc=gas_annual_smc,
    )
    return NormalizedBill(
        **data.model_dump(),
        estimated_annual_consumption_kwh=estimate_annual(period_start, period_end, period_kwh),
        estimated_annual_gas_consumption_smc=estimate_annual(gas_start or period_start, gas_end or period_end, gas_smc),
    )
