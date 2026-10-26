"""The output contract: which fields we extract and their types.

This file says what *shape* the answer must have. It deliberately says nothing
about how to read a bill: every instruction ("the supply address, not the
mailing one", "the total, not the F1/F2/F3 bands") lives in prompts/, so that
comparing two prompts compares two prompts and nothing else.

The LLM extracts *facts printed on the bill*. Anything that needs arithmetic
(e.g. the estimated yearly consumption) is computed in code afterwards, because
models are much more reliable at reading than at calculating.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict

BillType = Literal["power", "gas", "dual", "unknown"]


class BillData(BaseModel):
    """Fields the model is asked to extract."""

    model_config = ConfigDict(extra="ignore")

    supplier_name: Optional[str] = None
    bill_type: Optional[BillType] = None
    pod: Optional[str] = None
    pdr: Optional[str] = None
    holder_name: Optional[str] = None
    holder_tax_code: Optional[str] = None
    supply_address: Optional[str] = None
    is_resident: Optional[bool] = None
    contracted_power_kw: Optional[float] = None
    annual_consumption_kwh: Optional[int] = None
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    period_consumption_kwh: Optional[int] = None
    gas_period_start: Optional[str] = None
    gas_period_end: Optional[str] = None
    gas_period_consumption_smc: Optional[int] = None
    gas_annual_consumption_smc: Optional[int] = None


class NormalizedBill(BillData):
    """Validated output returned to the caller: BillData plus fields computed in code."""

    estimated_annual_consumption_kwh: Optional[int] = None
    estimated_annual_gas_consumption_smc: Optional[int] = None


# Fields the model extracts (order = order in reports).
EXTRACTED_FIELDS: list[str] = list(BillData.model_fields.keys())
# Fields graded by the evaluation (extracted + derived).
GRADED_FIELDS: list[str] = list(NormalizedBill.model_fields.keys())


def _clean_schema(node: Any) -> Any:
    """Drop pydantic-only keys ('title', 'default') that some providers dislike."""
    if isinstance(node, dict):
        return {k: _clean_schema(v) for k, v in node.items() if k not in ("title", "default")}
    if isinstance(node, list):
        return [_clean_schema(v) for v in node]
    return node


def json_schema() -> dict[str, Any]:
    """JSON Schema of BillData, shared by every provider for structured output."""
    schema = _clean_schema(BillData.model_json_schema())
    schema["required"] = EXTRACTED_FIELDS  # ask for every field, using null when absent
    schema["additionalProperties"] = False
    return schema
