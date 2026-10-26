"""Unit tests for the deterministic parts: tax code, normalization, graders."""

from datetime import date

from billmeter import fiscal_code
from billmeter.evals.graders import (
    CORRECT,
    CORRECT_NULL,
    HALLUCINATED,
    MISSING,
    WRONG,
    grade_field,
    hmac_value,
)
from billmeter.evals.pricing import PACKAGED_PRICING, cost_usd, pricing_file
from billmeter.extractor import parse_json
from billmeter.normalize import estimate_annual, normalize, parse_number
from billmeter.prompts import PACKAGED_PROMPTS, load_prompt, prompts_dir


def test_fiscal_code_known_example():
    assert fiscal_code.is_valid("RSSMRA85T10A562S")
    assert fiscal_code.generate("Mario", "Rossi", date(1985, 12, 10), False, "A562") == "RSSMRA85T10A562S"


def test_fiscal_code_rejects_wrong_check_char_and_vat_number():
    assert not fiscal_code.is_valid("RSSMRA85T10A562X")
    assert not fiscal_code.is_valid("01234567890")


def test_parse_number_italian_formats():
    assert parse_number("2.700") == 2700
    assert parse_number("3,5") == 3.5
    assert parse_number("1.234,5 kWh") == 1234.5
    assert parse_number("n/a") is None


def test_normalize_cleans_codes_and_derives_fields():
    bill = normalize({
        "pod": "it001e 12345678", "pdr": None, "holder_tax_code": "rssmra85t10a562s",
        "contracted_power_kw": "4,5", "period_start": "01/03/2026", "period_end": "30/04/2026",
        "period_consumption_kwh": "412", "bill_type": "gas",
    })
    assert bill.pod == "IT001E12345678"
    assert bill.bill_type == "power"  # the POD found wins over the declared type
    assert bill.holder_tax_code == "RSSMRA85T10A562S"
    assert bill.contracted_power_kw == 4.5
    assert bill.period_start == "2026-03-01"
    assert bill.estimated_annual_consumption_kwh == estimate_annual("2026-03-01", "2026-04-30", 412)


def test_normalize_drops_invalid_values():
    bill = normalize({"pod": "IT001X123", "holder_tax_code": "01234567890", "pdr": "123"})
    assert bill.pod is None and bill.pdr is None and bill.holder_tax_code is None
    assert bill.bill_type == "unknown"


def test_gas_bill_drops_electricity_fields():
    bill = normalize({"pdr": "12345678901234", "contracted_power_kw": 3, "period_consumption_kwh": 100})
    assert bill.bill_type == "gas"
    assert bill.contracted_power_kw is None and bill.period_consumption_kwh is None


def test_power_bill_drops_gas_fields():
    bill = normalize({"pod": "IT001E12345678", "gas_period_consumption_smc": 120, "gas_annual_consumption_smc": 900})
    assert bill.bill_type == "power"
    assert bill.gas_period_consumption_smc is None and bill.gas_annual_consumption_smc is None


def test_dual_bill_keeps_both_supplies_and_a_separate_gas_period():
    bill = normalize({
        "pod": "IT001E12345678", "pdr": "12345678901234", "contracted_power_kw": "3",
        "period_start": "2026-03-01", "period_end": "2026-04-30", "period_consumption_kwh": "412",
        "gas_period_start": "01/04/2026", "gas_period_end": "31/05/2026",
        "gas_period_consumption_smc": "1.120", "gas_annual_consumption_smc": "950 Smc",
    })
    assert bill.bill_type == "dual"
    assert bill.contracted_power_kw == 3.0 and bill.period_consumption_kwh == 412
    assert (bill.gas_period_start, bill.gas_period_end) == ("2026-04-01", "2026-05-31")
    assert bill.gas_period_consumption_smc == 1120 and bill.gas_annual_consumption_smc == 950
    # each yearly estimate uses its own period
    assert bill.estimated_annual_consumption_kwh == estimate_annual("2026-03-01", "2026-04-30", 412)
    assert bill.estimated_annual_gas_consumption_smc == estimate_annual("2026-04-01", "2026-05-31", 1120)


def test_gas_period_equal_to_the_bill_period_is_not_repeated():
    bill = normalize({
        "pdr": "12345678901234", "period_start": "2026-03-01", "period_end": "2026-04-30",
        "gas_period_start": "2026-03-01", "gas_period_end": "2026-04-30", "gas_period_consumption_smc": 100,
    })
    assert bill.gas_period_start is None and bill.gas_period_end is None
    assert bill.estimated_annual_gas_consumption_smc == estimate_annual("2026-03-01", "2026-04-30", 100)


def test_five_outcomes():
    assert grade_field("pod", "IT001E12345678", "IT001E12345678") == CORRECT
    assert grade_field("pod", None, None) == CORRECT_NULL
    assert grade_field("pod", "IT001E12345678", None) == MISSING
    assert grade_field("pod", "IT001E12345678", "IT001E00000000") == WRONG
    assert grade_field("pod", None, "IT001E00000000") == HALLUCINATED


def test_fuzzy_comparators():
    assert grade_field("holder_name", "Mario Rossi", "ROSSI MARIO") == CORRECT
    assert grade_field("supplier_name", "Luce Chiara S.p.A.", "LUCE CHIARA SPA") == CORRECT
    assert grade_field("supply_address", "Via G. Garibaldi 12, 50053 Empoli FI",
                       "V. Giuseppe Garibaldi 12, 50053 Empoli (FI)") == CORRECT
    assert grade_field("supply_address", "Via Roma 12, 50053 Empoli FI", "Via Roma 14, 50053 Empoli FI") == WRONG
    assert grade_field("estimated_annual_consumption_kwh", 2500, 2580) == CORRECT  # within 5%
    assert grade_field("is_resident", False, True) == WRONG


def test_hmac_expected_values(monkeypatch):
    monkeypatch.setenv("EVAL_HMAC_KEY", "test-key")
    hashed = hmac_value("holder_name", "Mario Rossi")
    assert grade_field("holder_name", hashed, "ROSSI  mario") == CORRECT
    assert grade_field("holder_name", hashed, "Luigi Rossi") == WRONG


def test_parse_json_tolerates_fences_and_text():
    assert parse_json('Here it is:\n```json\n{"pod": null}\n```') == {"pod": None}


def test_prompts_and_prices_come_from_the_folder_you_run_from(tmp_path, monkeypatch):
    """Run from the repo and its own files win; run from anywhere else and the wheel's copies do."""
    monkeypatch.chdir(tmp_path)
    assert prompts_dir() == PACKAGED_PROMPTS   # only exists in an installed wheel
    assert pricing_file() == PACKAGED_PRICING

    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "v9.md").write_text("mine: {{SCHEMA}}", encoding="utf-8")
    (tmp_path / "pricing.json").write_text('{"x:*": {"input_per_mtok": 2, "output_per_mtok": 4}}', encoding="utf-8")
    assert prompts_dir() == tmp_path / "prompts"
    assert load_prompt("v9").startswith("mine: {")   # the schema is substituted, not left as a marker
    assert cost_usd("x:any", 1_000_000, 1_000_000) == 6.0
