"""Labeling real bills: drafts proposed by a model, confirmed by a person."""

import json

import pytest
from PIL import Image
from typer.testing import CliRunner

from billmeter import cli
from billmeter.evals.graders import hmac_value
from billmeter.labeling import (
    SKIP,
    InvalidValue,
    add_case,
    agree,
    build_expected,
    drafts_by_doubt,
    find_doubts,
    load_manifest,
    parse_answer,
    pending_drafts,
    save_draft,
    unlabeled_files,
)
from billmeter.providers import Provider, ProviderResponse

KEY = "test-key"
TAX_CODE = "RSSMRA85T10A562S"  # fake but valid (check character included)
PROPOSED = {
    "supplier_name": "Luce Chiara S.p.A.", "bill_type": "power", "pod": "IT001E12345678", "pdr": None,
    "holder_name": "Mario Rossi", "holder_tax_code": TAX_CODE, "supply_address": "Via Roma 1, 50053 Empoli FI",
    "is_resident": True, "contracted_power_kw": 3.0, "annual_consumption_kwh": None,
    "period_start": "2026-01-01", "period_end": "2026-02-28", "period_consumption_kwh": 400,
    "gas_period_start": None, "gas_period_end": None, "gas_period_consumption_smc": None,
    "gas_annual_consumption_smc": None,
}


def test_parse_answer():
    assert parse_answer("", "IT001E12345678") == "IT001E12345678"  # Enter keeps the proposal
    assert parse_answer("  -  ", "x") is None                       # not on the bill
    assert parse_answer("?", "x") is SKIP                           # not graded
    assert parse_answer("3,5", 3.0) == "3,5"                        # correction


def test_build_expected_normalizes_and_hashes():
    reviewed = {**PROPOSED, "contracted_power_kw": "4,5", "period_start": "01/01/2026"}
    expected = build_expected(reviewed, hmac_key=KEY)

    assert expected["contracted_power_kw"] == 4.5
    assert expected["period_start"] == "2026-01-01"
    assert expected["holder_tax_code"] == hmac_value("holder_tax_code", TAX_CODE, KEY)
    assert expected["pod"].startswith("hmac:") and expected["pdr"] is None
    assert "supply_address" not in expected  # fuzzy field: stored only with keep_address
    # The yearly estimates are computed by normalize.py from fields already graded here,
    # so they are deliberately not part of the expected answer.
    assert "estimated_annual_consumption_kwh" not in expected


def test_build_expected_skips_and_keeps_address_in_clear_when_asked():
    reviewed = {**PROPOSED, "is_resident": SKIP}
    expected = build_expected(reviewed, hash_sensitive=False, keep_address=True)

    assert "is_resident" not in expected
    assert expected["supply_address"] == PROPOSED["supply_address"]
    assert expected["holder_tax_code"] == TAX_CODE


def test_build_expected_refuses_impossible_values():
    with pytest.raises(InvalidValue) as error:
        build_expected({**PROPOSED, "holder_tax_code": "RSSMRA85T10A562X"}, hash_sensitive=False)
    assert error.value.field == "holder_tax_code"
    with pytest.raises(InvalidValue):
        build_expected({**PROPOSED, "bill_type": "water"}, hash_sensitive=False)


def test_drafts_and_cases_track_which_bills_are_done(tmp_path):
    bills = tmp_path / "bills"
    bills.mkdir()
    for name in ("a.pdf", "b.jpg", "notes.txt"):
        (bills / name).write_bytes(b"x")

    assert [p.name for p in unlabeled_files(tmp_path)] == ["a.pdf", "b.jpg"]  # .txt is not a bill
    save_draft(tmp_path, bills / "a.pdf", "test:model", PROPOSED, None)
    add_case(tmp_path, "bills/b.jpg", {"bill_type": "power"}, ["image"])

    assert unlabeled_files(tmp_path) == []
    assert len(pending_drafts(tmp_path)) == 1
    manifest = load_manifest(tmp_path)
    assert manifest["contains_real_data"] is True
    assert manifest["cases"][0]["id"] == "b"


class FixedAnswerProvider(Provider):
    name = "fixed"
    pdf_native = True

    def generate(self, parts, prompt, schema, source=None):
        return ProviderResponse(json.dumps(PROPOSED), 100, 50)


def test_label_then_review_from_the_command_line(tmp_path, monkeypatch):
    (tmp_path / "bills").mkdir()
    (tmp_path / "bills" / "bill.pdf").write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr(cli, "get_provider", lambda spec, settings=None: FixedAnswerProvider("model"))
    monkeypatch.setenv("EVAL_HMAC_KEY", KEY)
    runner = CliRunner()

    labeled = runner.invoke(cli.app, ["label", "-m", "fixed:model", "-d", str(tmp_path)])
    assert labeled.exit_code == 0, labeled.output
    assert len(pending_drafts(tmp_path)) == 1

    # 17 fields: correct the power, mark residency as "not on the bill", keep the rest; then save with default tags.
    answers = ["", "", "", "", "", "", "", "-", "4,5", "", "", "", "", "", "", "", "", "y", ""]
    reviewed = runner.invoke(cli.app, ["review", "-d", str(tmp_path), "--no-open"], input="\n".join(answers) + "\n")
    assert reviewed.exit_code == 0, reviewed.output

    case = load_manifest(tmp_path)["cases"][0]
    assert case["file"] == "bills/bill.pdf" and case["tags"] == ["pdf"]
    assert case["expected"]["contracted_power_kw"] == 4.5
    assert case["expected"]["is_resident"] is None
    assert case["expected"]["holder_name"].startswith("hmac:")
    assert pending_drafts(tmp_path) == []


def test_dual_bill_review_keeps_gas_fields_and_accepts_a_gas_period_equal_to_the_bill_one():
    reviewed = {**PROPOSED, "bill_type": "dual", "pdr": "12345678901234",
                "gas_period_start": "2026-01-01", "gas_period_end": "28/02/2026",  # same as the bill period
                "gas_period_consumption_smc": "180", "gas_annual_consumption_smc": "1.050"}
    expected = build_expected(reviewed, hash_sensitive=False)

    assert expected["bill_type"] == "dual" and expected["pdr"] == "12345678901234"
    assert expected["gas_period_start"] is None and expected["gas_period_end"] is None
    assert expected["gas_period_consumption_smc"] == 180 and expected["gas_annual_consumption_smc"] == 1050


# ------------------------------------------------------------------ doubts

def test_a_clean_power_draft_has_no_doubts():
    assert find_doubts(PROPOSED) == {}


def test_doubts_flag_values_dropped_by_validation():
    proposed = {**PROPOSED, "pod": None}
    doubts = find_doubts(proposed, dropped={"pod": "IT001E1234"})  # too short: normalize dropped it

    assert "discarded by validation" in doubts["pod"][0] and "IT001E1234" in doubts["pod"][0]
    assert len(doubts["pod"]) == 1  # the same field is not also reported as simply empty


def test_doubts_flag_an_unknown_type_and_the_fields_that_type_should_carry():
    power = find_doubts({**PROPOSED, "holder_name": None, "period_consumption_kwh": None})
    assert set(power) == {"holder_name", "period_consumption_kwh"}

    gas = find_doubts({**PROPOSED, "bill_type": "gas", "pdr": "12345678901234",
                       "gas_period_consumption_smc": None})
    assert set(gas) == {"gas_period_consumption_smc"}  # a gas bill is not expected to carry a POD

    unknown = find_doubts({**PROPOSED, "bill_type": "unknown", "pod": None})
    assert "unknown" in unknown["bill_type"][0]
    assert "pod" not in unknown  # without a type we cannot say which supply codes to expect


def test_a_failed_extraction_makes_every_field_doubtful():
    doubts = find_doubts(dict.fromkeys(PROPOSED), error="TimeoutError: no answer")

    assert len(doubts) == len(PROPOSED)
    assert all("the extraction failed" in reasons[0] for reasons in doubts.values())


def test_a_second_model_disagrees_only_where_the_graders_would():
    second = {**PROPOSED, "holder_name": "ROSSI MARIO",          # same words: agrees
              "supply_address": "VIA ROMA 1, 50053 EMPOLI (FI)",  # fuzzy match: agrees
              "contracted_power_kw": 4.5,                         # different number: disagrees
              "pdr": "12345678901234"}                            # one is empty: disagrees
    doubts = find_doubts(PROPOSED, second=second, second_model="fake:other")

    assert set(doubts) == {"pdr", "contracted_power_kw"}
    assert "fake:other answered 4.5" in doubts["contracted_power_kw"][0]
    assert agree("holder_name", "Mario Rossi", "ROSSI MARIO") and not agree("pod", "IT001E12345678", None)


def test_drafts_are_reviewed_most_doubtful_first(tmp_path):
    bills = tmp_path / "bills"
    bills.mkdir()
    for name in ("clean.pdf", "some.pdf", "broken.pdf"):
        (bills / name).write_bytes(b"x")
    save_draft(tmp_path, bills / "clean.pdf", "m", PROPOSED, None)
    save_draft(tmp_path, bills / "some.pdf", "m", {**PROPOSED, "holder_name": None}, None)
    save_draft(tmp_path, bills / "broken.pdf", "m", dict.fromkeys(PROPOSED), "TimeoutError: no answer")

    assert [draft["file"] for _, draft in drafts_by_doubt(tmp_path)] == [
        "bills/broken.pdf", "bills/some.pdf", "bills/clean.pdf"]


def test_review_still_works_on_a_draft_written_before_doubts_existed(tmp_path, monkeypatch):
    (tmp_path / "bills").mkdir()
    (tmp_path / "bills" / "old.pdf").write_bytes(b"%PDF-1.4 fake")
    (tmp_path / "drafts").mkdir()
    (tmp_path / "drafts" / "old.pdf.json").write_text(json.dumps(
        {"file": "bills/old.pdf", "model": "old:model", "created": "2026-01-01T00:00:00+00:00",
         "error": None, "proposed": PROPOSED}), encoding="utf-8")
    monkeypatch.setenv("EVAL_HMAC_KEY", KEY)

    answers = [""] * len(PROPOSED) + ["y", ""]
    done = CliRunner().invoke(cli.app, ["review", "-d", str(tmp_path), "--no-open"], input="\n".join(answers) + "\n")

    assert done.exit_code == 0, done.output
    assert load_manifest(tmp_path)["cases"][0]["file"] == "bills/old.pdf"


def test_label_with_a_second_model_turns_disagreements_into_doubts(tmp_path, monkeypatch):
    (tmp_path / "bills").mkdir()
    (tmp_path / "bills" / "bill.pdf").write_bytes(b"%PDF-1.4 fake")

    class OtherAnswerProvider(FixedAnswerProvider):
        name = "other"

        def generate(self, parts, prompt, schema, source=None):
            return ProviderResponse(json.dumps({**PROPOSED, "contracted_power_kw": 6.0}), 100, 50)

    providers = {"fixed:model": FixedAnswerProvider("model"), "other:model": OtherAnswerProvider("model")}
    monkeypatch.setattr(cli, "get_provider", lambda spec, settings=None: providers[spec])

    labeled = CliRunner().invoke(cli.app, ["label", "-m", "fixed:model", "-d", str(tmp_path),
                                           "--second-model", "other:model"])
    assert labeled.exit_code == 0, labeled.output

    draft = json.loads(pending_drafts(tmp_path)[0].read_text(encoding="utf-8"))
    assert draft["second_model"] == "other:model"
    assert draft["second_proposed"]["contracted_power_kw"] == 6.0
    assert set(draft["doubts"]) == {"contracted_power_kw"}
    assert "other:model answered 6.0" in draft["doubts"]["contracted_power_kw"][0]


def test_label_skips_a_file_it_cannot_read_and_keeps_going(tmp_path, monkeypatch):
    """One corrupt bill in the folder used to kill the whole run, losing every call paid for."""
    bills = tmp_path / "bills"
    bills.mkdir()
    (bills / "a-broken.pdf").write_bytes(b"%PDF-1.4 truncated, pdfium cannot open this")
    Image.new("RGB", (40, 20), "white").save(bills / "b-fine.png")

    provider = FixedAnswerProvider("model")
    provider.pdf_native = False  # like Ollama: pages are rendered locally, so the bad file fails here
    monkeypatch.setattr(cli, "get_provider", lambda spec, settings=None: provider)

    labeled = CliRunner().invoke(cli.app, ["label", "-m", "fixed:model", "-d", str(tmp_path)])

    assert labeled.exit_code == 0, labeled.output
    assert "a-broken.pdf" in labeled.output and "unreadable" in labeled.output
    # the readable bill was still labeled; the broken one has no draft and stays in the to-do list
    assert [p.name for p in pending_drafts(tmp_path)] == ["b-fine.png.json"]
    assert [p.name for p in unlabeled_files(tmp_path)] == ["a-broken.pdf"]
