"""The static HTML report: it must be self-contained, honest and safe to open."""

import json
import re

import pytest
from fake_provider import FakeProvider
from typer.testing import CliRunner

from billmeter import cli
from billmeter.evals import html_report
from billmeter.evals.compare import compare
from billmeter.evals.runner import run_eval, save_results

synthetic = pytest.importorskip("billmeter.synthetic")


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    folder = tmp_path_factory.mktemp("dataset")
    synthetic.generate_dataset(folder, n=4, photos=1, seed=5)
    return run_eval(folder, FakeProvider("perfect")), run_eval(folder, FakeProvider("noisy"), runs=2)


def text_of(page: str) -> str:
    """The page as a reader sees it: tags removed, whitespace collapsed."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", page.split("<body>")[1]))


def test_the_report_shows_the_run_its_numbers_and_its_failures(runs):
    _, noisy = runs
    page = html_report.render(noisy)
    reader = text_of(page)

    assert page.startswith("<!doctype html>")
    for expected in (noisy["meta"]["model"], noisy["meta"]["prompt_sha256"], noisy["meta"]["dataset_sha256"],
                     noisy["meta"]["timestamp"], "field accuracy", "dangerous errors", "fully correct documents",
                     "stability", "latency p50", "tokens in / out", "Per field", "Documents with errors"):
        assert expected in reader, expected
    # a failing field is shown with the three values that say who got it wrong
    assert "expected" in reader and "raw" in reader and "normalized" in reader


def test_the_report_needs_nothing_from_the_network(runs):
    page = html_report.render(runs[1])

    assert "<script" not in page.lower()
    assert "http" not in page and "//" not in page  # no link, no CDN, no font, no comment-style URL
    assert "src=" not in page and "<link" not in page.lower()


def test_values_coming_from_the_model_are_escaped(runs):
    """A model can answer anything; the page must show it, never run it."""
    _, noisy = runs
    poisoned = json.loads(json.dumps(noisy))
    record = next(r for r in poisoned["cases"] if "wrong" in r["outcomes"].values())
    field = next(f for f, o in record["outcomes"].items() if o == "wrong")
    record["values"][field]["raw"] = "<script>alert('x')</script>"
    record["id"] = "<img onerror=alert(1)>"

    page = html_report.render(poisoned)

    assert "<script>alert" not in page and "<img onerror" not in page
    assert "&lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt;" in page
    assert "&lt;img onerror=alert(1)&gt;" in page


def test_two_runs_produce_a_comparison_with_deltas_and_regressions(runs):
    perfect, noisy = runs
    page = html_report.render(noisy, compare(perfect, noisy))
    reader = text_of(page)

    assert "Compared with the baseline" in reader
    assert "Regressions" in reader and "Improvements" in reader
    assert "pt" in reader  # at least one delta in percentage points
    assert perfect["meta"]["timestamp"] in reader


def test_a_report_of_real_data_quotes_no_value(runs, tmp_path):
    real = json.loads(json.dumps(runs[1]))
    real["meta"]["values_stored"] = False
    for record in real["cases"]:
        record.pop("values"), record.pop("raw_text"), record.pop("dropped")

    reader = text_of(html_report.render(real))

    assert "no extracted value is stored" in reader
    assert "expected" not in reader


def test_the_report_command_writes_the_file_where_the_data_allows(runs, tmp_path, monkeypatch):
    perfect, noisy = runs
    monkeypatch.chdir(tmp_path)
    candidate = save_results(noisy, tmp_path)
    baseline = save_results(perfect, tmp_path)
    runner = CliRunner()

    assert runner.invoke(cli.app, ["report", str(candidate)]).exit_code == 0
    written = list((tmp_path / "eval" / "reports").glob("*.html"))
    assert [p.name for p in written] == [f"{candidate.stem}.html"]

    assert runner.invoke(cli.app, ["report", str(candidate), str(baseline)]).exit_code == 0
    assert (tmp_path / "eval" / "reports" / f"{candidate.stem}_vs_{baseline.stem}.html").exists()

    real = json.loads(json.dumps(noisy))
    real["meta"]["values_stored"] = False
    real_file = tmp_path / "real.json"
    real_file.write_text(json.dumps(real), encoding="utf-8")
    assert runner.invoke(cli.app, ["report", str(real_file)]).exit_code == 0
    assert (tmp_path / "eval" / "reports" / "private" / "real.html").exists()
