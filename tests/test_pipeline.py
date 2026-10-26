"""End-to-end: generate a small synthetic dataset, run the eval with the fake provider, compare."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fake_provider import FakeProvider

from billmeter.evals.compare import compare
from billmeter.evals.runner import run_eval, save_results
from billmeter.extractor import extract

synthetic = pytest.importorskip("billmeter.synthetic")


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    folder = tmp_path_factory.mktemp("dataset")
    synthetic.generate_dataset(folder, n=8, photos=2, seed=7)
    return folder


def test_generated_expected_values_are_valid(dataset):
    manifest = json.loads((dataset / "expected.json").read_text())
    assert len(manifest["cases"]) == 10
    assert manifest["contains_real_data"] is False
    for case in manifest["cases"]:
        assert (dataset / case["file"]).exists()
        expected = case["expected"]
        has_gas = expected["bill_type"] in ("gas", "dual")
        assert (expected["gas_period_consumption_smc"] is not None) == has_gas
        assert (expected["period_consumption_kwh"] is not None) == (expected["bill_type"] != "gas")


def test_perfect_answers_score_100(dataset):
    results = run_eval(dataset, FakeProvider("perfect"), concurrency=2)
    assert results["summary"]["field_accuracy"] == 1.0
    assert results["summary"]["dangerous_rate"] == 0.0


def test_noisy_answers_regressions_are_detected(dataset, tmp_path):
    perfect = run_eval(dataset, FakeProvider("perfect"))
    noisy = run_eval(dataset, FakeProvider("noisy"), runs=2)
    assert noisy["summary"]["field_accuracy"] < 1.0
    assert noisy["summary"]["stability"] == 1.0  # the fake provider is deterministic
    report = compare(perfect, noisy)
    assert report["regressions"] and not report["improvements"]
    assert save_results(noisy, tmp_path).exists()


def test_pdf_is_rasterized_for_image_only_providers(dataset):
    provider = FakeProvider("perfect")
    provider.pdf_native = False  # behave like Ollama: pages are sent as PNG images
    result = extract(dataset / "bills" / "syn-001.pdf", provider)
    assert result.ok and result.bill.pod


def _generate_in_another_process(out_dir: Path, hash_seed: str) -> None:
    """Generate the dataset in a fresh interpreter with a chosen PYTHONHASHSEED.

    A separate process is the only way to test this: PYTHONHASHSEED is read once
    at interpreter start-up, so it cannot be changed from inside a test.
    """
    env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": str(Path(synthetic.__file__).parents[1])}
    code = ("from billmeter.synthetic import generate_dataset;"
            f"generate_dataset({str(out_dir)!r}, n=4, photos=1, seed=3)")
    subprocess.run([sys.executable, "-c", code], env=env, check=True, capture_output=True)


def test_the_same_seed_gives_the_same_bytes_whatever_the_hash_seed(tmp_path):
    """Faker picks some values from unordered collections, so the generator must not rely on them.

    Without the fix the cities (and every PDF that prints them) change between
    processes, and `billmeter generate` would rewrite the dataset at every run.
    """
    first, second = tmp_path / "a", tmp_path / "b"
    _generate_in_another_process(first, "1")
    _generate_in_another_process(second, "2")

    files = sorted(p.relative_to(first) for p in first.rglob("*") if p.is_file())
    assert [p.as_posix() for p in files] == [
        "bills/syn-001-photo.jpg", "bills/syn-001.pdf", "bills/syn-002.pdf", "bills/syn-003.pdf",
        "bills/syn-004.pdf", "expected.json",
    ]
    assert files == sorted(p.relative_to(second) for p in second.rglob("*") if p.is_file())
    for name in files:
        assert (first / name).read_bytes() == (second / name).read_bytes(), f"{name} differs between hash seeds"


def test_a_run_on_synthetic_data_keeps_the_traces_needed_to_blame_model_or_normalization(dataset):
    results = run_eval(dataset, FakeProvider("noisy"))

    assert results["meta"]["values_stored"] is True
    record = next(r for r in results["cases"] if "wrong" in r["outcomes"].values())
    assert record["raw_text"].startswith("{")
    assert isinstance(record["dropped"], dict)
    assert all(set(v) == {"expected", "raw", "actual"} for v in record["values"].values())
    # "raw" is the model's own answer before normalize: that is what makes the blame visible
    wrong = next(f for f, o in record["outcomes"].items() if o == "wrong")
    assert record["values"][wrong]["raw"] != record["values"][wrong]["expected"]


def test_a_run_on_real_data_stores_no_values_and_no_raw_text(dataset, tmp_path):
    """A dataset marked as real must leave no trace of its content in the results file."""
    real = tmp_path / "real"
    real.mkdir()
    manifest = json.loads((dataset / "expected.json").read_text())
    manifest["contains_real_data"] = True
    (real / "expected.json").write_text(json.dumps(manifest))
    (real / "bills").symlink_to(dataset / "bills")

    results = run_eval(real, FakeProvider("noisy"))

    assert results["meta"]["values_stored"] is False
    assert all(set(r) & {"values", "raw_text", "dropped"} == set() for r in results["cases"])
    dumped = json.dumps(results)
    assert "IT0" not in dumped and "expected" not in dumped.replace('"n_cases"', "")


def test_one_corrupt_file_does_not_kill_a_whole_eval_run(dataset, tmp_path):
    """The run keeps going and records the failure: the calls already paid for are not lost."""
    folder = tmp_path / "with-a-bad-file"
    (folder / "bills").mkdir(parents=True)
    manifest = json.loads((dataset / "expected.json").read_text())
    for case in manifest["cases"]:
        (folder / case["file"]).write_bytes((dataset / case["file"]).read_bytes())
    good = manifest["cases"][0]
    manifest["cases"].append({"id": "broken", "file": "bills/broken.pdf", "tags": ["pdf"],
                              "expected": good["expected"]})
    (folder / "expected.json").write_text(json.dumps(manifest))
    (folder / "bills" / "broken.pdf").write_bytes(b"%PDF-1.4 not really a pdf")

    provider = FakeProvider("perfect")
    provider.pdf_native = False  # like Ollama: pages are rendered here, so a broken PDF fails here

    results = run_eval(folder, provider, concurrency=2)

    broken = next(r for r in results["cases"] if r["id"] == "broken")
    assert "UnreadableDocument" in broken["error"] and "broken.pdf" in broken["error"]
    # nothing was extracted, so every expected value is missing and none is invented
    assert set(broken["outcomes"].values()) <= {"missing", "correct_null"}
    assert "missing" in broken["outcomes"].values()
    assert results["summary"]["n_docs"] == len(manifest["cases"])  # every other document still ran
    assert results["summary"]["error_rate"] == round(1 / len(manifest["cases"]), 4)
