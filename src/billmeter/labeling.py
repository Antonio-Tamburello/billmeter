"""Turn real bills into an evaluation dataset, with a human in the loop.

Measuring a model on a real bill needs the right answer for that bill. Writing
it by hand for every field is slow, so the work is split in two steps:

1. `billmeter label`  a model reads every new bill and saves a DRAFT of the
                      answers in `<dataset>/drafts/` (nothing is graded yet).
2. `billmeter review` you check each draft field by field: accept, correct,
                      mark as "not on the bill", or skip. Only what you
                      confirmed becomes the expected answer in expected.json.

The model only proposes; a person decides. That keeps the dataset honest: if
the drafts became the truth without review, the eval would end up measuring
the model against itself.

Reviewing is the slow part, so every draft also carries a list of "doubts":
the fields that deserve a careful look, each with the reason why (see
`find_doubts`). `review` shows the most doubtful drafts first and marks those
fields while it asks for them, so attention goes where mistakes are likely
instead of being spread evenly over every field of every bill.

Sensitive values (codes and names) are stored as HMACs, so expected.json never
contains them in clear text (see evals/graders.py).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .documents import is_supported
from .evals.graders import COMPARATORS, exact, hmac_value, is_empty
from .normalize import normalize, parse_date
from .schema import EXTRACTED_FIELDS

BILLS_DIR = "bills"
DRAFTS_DIR = "drafts"
MANIFEST = "expected.json"

# Stored as HMAC: compared exactly, never written in clear text.
SENSITIVE_FIELDS = ("pod", "pdr", "holder_tax_code", "holder_name")
# Compared with a fuzzy match, so it cannot be hashed: kept only if you ask for it.
ADDRESS_FIELD = "supply_address"
BILL_TYPES = ("power", "gas", "dual", "unknown")
# Stored only when gas has its own period; a gas period equal to the bill period becomes empty.
GAS_PERIOD_FIELDS = ("gas_period_start", "gas_period_end")

# Fields printed on every bill, whatever it supplies: empty means the model missed something.
ALWAYS_EXPECTED = ("holder_name", "supply_address", "period_start", "period_end")
# Fields printed only when that supply is on the bill.
POWER_EXPECTED = ("pod", "period_consumption_kwh")
GAS_EXPECTED = ("pdr", "gas_period_consumption_smc")


class Skip:
    """Marker for a field you chose not to grade."""

    def __repr__(self) -> str:
        return "SKIP"


SKIP = Skip()


class InvalidValue(ValueError):
    def __init__(self, field: str, value: Any):
        super().__init__(f"'{value}' is not a valid value for {field}")
        self.field = field


# ------------------------------------------------------------------ files

def load_manifest(dataset_dir: Path) -> dict[str, Any]:
    path = dataset_dir / MANIFEST
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"name": dataset_dir.name, "contains_real_data": True, "cases": []}


def save_manifest(dataset_dir: Path, manifest: dict[str, Any]) -> Path:
    path = dataset_dir / MANIFEST
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def draft_path(dataset_dir: Path, file: Path) -> Path:
    return dataset_dir / DRAFTS_DIR / f"{file.name}.json"


def unlabeled_files(dataset_dir: Path) -> list[Path]:
    """Bills in <dataset>/bills that have neither an expected answer nor a draft yet."""
    bills = dataset_dir / BILLS_DIR
    if not bills.is_dir():
        return []
    done = {case["file"] for case in load_manifest(dataset_dir)["cases"]}
    files = sorted(p for p in bills.rglob("*") if p.is_file() and is_supported(p))
    return [
        p for p in files
        if p.relative_to(dataset_dir).as_posix() not in done and not draft_path(dataset_dir, p).exists()
    ]


def save_draft(
    dataset_dir: Path,
    file: Path,
    model: str,
    proposed: dict[str, Any],
    error: Optional[str],
    dropped: Optional[dict[str, Any]] = None,
    second: Optional[dict[str, Any]] = None,
    second_model: Optional[str] = None,
) -> Path:
    """Write one draft, with everything the review needs to know what to distrust.

    `dropped` holds the raw values validation threw away and `second` the answers of
    a second model, when one was used; both only feed the doubts, they never become
    expected values.
    """
    path = draft_path(dataset_dir, file)
    path.parent.mkdir(parents=True, exist_ok=True)
    proposed = {name: proposed.get(name) for name in EXTRACTED_FIELDS}
    draft = {
        "file": file.relative_to(dataset_dir).as_posix(),
        "model": model,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "error": error,
        "proposed": proposed,
        "dropped": dict(dropped or {}),
        "doubts": find_doubts(proposed, error=error, dropped=dropped, second=second, second_model=second_model),
    }
    if second is not None:
        draft["second_model"] = second_model
        draft["second_proposed"] = {name: second.get(name) for name in EXTRACTED_FIELDS}
    path.write_text(json.dumps(draft, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def pending_drafts(dataset_dir: Path) -> list[Path]:
    folder = dataset_dir / DRAFTS_DIR
    return sorted(folder.glob("*.json")) if folder.is_dir() else []


def drafts_by_doubt(dataset_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Every pending draft with its content, the most doubtful one first.

    Reviewing is attention-limited: the bills where the model probably went wrong
    are worth your fresh attention, not the ones left over at the end. Drafts with
    the same number of doubtful fields keep file-name order, so the list is stable.
    """
    drafts = []
    for path in pending_drafts(dataset_dir):
        draft = json.loads(path.read_text(encoding="utf-8"))
        drafts.append((path, draft))
    return sorted(drafts, key=lambda item: (-len(item[1].get("doubts") or {}), item[0].name))


# ------------------------------------------------------------------ doubts

def agree(field: str, first: Any, second: Any) -> bool:
    """Would the eval call these two answers the same value?

    Uses the graders' comparators, so "two models disagree" means exactly what
    "wrong" means when a model is graded: 'ROSSI MARIO' and 'Mario Rossi' agree,
    3.0 and 3.5 kW do not.
    """
    if is_empty(first) and is_empty(second):
        return True
    if is_empty(first) or is_empty(second):
        return False
    try:
        return COMPARATORS.get(field, exact)(first, second)
    except (TypeError, ValueError):
        return False


def find_doubts(
    proposed: dict[str, Any],
    error: Optional[str] = None,
    dropped: Optional[dict[str, Any]] = None,
    second: Optional[dict[str, Any]] = None,
    second_model: Optional[str] = None,
) -> dict[str, list[str]]:
    """Which proposed fields deserve a careful look, and why.

    Deterministic rules only: no model decides what to distrust, so the same draft
    always produces the same doubts and the review order never moves under you.
    """
    doubts: dict[str, list[str]] = {}

    def add(field: str, reason: str) -> None:
        doubts.setdefault(field, []).append(reason)

    if error:  # nothing was proposed: every field has to be typed by hand
        for name in EXTRACTED_FIELDS:
            add(name, f"the extraction failed ({error}): nothing was proposed")
        return doubts

    for name, raw in (dropped or {}).items():
        add(name, f"the model answered {raw!r}, discarded by validation")

    bill_type = proposed.get("bill_type")
    if bill_type in (None, "unknown"):
        add("bill_type", "the kind of bill is unknown: no valid POD or PDR was found")

    expected = list(ALWAYS_EXPECTED)
    if bill_type in ("power", "dual"):
        expected += POWER_EXPECTED
    if bill_type in ("gas", "dual"):
        expected += GAS_EXPECTED
    for name in expected:
        if is_empty(proposed.get(name)) and name not in doubts:
            add(name, f"expected on a {bill_type} bill, but empty")

    if second is not None:
        label = second_model or "the second model"
        for name in EXTRACTED_FIELDS:
            if not agree(name, proposed.get(name), second.get(name)):
                add(name, f"{label} answered {second.get(name)!r}")

    return {name: doubts[name] for name in EXTRACTED_FIELDS if name in doubts}


# ------------------------------------------------------------------ review logic

def parse_answer(answer: str, proposed: Any) -> Any:
    """What you typed for one field during the review.

    Enter     keep the proposed value (a proposed null stays null)
    -         the value is NOT on the bill: expected null
    ?         don't grade this field for this bill
    anything  your correction
    """
    answer = answer.strip()
    if answer == "":
        return proposed
    if answer == "-":
        return None
    if answer == "?":
        return SKIP
    return answer


def build_expected(
    reviewed: dict[str, Any],
    hash_sensitive: bool = True,
    keep_address: bool = False,
    hmac_key: Optional[str] = None,
) -> dict[str, Any]:
    """Validate the reviewed values and turn them into the "expected" block of a case.

    Values go through the same normalization as model answers, so "3,0" and 3.0, or
    "01/03/2026" and "2026-03-01", are stored the same way the graders will see them.
    Raises InvalidValue if something you typed cannot be valid (e.g. a tax code with
    the wrong check character): better to ask again than to store a wrong truth.
    """
    graded = {name: value for name, value in reviewed.items() if value is not SKIP}
    bill_type = graded.get("bill_type")
    if bill_type is not None and bill_type not in BILL_TYPES:
        raise InvalidValue("bill_type", bill_type)

    normalized = normalize(graded).model_dump()
    expected: dict[str, Any] = {}
    for name, value in graded.items():
        if name == ADDRESS_FIELD and not keep_address:
            continue
        clean = bill_type if name == "bill_type" else normalized.get(name)
        same_as_bill_period = name in GAS_PERIOD_FIELDS and parse_date(value) is not None
        if value not in (None, "") and clean is None and not same_as_bill_period:
            raise InvalidValue(name, value)
        expected[name] = clean

    if hash_sensitive:
        for name in SENSITIVE_FIELDS:
            if expected.get(name) is not None:
                expected[name] = hmac_value(name, expected[name], hmac_key)
    return expected


def add_case(dataset_dir: Path, file: str, expected: dict[str, Any], tags: list[str]) -> str:
    """Append a reviewed bill to expected.json and return its id."""
    manifest = load_manifest(dataset_dir)
    taken = {case["id"] for case in manifest["cases"]}
    case_id = Path(file).stem
    suffix = 2
    while case_id in taken:
        case_id = f"{Path(file).stem}-{suffix}"
        suffix += 1
    manifest["contains_real_data"] = True
    manifest["cases"].append({"id": case_id, "file": file, "tags": tags, "expected": expected})
    save_manifest(dataset_dir, manifest)
    return case_id


def default_tags(file: str) -> list[str]:
    suffix = Path(file).suffix.lower().lstrip(".")
    return ["pdf"] if suffix == "pdf" else ["image", suffix]
