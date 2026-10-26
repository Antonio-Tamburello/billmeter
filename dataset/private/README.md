# Private dataset (not versioned)

Everything in this folder except this file is ignored by git. Use it for real bills you are allowed to use.

## From bills to an evaluation dataset

```bash
# 1. put the bills here (PDF, PNG, JPG, WEBP, HEIC, TIFF, BMP, GIF; subfolders are fine)
dataset/private/bills/

# 2. a model proposes the answers: one draft per bill in dataset/private/drafts/
uv run billmeter label -m ollama:<model>

# 3. you confirm or correct every field; confirmed bills are added to expected.json
uv run billmeter review

# 4. measure any model on them
uv run billmeter eval -m <provider:model> -d dataset/private
```

During the review, for each field: **Enter** keeps the proposal, **-** means "not on this bill"
(the right answer is empty), **?** means "don't grade this field", anything else is your correction.
Values are checked on the spot: a tax code with a wrong check character, for example, is refused.

Every draft carries the fields that deserve a careful look and the reason why — a value validation
discarded, a failed extraction, an undecided bill type, a field the bill type should carry but is
empty. `review` shows the most doubtful drafts first and marks those fields with a `⚠`. Add
`label --second-model <provider>:<model>` and every field the two models disagree on becomes a doubt
too, judged by the same comparators the graders use.

**Do not label with the model you intend to measure.** A ground truth confirmed without rereading
the document reproduces that model's errors inside the eval set.

## What is stored

```
dataset/private/
  bills/…            your files
  drafts/…           proposals waiting for review (deleted once reviewed)
  expected.json      the confirmed answers
```

```json
{
  "name": "private",
  "contains_real_data": true,
  "cases": [
    {
      "id": "001",
      "file": "bills/001.pdf",
      "tags": ["pdf"],
      "expected": {
        "pod": "hmac:…",
        "holder_tax_code": "hmac:…",
        "holder_name": "hmac:…",
        "bill_type": "power",
        "contracted_power_kw": 3.0
      }
    }
  ]
}
```

- `"contains_real_data": true` keeps extracted values out of the results files, and `eval` saves the
  results in `eval/results/private/`, which is git-ignored too.
- What those results **do** keep is the case id, which is the file name — without it you could not
  tell which bill failed. So `bolletta-enel-marco.pdf` names a supplier and a person in every run it
  appears in. Name the files in `bills/` as neutrally as you want those runs to be shareable;
  everything else in them is already anonymous.
- Codes and names (`pod`, `pdr`, `holder_tax_code`, `holder_name`) are stored as HMAC, so the expected
  values are never written in clear text. It needs `EVAL_HMAC_KEY` in `.env` (see `.env.example`).
  `review --plain` stores them in clear text instead.
- The supply address is compared with a fuzzy match, so it cannot be hashed: it is stored (in clear text)
  only with `review --keep-address`. Otherwise it is not graded.
- You can also write `expected.json` by hand; `billmeter hash <field> <value>` prints the HMAC of a value.
  Fields you leave out are not graded.
