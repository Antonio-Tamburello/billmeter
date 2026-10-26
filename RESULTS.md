# Results

Two prompts, two models, 30 documents, every field graded. Four things came out of it that no
amount of staring at the prompts would have produced.

Every number below comes from a saved run in [`eval/results/`](eval/results/), stored next to the
model, the prompt and its hash, the dataset hash, the git commit and the timestamp. The dataset is
`dataset/synthetic`: 30 documents (24 PDFs, 6 phone photos), 17 graded fields each — **510 grades
per run** — and `billmeter generate` rebuilds those files byte for byte from the same seed, so
anyone who clones this repo can get the same numbers back.

One warning about the headline metric before the tables. `field_accuracy` also rewards a correct
`null`, and 155 of those 510 fields are expected-null: a model that answered null to everything
would already score 30 %. Two more numbers split that floor apart, and they **fail in opposite
directions**:

| Metric | Asks |
|---|---|
| `filled_accuracy` | did we get it right where the bill **has** a value? |
| `empty_accuracy` | did we keep quiet where the bill has **nothing**? |

Watch those two swap places between section 1 and section 3. That swap is the most useful thing in
this document.

## 1. The prompt, measured instead of assumed

Two models, same 30 documents, same dataset hash, temperature 0. Only the prompt changes. The hosted
model ran every document **three times** (1530 grades) to separate a real effect from sampling noise.

| | local v1 | local v2 | hosted v1 | hosted v2 |
|---|---|---|---|---|
| Field accuracy | 96.1 % | 99.6 % | 96.9 % | **100 %** |
| Right where the bill has a value | 99.2 % | 99.7 % | 97.8 % | **100 %** |
| Quiet where it has none | **89.0 %** | 99.4 % | 94.8 % | **100 %** |
| Dangerous rate (`wrong` + `hallucinated`) | 3.5 % | 0.2 % | 1.6 % | **0 %** |
| Fully correct documents | 43.3 % | 93.3 % | 86.7 % | **100 %** |
| Stability across runs | — | — | **100 %** | **100 %** |
| Latency p50 | 36.8 s | 19.8 s | 3.4 s | 3.9 s |
| Cost (whole run) | 0 | 0 | $0.072 | $0.091 |

Local is `ollama:qwen2.5vl:7b`, hosted is `anthropic:claude-haiku-5-5`. `compare` reports
**18 improvements and 0 regressions** on the local model, **16 and 0** on the hosted one.

**Stability is 100 % on both hosted runs**: every one of the 510 (document, field) pairs got the
same outcome in all three repetitions. Temperature 0 does not guarantee that — it is measured, not
assumed — and it is what licenses reading a 3-point gap as a real effect rather than a lucky draw.

### Read the two middle rows again

On the 355 fields the bill actually fills, the naive prompt scores **99.2 %** on the local model
against the tuned prompt's 99.7 %. Three fields apart. **The naive prompt reads the bill essentially
as well as the tuned one.**

On the 155 fields the bill leaves empty it scores **89.0 %**: it invents a value 17 times. The
tuned prompt does it once.

| Run | correct | correct_null | missing | **wrong** | **hallucinated** |
|---|---|---|---|---|---|
| local v1 | 352 | 138 | 2 | **1** | **17** |
| local v2 | 354 | 154 | 1 | **0** | **1** |
| hosted v1 (×3) | 1041 | 441 | 24 | **0** | **24** |
| hosted v2 (×3) | 1065 | 465 | 0 | **0** | **0** |

The local model's problem was never reading. It was not knowing when to stop. v2 adds one sentence
per trap and one rule — *"if a value is absent, illegible or ambiguous, return null; null is a
correct answer, not a failure"* — and the inventions collapse from 17 to 1.

A single average would have called this "three points better". The split calls it what it is: the
difference between a form you can trust and one you cannot. **This is the whole argument for
evaluating anything at all** — not that the number went up, but that it went up *somewhere
specific*, for a reason you can name and a fix you can keep.

### The same prompt fails differently on different models — and the aggregate lies about how

The hosted model's v1 run has exactly 48 failures: 24 `missing` and 24 `hallucinated`, spread over
four fields.

| Field | hosted v1 | The summary says |
|---|---|---|
| `period_start`, `period_end` | 86.7 % | the billing period is missing on 4 documents |
| `gas_period_start`, `gas_period_end` | 86.7 % | a gas period was invented on 4 documents |

Read like that it looks like two unrelated problems, one on each side of the ledger. Now look at
what the model actually returned, which every run on non-real data stores next to the grade:

```
syn-003  period_start      expected 2026-05-01   raw None         → missing
syn-003  gas_period_start  expected None         raw '01/05/2026' → hallucinated
```

Same date. **The model read the billing period perfectly and filed it under the gas period.** It is
not two problems, it is one mislabelled field — and it happens on **all four gas-only bills in the
dataset, in all three runs**, never on a power or dual bill. On a gas bill, "periodo di
fatturazione" simply looks like the gas period, and the naive prompt gives the model nothing to
distinguish them. v2 does, in one line: *"ONLY if the bill prints a gas period different from the
bill period. Otherwise null."*

Two things follow. First, **the aggregate would have sent you after the wrong bug** — tightening the
date parser, or hunting for a layout the model cannot read, when the parser and the reading were
both already right. The raw-value traces cost a few hundred bytes per record and they turned a
four-field mystery into a one-line prompt fix.

Second, nothing in the local model's failure list overlaps with this one. Same prompt, same
documents, two entirely different weaknesses. **A prompt is tuned against a model, not in the
abstract** — which is why a harness that re-grades both in one command beats any intuition about
which wording reads better.

### Where the gain came from, on the local model

Only the fields that are not 100 % in both runs:

| Field | v1 | v2 | What changed |
|---|---|---|---|
| `annual_consumption_kwh` | 73.3 % | **96.7 %** | stopped inventing a yearly figure the bill never labels as annual |
| `is_resident` | 76.7 % | **100 %** | stopped guessing residency when the bill does not state it |
| `gas_annual_consumption_smc` | 93.3 % | **100 %** | same, for gas |
| `pdr` | 96.7 % | **100 %** | one gas code it had missed |
| `holder_name` | 96.7 % | **100 %** | dropped the `Spett.le` honorific in front of a company name |
| `holder_tax_code` | 96.7 % | 96.7 % | unchanged — see below |

The one field neither prompt fixes is instructive. On `syn-024` the model misreads one character of
a tax code (`FVTLLT…` for `FVTLTT…`). The check-character validation in `normalize.py` refuses it,
so the field comes back **empty instead of wrong** — graded `missing`, not `wrong`. A blank tax code
gets noticed; a plausible wrong one does not. That is the project's thesis caught in a measurement:
code turned a dangerous error into a visible one.

## 2. Why the comparison is clean

A prompt A/B is only worth running if the prompt file is the *only* thing that differs. That is
harder than it sounds, because a model is fed instructions from more places than the prompt.

Here v1 is the whole baseline, verbatim:

```markdown
Extract the data from the attached Italian utility bill.

Answer with a JSON object that follows this JSON Schema:

{{SCHEMA}}
```

Four lines, and `{{SCHEMA}}` expands to field names and types. Nothing else: `schema.py` carries no
prose at all — no *"the supply address, not the mailing one"*, no *"the total, not the F1/F2/F3
bands"*, no *"only if the bill labels it as annual"*. It declares shape. Every instruction about how
to read a bill lives in `prompts/`, which means the diff between two runs is a text file and the
numbers belong to that diff.

**This matters well past this repo.** A JSON Schema with helpful `description` fields is a prompt.
So is a tool definition, a function signature, a few-shot example, a system message someone else
owns. Put a rule in any of them and both arms of your A/B get it for free — the test then measures
nothing, and the conclusion you draw is that prompt wording does not matter.

What a rule is worth when only the prompt carries it, measured on one field:

| `is_resident` | v1 — no rule anywhere | v2 — rule in the prompt |
|---|---|---|
| Accuracy | 76.7 % | **100 %** |
| Dangerous errors | 7 invented values | **0** |

The rule is one sentence: *"true if the bill says 'residente', false if it says 'non residente',
otherwise null."* Eight documents in this dataset say neither — and without that sentence the model
picks a side on seven of the eight. With it, on all eight it says nothing.

## 3. A benchmark you build yourself only contains the traps you thought of

Same model, same prompt, two datasets. This is what isolates the dataset rather than the model.

| `ollama:qwen2.5vl:7b`, prompt v2 | Synthetic — 30 docs, 510 fields | Real — 8 docs, 128 fields |
|---|---|---|
| Field accuracy | 99.6 % | **82.8 %** |
| Right where the bill **has** a value | 99.7 % | **74.0 %** |
| Quiet where it has **nothing** | 99.4 % | 94.5 % |
| Dangerous errors | 1 in 510 | **19 in 128** |
| Fully correct documents | 93.3 % | **12.5 %** |
| Latency p50 | 19.8 s | 223.7 s |

> The bills are not in this repository. The run stores per-field outcomes only — no extracted value,
> no raw model answer, no discarded value — and it was produced by a model running locally, so
> nothing left the machine.

**The two accuracies swap roles, and that is the finding.** On generated bills the weakness is
over-answering: the model reads well and fills in blanks that should stay blank. On real bills it is
the exact opposite. `empty_accuracy` holds at 94.5 % — it still knows when to keep quiet — while
`filled_accuracy` falls to **74.0 %**. It is not talking too much any more. It is **misreading the
document**, which is a different problem wanting a different fix. One average over everything would
have shown neither weakness, only a number going down.

**The errors are concentrated, not spread out.** Two documents account for **14 of the 19** dangerous
errors; two others are perfect. The fields that break are the ones carrying numbers:

| Field | Accuracy on real bills |
|---|---|
| `gas_period_consumption_smc` | 50.0 % — 4 of 8 wrong |
| `period_start` / `period_end` | 62.5 % |
| `period_consumption_kwh`, `annual_consumption_kwh`, `gas_annual_consumption_smc` | 75.0 % |

Those layouts print consumption with six decimals (`Totale kWh 244,045`, `Fatturato 69,420820 smc`)
and the model reads the Italian decimal comma as a thousands separator — a thousand times too large.
One layout habit, one class of error, and it costs more than everything else combined.

The synthetic set has mailing addresses next to supply addresses, F1/F2/F3 bands next to totals, kWh
figures inside tax tables — and the model handles all of them. It does not have six-decimal
consumption figures, because that habit never occurred to me until a real bill showed it.

Two of the real bills are the **same document re-exported with a different contracted power**. Their
outcome maps come back **identical across all 16 fields** — same successes, same single failure in
the same place. Whatever the model is doing wrong, it is doing it to the content, reproducibly, not
to a rendering accident.

## 4. The quietest way to ruin all of this

Every number above rests on `expected.json` being right. For generated bills that is free: the data
is invented first and the document printed from it, so the answer is known by construction.

For real bills it is not free, and it is where an eval set dies. Writing 17 fields by hand per
document is slow, so a model proposes and a person confirms — and a person confirming 136 fields in
a row starts pressing Enter. The model's mistakes become the truth it is later measured against, the
eval reports agreement and calls it accuracy, and nothing in the numbers looks wrong, because the
thing being measured and the thing doing the measuring are now the same system.

That is why `review` refuses to be a flat list. Every draft arrives with the fields that deserve a
second look and the reason why — a value validation discarded, an undecided bill type, a field this
bill type must carry but left empty, or a disagreement with `--second-model` judged by the same
comparators the graders use. Drafts are sorted most-doubtful-first, so attention lands where
mistakes are likely instead of spreading evenly over every field of every bill
([details](HOW-IT-WORKS.md#2-where-the-right-answers-come-from)).

**A golden dataset is only golden if it was verified independently of the system it has to
measure**, and a human pressing Enter is not independent of anything.

## Reproducing these numbers

```bash
uv sync --all-extras
uv run billmeter generate                               # rebuilds dataset/synthetic identically
uv run billmeter eval -m ollama:qwen2.5vl:7b -p v1
uv run billmeter eval -m ollama:qwen2.5vl:7b -p v2
uv run billmeter compare eval/results/<v1>.json eval/results/<v2>.json
uv run billmeter report  eval/results/<v2>.json eval/results/<v1>.json
```

Runs on real bills need your own `dataset/private/` — see
[dataset/private/README.md](dataset/private/README.md).

Back to the [README](README.md) · how the measurement works: [HOW-IT-WORKS.md](HOW-IT-WORKS.md)
