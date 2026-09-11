# Hiver Support Agent — VirginTrains

An AI support agent for **VirginTrains** (from the *Customer Support on Twitter*
dataset). For each incoming customer message it: (1) classifies intent, (2) drafts
a reply grounded in how VirginTrains historically resolved similar issues, and
(3) decides auto-handle vs. escalate-to-human with a stated reason.

The emphasis is on **proof over system**: a hand-labelled golden set, two
baselines, and an honest account of what the headline number hides. See
[`REPORT.md`](REPORT.md) for the full write-up.

---

## Requirements

- Python 3.11+
- The dataset (see **Getting the data** below)
- An OpenAI-compatible API key, **only** to produce the LLM-pipeline and judge numbers
  (the baselines run without one)

## Setup

**macOS / Linux**

```bash
git clone https://github.com/ANSUJKMEHER/hiver.git && cd hiver-support-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env          # then put your OPENAI_API_KEY in .env
```

**Windows (PowerShell)**

```powershell
git clone https://github.com/ANSUJKMEHER/hiver.git; cd hiver-support-agent
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
Copy-Item .env.example .env    # then put your OPENAI_API_KEY in .env
```

> `make` is used below for convenience; it requires GNU Make (WSL / Git Bash /
> Chocolatey on Windows). Every `make X` has a direct equivalent — run `python -m`
> with the command listed next to each target.

---

## Getting the data (one brand, never the full set)

The pipeline runs on a **subsample**: one brand (VirginTrains), never the full
~3M-tweet dataset.

**Option A — original Kaggle dataset (recommended, matches the assignment).**
Download `twcs.csv` from Kaggle
(`thoughtvector/customer-support-on-twitter`) and put it at `data/raw/twcs.csv`.
Then reconstruct threads yourself:

```bash
python -m src.data_prep        # (make data)
```

`src/data_prep.py` stitches the flat `tweet_id / response_tweet_id /
in_response_to_tweet_id` reply graph into ordered conversations, filters to
VirginTrains, and cleans the text.

**Option B — public mirror (fallback, no Kaggle account needed).** Set
`data.source: mirror` in `config.yaml` and run the same command; it downloads the
pre-reconstructed HF mirror `TNE-AI/customer-support-on-twitter-conversation`.
This mirror is derived from the same underlying dataset but is a third-party
re-processing — prefer Option A for the final run (see CREDITS.md).

## Reproduce the baselines (no key needed)

```bash
python -m eval.run_eval      # (make eval)
```

Runs trivial + simple baselines on the held-out test slice and writes
`results/results.json`. These numbers are real and deterministic.

## Produce the LLM numbers (needs your key)

The intent/ground/draft/escalate stages and the judge call an LLM. With
`OPENAI_API_KEY` set in `.env`:

```bash
python -m eval.run_eval      # now also runs the LLM pipeline row
```

Every LLM call is cached (`data/cache/llm_cache.jsonl`), so a second run replays
from cache. After running, update `REPORT.md` §2 (the LLM row of the table) and
§4 with your real numbers.

## Judge-vs-human agreement (required deliverable)

This is a three-step workflow (see `eval/run_calibration.py`):

```bash
python -m eval.run_calibration judge --n 40
#   -> scores 40 replies with the judge model, and emits a BLIND worksheet:
#      data/golden/human_scores.csv  (you score 1-5 per dimension, no judge scores shown)
python -m eval.run_calibration agree
#   -> reads your scores + the judge's, computes Cohen's kappa + Spearman,
#      writes results/judge_agreement.json
```

You personally score ~40 rows (four dimensions: groundedness, factual correctness,
tone match, likely-to-resolve). Paste the resulting kappa/Spearman numbers into
`REPORT.md` §4 ("Judge scores without calibration are unvalidated").

## Golden set — review before submitting

The 200 labels in `data/golden/golden_set.jsonl` were LLM-drafted. You must review
and own every label before submitting — see **"Reviewing the golden set"** below.

---

## Layout

```
src/          pipeline (classify, ground, draft, escalate, llm_client) + baselines
eval/         golden-set builder, metrics, judge, judge calibration, failure analysis
data/         raw (gitignored), processed, golden, cache
tests/        pytest suite
REPORT.md     the ≤6-page report
DECISIONS.md  10–15 non-obvious decisions + why
WALKTHROUGH.md plain-language architecture explainer
CREDITS.md    what was borrowed
```

## Commands

| Purpose | make | direct (Windows-friendly) |
|---|---|---|
| Fetch/prepare data | `make data` | `python -m src.data_prep` |
| Full eval | `make eval` | `python -m eval.run_eval` |
| 3-message demo | `make demo` | `python -m src.pipeline` |
| Judge calibration | `make judge-calibration` | `python -m eval.run_calibration judge --n 40` |
| Tests | `make test` | `python -m pytest -q` |

## Environment

Only `OPENAI_API_KEY` is required (OpenAI-compatible endpoint). Swap `base_url` +
`model` in `config.yaml` to use any provider — a one-file change in
`src/llm_client.py`.

---

## Reviewing the golden set (what to look for)

Open `data/golden/golden_set.jsonl` (one JSON object per line: `text`, `intent`,
`escalate`, `reference_reply`). Work through it grouped by `intent`, and flag
anything where you'd argue with the label. The five failure modes in REPORT.md §3
are your checklist — the most likely label errors are the *same* mistakes:

1. **Money hidden behind delay words** — a message containing "delayed/cancelled"
   that actually asks for compensation should be `refund_compensation`, not
   `delay_cancellation`.
2. **Sarcasm/irony** — "foolish optimism… WiFi", "loving the wifi" → `wifi_onboard`
   or `complaint`, never `compliment`.
3. **Safety/anger read as a status question** — "you won't let me off the train"
   is `complaint` (and escalate), not `delay_cancellation`.
4. **Escalation direction** — re-check every `auto_handle`: does it touch money,
   private/booking details, safety, a vulnerable customer, or a churn threat? If
   yes, it should be `escalate`. A missed escalation is the expensive error.
5. **`other` is a grab-bag** — re-read every `other` label; many are really a
   specific intent (lost property, a booking query, an off-topic joke). Move them.

Fix the file directly in `data/golden/golden_set.jsonl` — but keep `example_id`/`text`
unchanged, and don't touch `golden_dev.jsonl`/`golden_test.jsonl` (the split is fixed).

## Before making the repo public

- Confirm `.env` is not committed (it is in `.gitignore`).
- `data/raw/` and `data/processed/` are regenerable and gitignored. `data/cache/`
  is committed *intentionally* (DECISIONS #10) so a fresh clone replays cached LLM
  calls; it holds no secrets — but double-check before pushing.
- Replace `https://github.com/ANSUJKMEHER/hiver.git` above with your actual GitHub URL and submit that link
  (public, or private with access granted) through the Notion form — not email.
