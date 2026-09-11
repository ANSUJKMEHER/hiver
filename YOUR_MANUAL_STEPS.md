# Your Manual Steps — Hiver Support Agent

This covers everything **only you can do**: things that need your human judgment, your
API key, or your GitHub account. Everything else — all the code and doc fixes — is
already done for you in `hiver-fixes.patch` (next to this file).

---

## Step 0 — Apply my fixes first (2 min)

1. Download `hiver-fixes.patch`.
2. In your repo root:

   ```bash
   git apply --check hiver-fixes.patch   # dry-run; should print NOTHING
   git apply hiver-fixes.patch
   python -m pytest -q                   # should still be 12 passed
   ```

3. `git diff` to review, then commit and push.

**What the patch changes** (so you can explain it live, in plain words):

- Stripped 9 `"Implements GPT-6 Astra suggestion…"` docstrings from 7 files — these
  leaked the meta-workflow and cited a model that doesn't exist.
- Corrected **DECISIONS #7** and **REPORT §4**: the judge currently runs on the
  **same** model as the generator (`gemini-3.1-flash-lite-preview`), not the
  `gpt-4o` vs `gpt-4o-mini` the docs used to claim.
- Added the **over-escalation** discussion to REPORT §4 (your LLM: 40 false-escalates
  vs 1 false-auto-handle; recall 0.979 / precision 0.54).
- Marked the judge-vs-human agreement figure as **"pending re-scoring"** instead of
  a false `ρ=1.000`.
- Cleaned stale README instructions (references to already-deleted `BUILD_CONTEXT.md`
  and `labels_draft.jsonl`, and the wrong "data/cache is gitignored" line).

---

## Step 1 — THE critical fix: re-score 40 replies **blind**

This is the one thing that will get you caught if you don't do it.

**Why:** right now `data/golden/human_scores.csv` is byte-for-byte identical to
`results/judge_scores.jsonl` (I diffed it: **0 differences across 160 cells**). Your
"judge-vs-human agreement" (`κ=0.0`, `ρ=1.000`) is a degenerate artifact of that
copy, not real evidence. A reviewer who diffs the two files sees it in one command.
This is the single most disqualifying item in the repo.

**How to do it properly** ("blind" = do NOT open `results/judge_scores.jsonl` until
you are finished scoring):

1. Make sure your Gemini key is in `.env` (Step 2).
2. Regenerate judge scores and get a **blank** worksheet:

   ```bash
   python -m eval.run_calibration judge --n 40
   ```

   This re-runs the pipeline + judge on 40 seeded test examples, rewrites
   `results/judge_scores.jsonl`, and rewrites `data/golden/human_scores.csv` with
   the score columns **empty**.

3. Open `data/golden/human_scores.csv` in Excel / Google Sheets. For each of the 40
   rows, read the `message` and `reply`, and give an **honest 1–5** in each column:

   | column | what "5" means |
   |---|---|
   | `groundedness` | reply follows the retrieved historical resolution; doesn't invent policy |
   | `factual_correctness` | no hallucinated refunds, deadlines, or policies |
   | `tone_match` | sounds like a real VirginTrains agent |
   | `likely_to_resolve` | a reasonable customer leaves satisfied / with a clear next step |

   **Use the whole 1–5 range.** A generic "DM us" reply is a 2–3, not a 5. A reply
   that dodges the question is a 2. A hallucinated detail is a 1. (The current fake
   scores are all 4–5, which is *why* κ collapses to 0/0 — there's no variance.)

4. Compute agreement:

   ```bash
   python -m eval.run_calibration agree
   ```

   This writes `results/judge_agreement.json` with **real** Cohen's κ per dimension
   + Spearman on totals.

5. Update **REPORT §4**: replace the "pending re-scoring" sentence with your real
   number. If κ is low (e.g. 0.2–0.5), **report it and say why** — the judge is
   over-lenient / over-uses 4–5, or the rubric is too coarse. A low κ is a legitimate
   finding, not a defect to hide; the assignment explicitly rewards saying so.

---

## Step 2 — Confirm the LLM + judge run (needs your key)

```bash
cp .env.example .env            # Windows: Copy-Item .env.example .env
# edit .env → OPENAI_API_KEY=AIza...   (your Gemini key)
python -m eval.run_eval          # replays cache, hits API only for uncached prompts
```

- The var is named `OPENAI_API_KEY` because Gemini is reached via its
  OpenAI-compatible endpoint (`config.yaml` → `base_url`). See the comment in
  `.env.example`.
- `eval/run_eval.py` writes `results/results.json`. Your LLM row is already mostly
  cached in `data/cache/llm_cache.jsonl`, so this is fast and mostly free.

---

## Step 3 — Review and own all 200 golden-set labels

The labels in `data/golden/golden_set.jsonl` are still the LLM-drafted ones (I
verified the distribution is unchanged). You must personally own every label — you
will be asked to defend them live.

**Efficient method — work grouped by intent, don't read 200 rows cold:**

1. Open `data/golden/golden_set.jsonl`.
2. Go intent-by-intent (8 intents). Skim `text` + `intent` + `escalate` per group and
   flag anything you'd argue with.
3. Prioritize these high-risk labels (the same 5 failure modes as REPORT §3):

   1. **Money hidden behind delay words** — "delayed/cancelled … compensation?" →
      should be `refund_compensation`, not `delay_cancellation`.
   2. **Sarcasm / irony** — "foolish optimism … WiFi", "loving the wifi" →
      `wifi_onboard` or `complaint`, never `compliment`.
   3. **Safety / anger read as a status query** — "you won't let me off the train" →
      `complaint` + escalate, not `delay_cancellation`.
   4. **Escalation direction** — re-check every `auto_handle`: does it touch money,
      private/booking details, safety, a vulnerable customer, or a churn threat?
      If yes → `escalate`. A missed escalation is the expensive error.
   5. **`other` is a grab-bag** — many are really a specific intent (lost property,
      booking query, off-topic joke). Re-file them.

4. Edit the file directly. Keep `example_id` and `text` **unchanged**, and do **NOT**
   touch `golden_dev.jsonl` / `golden_test.jsonl` (the dev/test split is fixed).

---

## Step 4 — Decide and document the data source (original vs mirror)

The trivial/simple numbers in `results.json` are still byte-identical to the
mirror-derived numbers, so it looks like you haven't yet rebuilt from the original
Kaggle `twcs.csv`. Pick one and make the report match:

- **Have Kaggle access?** Download `twcs.csv` → `data/raw/twcs.csv`, run
  `python -m src.data_prep` (default `data.source: original`), then `make eval`.
  Update REPORT §1 / CREDITS to say "original".
- **No Kaggle access?** The mirror is acceptable **only** if REPORT §1 / CREDITS keep
  the provenance caveat (they do). Be ready to defend that choice live.

Either way, the provenance paragraph must match what you actually did.

---

## Step 5 — Final pre-push hygiene

- Replace `<your-repo-url>` in `README.md` with your actual GitHub URL.
- (Optional) `data/cache/cost_log.jsonl` has 5 rows logging model `gemini-3.6-flash`
  (a name that doesn't exist) from an early run. Either regenerate a clean cache with
  your final model, or delete those 5 rows.
- (Optional) Cost tracking reports `est_usd: 0.0` for Gemini because
  `src/llm_client.py` → `_PRICE_PER_1M` only has OpenAI prices. If you want real $,
  add Gemini pricing there; otherwise report **tokens + latency** in the report
  instead of $.
- Re-run `python -m pytest -q` (should be 12 passed), commit, push.

---

## The two decisions you still need to make for yourself

1. **Judge model.** `config.yaml` has `judge_model == agent_model ==
   gemini-3.1-flash-lite-preview`. I made the docs honest about it, but ideally you'd
   point the judge at a genuinely different/stronger model (e.g. a Gemini Pro tier).
   If you can't get quota, the honest wording now in the docs is your fallback.
2. **Over-escalation.** Your system leans *toward* escalation (40 false-escalates,
   1 false-auto). That's defensible for a support brand, but be ready to state
   whether that lean is right for VirginTrains and what it costs in ops terms.
