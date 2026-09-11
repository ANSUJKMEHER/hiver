# Walkthrough — plain-language map of the call path

This is the mental map I use to explain the system in an interview. If a component
takes more than two sentences to explain, it should be simpler.

## One line each

- **`config.yaml`** — every tunable (brand, model names, K, sample sizes, escalation
  rules) in one place; nothing is hardcoded in scripts.
- **`src/llm_client.py`** — the only place that talks to an LLM. One provider-agnostic
  `complete()` with disk caching (prompt hash → response) and cost logging. Swap
  providers by editing `config.yaml`, not code.
- **`src/taxonomy.py`** — the 8 intent labels + one-line definitions + one real data
  example each, plus a validator that rejects hallucinated labels loudly.
- **`src/data_prep.py`** — fetch the mirror, filter to VirginTrains, stitch/parse
  threads into ordered `customer`/`support` turns, clean the text.
- **`src/ground.py`** — the "memory." TF-IDF-embeds every *resolved* thread's customer
  message; at query time returns the K most similar resolved threads to ground on.
- **`src/classify.py`** — one LLM call: taxonomy + few-shot (dev only) → intent label.
- **`src/draft.py`** — one LLM call: message + intent + retrieved precedent → a reply
  told to reuse the brand's actual phrasing/policy.
- **`src/escalate.py`** — hard rules first (money/booking intents always escalate),
  then one LLM call for the judgment middle (anger/churn/safety) → decision + reason.
- **`src/pipeline.py`** — wires the above into `handle_message(text) -> dict`; the demo
  and the tests both call this.
- **`src/baselines.py`** — trivial (majority + canned + always-auto) and simple
  (TF-IDF+LR intent, nearest-thread template reply, rule escalation).

## The call path (`handle_message`)

```
incoming message
      │
      ├─ classify()  ───────────────► intent (+ raw output, logged)
      │
      ├─ grounder.retrieve() ───────► K resolved precedent threads (offline, deterministic)
      │
      ├─ draft(intent, retrieved) ──► grounded reply
      │
      └─ escalate(intent, msg) ─────► decision + reason (hard rules → LLM judgment)
      └──► {intent, reply, decision, reason, retrieved}
```

Classification and escalation are independent of each other *except* escalation reads
the intent as one input (and can override it). Drafting depends on both the intent and
the retrieval, which is what makes the reply "grounded" rather than generic.

## The eval path (`eval/run_eval.py`)

1. Load the held-out **test** slice (156 examples).
2. Run trivial, simple, and (optionally) the LLM pipeline over the same slice.
3. `eval/metrics.py` computes intent accuracy/macro-F1/confusion matrix and escalation
   P/R/F1 + a cost-weighted error count.
4. `eval/failure_analysis.py` surfaces the worst intent/escalation cases.
5. `eval/judge.py` scores replies 1–5 across groundedness / factual-correctness /
   tone / likely-to-resolve; `eval/judge_calibration.py` checks the judge against a
   human (κ + rank correlation).

## Why this is easy to modify live

- "Swap the model" → edit `llm.agent_model`/`judge_model` in `config.yaml`.
- "Change the taxonomy" → edit `src/taxonomy.py`; the classifier and report read the
  same source.
- "Change escalation policy" → edit `escalation:` in `config.yaml` (hard rules).
- "Use a better retriever" → swap one line in `src/ground.py`.
- "Re-run everything" → `make eval` (replays cache, so it's fast and deterministic).
