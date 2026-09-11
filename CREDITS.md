# Credits & Borrowed Material

Everything borrowed is listed here; the rule is "cite what you borrowed," and
not knowing is the actual violation.

## Data

- **Customer Support on Twitter** — Axelbrooke, Stuart (ThoughtVector), 2017.
  Original: Kaggle `thoughtvector/customer-support-on-twitter` (~2.8M tweets,
  columns `tweet_id, author_id, inbound, created_at, text, response_tweet_id,
  in_response_to_tweet_id`). Used under its original license; we run on a small
  subsample, never the full set. `src/data_prep.py` reconstructs threads from
  this flat reply graph (the default `data.source: original` path).
- **Conversation mirror (fallback)** — Hugging Face
  `TNE-AI/customer-support-on-twitter-conversation` (794,335 pre-reconstructed
  threads, schema `conversation_id, company, conversation, summary`).

  **Honest provenance note:** this mirror is a *third-party re-processing* of the
  same underlying twcs.csv. I could not verify its exact fidelity against the
  gated Kaggle original (no Kaggle credentials were available in the build
  environment), so the thread stitching, company-name normalisation, and text
  cleaning may differ in small ways from a reconstruction done directly on
  twcs.csv. The golden set and baseline numbers in this repo were **built from
  the mirror**. For the final run, download the original twcs.csv and use
  `data.source: original` — the reconstruction code is identical in interface, so
  the golden set (whose labels live on tweet text, not on any particular
  reconstruction) still evaluates the same systems.

## Code / libraries

- `pandas`, `pyarrow`, `numpy`, `scikit-learn` (TfidfVectorizer, LogisticRegression,
  precision/recall/F1), `PyYAML`, `requests`, `structlog`, `pytest` — standard OSS.
- No copied code snippets or prompt templates beyond general library usage. The
  LLM prompts are written from scratch for this assignment.

## Prompt / method influences

- The rubric-style LLM-as-judge (dimension scores 1–5 + a human-agreement check)
  follows the common "LLM-as-judge" evaluation pattern (e.g. Zheng et al., *Judging
  LLM-as-a-Judge with MT-Bench and Chatbot Arena*, 2023). We implement it minimally
  and add a quantitative judge-vs-human calibration, per the assignment.
- The "trivial vs simple baseline" framing and the cost-weighted escalation error
  asymmetry are standard evaluation practice, adapted to this task.

No other code, data, or prompts were copied.
