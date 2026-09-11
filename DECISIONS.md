# Decision Log

Non-obvious calls, one decision + one "why" each.

1. **Brand = VirginTrains (not Amazon/Apple/Delta).** Bounded domain (UK rail) with
   substantive, *groundable* resolutions and a clean escalation signal; Amazon/Apple
   replies are dominated by "DM us" boilerplate and too diverse for a 6–10 intent
   taxonomy. See REPORT §problem framing for the numbers.

2. **Taxonomy lives in `src/taxonomy.py`, not `config.yaml`.** The taxonomy is code
   (labels + definitions + real data examples + a label validator) that `classify.py`
   imports; config holds only tunables. A single source of truth beats splitting
   labels/definitions across yaml and code.

3. **Taxonomy is 8 intents derived by reading ~120 real threads, not chosen a priori.**
   Grounding categories in the data avoids a taxonomy that poisons everything downstream.

4. **Golden set = one example per thread, using the *first* customer message.** The
   cleanest "incoming message" unit; avoids multi-turn context ambiguity (which we
   instead list as a known limitation). Chosen over sampling arbitrary mid-thread turns.

5. **Sampling = hash-shuffle + keyword-stratification + 10% messy-case oversample.**
   A pure "first N" or pure-random sample under-represents hard cases; stratification
   (with a rough keyword pre-tag used *only* for stratification) guarantees every
   intent appears, and the messy oversample (short/emoji/non-Latin messages) keeps the
   eval honest on the cases that actually break systems.

6. **Escalation = hard rules first, LLM for the judgment middle.** Money/refund and
   booking/account intents always escalate regardless of confidence (encoded in
   `config.yaml`); the model only decides anger/churn/safety nuance. Encoding hard
   rules explicitly beats hoping the LLM infers them.

7. **Judge model currently shares the generator model
   (`gemini-3.1-flash-lite-preview`), with a distinct rubric-scorer prompt/persona.**
   A different, stronger judge would reduce self-grading bias more, but the stronger
   tier was out of quota at build time; the separate persona is a partial mitigation.
   Stated in the report, not left implicit.

8. **Retrieval = TF-IDF cosine, not a dense embedder or full RAG.** The assignment
   says "embedding similarity is enough"; TF-IDF is deterministic, dependency-light,
   fast, and cacheable. Swapping to sentence-transformers is a one-line change in
   `ground.py` (listed as a next-week item).

9. **Grounding corpus = resolved threads whose last support reply is ≥20 chars.**
   Filtering out bare "DM us" boilerplate keeps retrieval on *substantive* resolutions
   rather than the most common (but useless) reply.

10. **Cache keyed on the full prompt hash, committed to the repo.** This is what makes
    "reproduce in <15 min" true — a fresh clone replays cached calls and only hits the
    API for a small confirmation subset.

11. **Simple baseline trained on the dev slice (44 examples), with a cross-validated
    number reported alongside.** The golden set is small; 5-fold CV on the full set is
    the fair "classical" ceiling, while the dev-trained→test number is the honest
    held-out figure. The asymmetry is called out in the "misleading" section.

12. **Trivial baseline = majority class + canned reply + always auto-handle.** Auto-handle
    is the majority escalation class (71%), so always-auto is the correct floor — it
    makes the real numbers legible (12% accuracy is the "beat this or you've built
    nothing" bar).

13. **Provider-agnostic client via OpenAI-compatible HTTP, not an SDK.** Swapping
    providers/models is a config change, not a code change; also the easiest thing to
    demo live ("here's how I'd swap models").

14. **Cost/latency logged per call (tokens + estimated $).** Reporting total eval cost
    and per-message latency is the kind of detail that separates "it works" from "it's
    cheap enough to ship"; it feeds directly into the cost/quality trade-off section.

15. **`handle_message(text) -> dict` is the single entry point.** Both the demo and the
    tests call the same function, so a live "modify and re-run" is one line, and the
    eval harness consumes the exact same structured output.

16. **Support both the original `twcs.csv` and the mirror, defaulting to the original.**
    The assignment's source is the flat Kaggle CSV, but the build environment had no
    Kaggle access, so the golden set was built from a public mirror. `data_prep.py`
    reconstructs threads from the raw reply graph by default (matching the brief) and
    keeps the mirror as a documented fallback — same interface, so the golden labels
    (which live on tweet text) evaluate the same systems either way.
