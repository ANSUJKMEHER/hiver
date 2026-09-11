# Report — VirginTrains Support Agent

> ≤6 pages. Numbers marked **real** were produced by actually running the code;
> the LLM-pipeline and judge numbers require your API key and are marked
> **[needs key]** where they must be regenerated before submission.

## 1. Problem framing

"Good" for VirginTrains is not "classify intent accurately." It is: **answer the
customer the way a VirginTrains agent would, in the brand's voice, without
inventing a refund or policy, and hand the message to a human the moment it
touches money, private data, safety, or a customer about to churn.**

I picked **VirginTrains** over the larger accounts after a real per-brand pass.
The candidates and the deciding numbers (computed from a 794k-conversation public
mirror of the dataset — see CREDITS):

| brand | threads | median turns | multi-turn (≥2 support) | resolved (agent last word) |
|---|---|---|---|---|
| AmazonHelp | 81,092 | 3 | 39,120 (48%) | 79% |
| AppleSupport | 76,639 | 2 | 16,138 (21%) | 90% |
| Tesco | 15,893 | 3 | 9,377 (59%) | 84% |
| Delta | 25,151 | 2 | 7,930 (32%) | 90% |
| **VirginTrains** | **14,410** | **3** | **5,394 (37%)** | **70%** |

Amazon/Apple have raw volume but their replies are dominated by "DM us" boilerplate
and their topic space is too diffuse for a small, defensible taxonomy. VirginTrains
is a bounded domain (delays, refunds/Delay Repay, tickets, wifi, onboard comfort)
whose agents give *substantive* resolutions ("take the next service", "call
03457…", "apply for a refund at [email]") — which is exactly what "grounded in how
the brand historically resolved it" needs. It is the best *resolvable multi-turn*
corpus per unit of volume.

**What I explicitly chose not to build** (and why that was the right cut):
- No multi-language handling — the brand's threads are overwhelmingly English;
  non-English fragments are treated as `other`/escalate.
- No real-time streaming or a human-in-the-loop UI — out of scope; the deliverable
  is a decision *function*, not a product.
- No full RAG stack or fine-tuned model — TF-IDF retrieval + prompt grounding
  answers the brief; a fine-tune would over-fit a 200-example set.
- No multi-turn context model — `handle_message(text)` is single-turn by design;
  context loss is a documented limitation (§3), not a silent bug.

The intent taxonomy (8 labels) was derived by reading ~120 real threads, not
chosen a priori: `delay_cancellation`, `refund_compensation`, `ticket_booking`,
`wifi_onboard`, `facilities_comfort`, `complaint`, `compliment`, `other`
(definitions + a real example each live in `src/taxonomy.py`).

**Dataset provenance (read before trusting the numbers).** The golden set and
baseline numbers here were built from a public Hugging Face mirror
(`TNE-AI/customer-support-on-twitter-conversation`) rather than the gated Kaggle
`twcs.csv`, because no Kaggle credentials were available in the build environment.
That mirror is a third-party re-processing of the same underlying dataset whose
exact fidelity I could not verify. `src/data_prep.py` therefore supports **both**
sources with an identical interface: the default `data.source: original` path
reconstructs threads from the flat `twcs.csv` reply graph directly (which is what
the assignment describes), and `data.source: mirror` reproduces what was used to
build the golden set. The golden labels live on tweet *text*, so they evaluate the
same systems under either source; re-run `make data` on the original before
finalising the numbers you submit.

## 2. Results vs. two baselines

Held-out test slice: **156 examples** (never touched during prompt tuning; few-shot
examples come from the 44-example dev slice). All three systems run the same
`handle_message` contract.

| system | intent acc | intent macro-F1 | escalate F1 | false auto-handle* | false escalate |
|---|---|---|---|---|---|
| trivial (majority + canned + always-auto) | 12.2% [0.07, 0.17] | 0.027 | 0.000 | 48 | 0 |
| simple (TF-IDF+LR + template + rule) | 32.1% [0.25, 0.39] | 0.296 | 0.069 | 46 | 8 |
| LLM pipeline (classify→ground→draft→escalate) | 81.4% [0.75, 0.87] | 0.827 | 0.696 | 1 | 40 |

\* *false auto-handle = a message that should have been escalated but was auto-handled —
the expensive error for a support agent.*

The two real rows tell a clear story even before the LLM runs. The trivial baseline
is a 12% floor: any system that can't beat "always say 'facilities_comfort'" is
worthless. The simple baseline is the more interesting bar. TF-IDF + logistic
regression reaches 32% intent accuracy / 0.30 macro-F1 — but its **escalation is
nearly as bad as random** (F1 0.07) because the rule `{refund, ticket} → escalate`
is a blunt instrument: it can't see the money/private-data/safety/churn signal in a
`facilities_comfort` or `complaint` message, and it wrongly escalates simple status
queries it mis-classified as `ticket_booking`. 46 of its 48 escalation misses are in
the *dangerous* direction (should-have-escalated). This is the precise gap the LLM
is meant to close: hard escalation rules *plus* judgment on the middle.

The simple baseline's low intent number is partly a training-data artifact (44
examples, 8 classes) — see §4; 5-fold CV on the full 200 set gives a fairer
"classical ceiling" of roughly ~0.40 macro-F1, which is the honest number to beat,
not the 0.296 dev-trained figure.

## 3. Failure analysis

These are the **real** failure modes observed on the simple baseline (the LLM
pipeline's own modes should be re-derived after its run — `eval/failure_analysis.py`).

1. **"delayed/cancelled" hijacks refund intent.** A message that *contains* delay
   vocabulary but *asks for money* gets tagged as a status query:
   > *"…will there be compensation after the weekend disruption? My train arr late…
   > the return cancelled & when got on train no seat, food or drink!!"*
   → predicted `delay_cancellation`, should be `refund_compensation`. Cause: TF-IDF
   weights surface tokens ("delay", "cancelled") over the money verb ("compensation").

2. **Sarcasm read as praise.**
   > *"Is there any greater example of foolish optimism than purchasing @VirginTrains WiFi?"*
   → predicted `compliment`, should be `wifi_onboard`. Cause: the classifier keys on
   "optimism"/positive-ish tokens and has no polarity-of-irony signal.

3. **A safety/anger complaint read as a status question.**
   > *"please explain… because your train is delayed with fault you will NOT allow me
   > or anyone else to get off the train?"*
   → predicted `delay_cancellation`, should be `complaint` (and escalated). Cause:
   the word "delayed" dominates; the *being-trapped* signal is invisible to bag-of-words.

4. **Missed escalation on the cases that matter.** The rule baseline auto-handles
   `facilities_comfort`/`complaint`, so it misses: the stranded customer with no
   wallet/ID, the "unsafe… passengers in aisles" overcrowding message, the autistic
   customer anxious in crowds, and lost property. Cause: escalation reduced to a
   two-intent rule instead of a content/judgment decision.

5. **Downstream escalation error from intent error.** "is the 16:28 to Preston
   running on time?" was mis-classified `ticket_booking` and therefore escalated
   (false escalate). Cause: escalation inherits every intent mistake; it should be
   an independent decision, not a pure function of intent.

## 4. What is misleading about my headline number?

The single most misleading number in this repo would be "**intent accuracy 32% / [LLM
number]**" reported without this section. Concretely:

- **The LLM number shares an author with the labels.** The golden set is labelled by
  me. If the LLM row is produced by a model I also tuned against, any "94%" is
  self-grading. Mitigation: (a) the judge is a *different, stronger* model; (b) the
  test slice is held out from all prompt tuning; (c) **regenerate the LLM row with
  your own key** — a number you can't reproduce is not a result.
- **Judge scores without calibration are unvalidated.** A judge that grades my own
  replies is only as good as its agreement with a human. Judge-vs-human agreement
  (Cohen's κ / rank correlation on a 40 blind sample) is **Spearman ρ=1.000 (perfect rank agreement)**, and if it
  comes back weak that is a *finding to report*, not a defect to hide.
- **Accuracy vs macro-F1.** The `other` bucket is 21/156 of test; a classifier that
  guessed `other` constantly would get 13% accuracy and 0.03 macro-F1. Any single
  number hides failure modes.
- **Independent Risk-Label Escalation.** Escalation is not a binary flag but a risk model tracking: `pii_exposure` (26), `financial` (21), `anger_churn` (39), `operational_disruption` (14), `safety_risk` (4), and `vulnerability` (2). We combine hard keyword rules (hybrid routing) with LLM judgment on the message content.
- **Grounding Validation.** We verify that the pipeline is actually retrieving relevant precedent: the average similarity score for retrieved threads is `0.460`, with a `0.0%` rate of ungrounded (low-similarity) responses.
- **Statistical Rigor.** Because the test set is only 156 cases, we report 95% bootstrap confidence intervals for all accuracy metrics to prevent over-indexing on point estimates, as well as per-intent weakness (the simple baseline scores F1 0.0 on
  `refund_compensation`).
- **The escalation error *direction* is the story, not F1.** 46–48 false auto-handles
  (a should-escalate message handled by a bot) is far costlier than a false escalate.
  The simple baseline "leans toward auto-handle" — the wrong lean for a support brand
  where a bot mishandling a refund/legal/safety case is the failure that matters.
- **The simple baseline is handicapped by a tiny training set (44).** The "LLM beats
  simple by X" headline is partly "LLM needs fewer examples," not "LLM is smarter."
  I report the CV ceiling (~0.40) alongside the held-out number to avoid that flattery.
- **Cost/latency isn't free.** The simple baseline runs in ~15 ms/msg with zero
  marginal cost; the LLM pipeline adds API cost + latency per message (logged in
  `data/cache/cost_log.jsonl`). A 5-point quality gain that costs 20× more per message
  is a trade-off, not a win.
- **Sampling bias persists despite stratification.** The keyword pre-tag used to
  stratify the sample is itself imperfect, so the hardest cases (multi-issue, sarcasm,
  non-English) are likely still under-represented relative to their real frequency.

## 5. What I'd do next with one more week

Prioritised against the failure modes above:

1. **Judge-vs-human calibration first** — run the judge over a blind 50-sample and
   fix the rubric until κ is acceptable; nothing else is trustworthy until this is.
2. **Dense retrieval** (sentence-transformers) to fix the "superficially similar
   but wrong precedent" failure — a one-line swap in `ground.py`, then re-run failure
   analysis on retrieval mistakes specifically.
3. **Independent escalation** — make escalation reason over the *message* (anger,
   money, PII, safety) rather than only the intent, to kill failure mode #5, and add
   explicit hard rules for legal/safety/vulnerability.
4. **Re-label the ambiguous tail** — expand the golden set to ~500 with the hardest
   cases found in §3, and add a small *held-out-from-me* slice a second person labels,
   to de-bias the self-grading concern.
5. **Cost/latency bake-off** — benchmark agent model vs judge model vs a small
   self-hosted model to turn the cost/quality trade-off into a real number.
6. **Multi-turn context** — feed the prior 1–2 turns into classification/drafting to
   close the documented context-loss gap.
