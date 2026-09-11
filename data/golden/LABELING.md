# Golden set — sampling & labelling method

The golden evaluation set is `golden_set.jsonl` (200 examples), split into
`golden_dev.jsonl` (44) and `golden_test.jsonl` (156) by a stable hash of the
example id. `sampling_metadata.json` records the exact procedure and seed.

## Sampling

- **Frame:** all VirginTrains threads with a non-empty first customer message
  (≥10 chars) and ≥1 support reply (~13k threads).
- **Procedure (deterministic, seed 42):**
  1. Hash-shuffle the frame (stable across machines).
  2. Reserve ~10% (20 slots) for **messy cases** — short (<45 chars), emoji-heavy, or
     non-Latin-heavy messages — so hard cases are deliberately over-represented.
  3. Stratify the remainder by a **rough keyword pre-tag** (used *only* to stratify,
     not to classify), taking an equal floor per intent and topping up.
- **Exclusions:** threads with no support reply (no reference answer to ground on),
  and near-empty first messages.

## Labelling

Each example is labelled with three fields:

- `intent` — one of the 8 taxonomy labels.
- `escalate` — `auto_handle` | `escalate`, decided by an explicit rule (below), not
  by gut feel.
- `reference_reply` — a one-line note on what a good reply must contain (chosen over
  writing 200 full reference replies, so the "gold" is a *criteria* rather than a
  single canonical string).

**Escalation rule used while labelling** (the judgment middle is documented so it can
be defended live): escalate if the message (a) asks for or clearly implies money /
refund / compensation, (b) needs private/booking/account details to resolve (lost
ticket, terminated card, retrieve booking, lost property, accessibility), (c) raises
safety / legal / vulnerability (trapped passenger, unsafe overcrowding, an anxious or
autistic customer), or (d) shows a churn threat tied to a specific unresolved issue;
otherwise auto-handle.

**Method note:** Labels were drafted with LLM assistance, and manually reviewed/corrected against the source messages.
