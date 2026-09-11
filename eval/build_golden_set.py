"""Build the golden evaluation set.

Design goals (see REPORT.md §problem framing and DECISIONS.md):
- Deterministic and reproducible (fixed seed + documented rules).
- Stratified so every intent is represented and hard/messy cases are over-sampled.
- Labels are hand-assigned (agent-drafted, then human-reviewed) — the file that
  matters is `data/golden/golden_set.jsonl`, which is checked in.

Pipeline of this script:
  1. `sample`  — draw the stratified sample from processed threads and write an
                 UNLABELED golden file plus a sampling-metadata file.
  2. `merge`   — merge a labels draft (agent-produced) into the golden file and
                 validate it.
  3. `split`   — write the fixed dev/test split (hash-based, stable across runs).

The intent pre-tagging below is a ROUGH keyword estimate used ONLY to stratify
the sample (so we don't under-represent a class). It is not the classifier.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from src.config import load_config
from src.taxonomy import LABELS, INTENT_BY_LABEL

# Rough keyword pre-tags, used ONLY for stratification of the sample.
# Ordered by priority: specific/rare intents are checked first so generic words
# ("train", "seat") don't swallow everything. First match wins.
_KEYWORD_TAGS: list[tuple[str, list[str]]] = [
    ("wifi_onboard", ["wifi", "beam", "internet", "router", "connect", "broadband", "4g", "hotspot"]),
    ("refund_compensation", ["refund", "compens", "delay repay", "money back", "reimburse",
                             "charged", "overcharged", "paid", "money", "repay"]),
    ("compliment", ["thank", "thanks", "cheers", "great", "amazing", "excellent", "brilliant",
                    "love", "shout out", "well done", "wonderful", "star", "appreciate", "massive"]),
    ("complaint", ["appalling", "awful", "disgust", "nightmare", "rubbish", "pathetic", "shame",
                   "joke", "furious", "angry", "clueless", "worst", "terrible", "unacceptable",
                   "shocking", "embarrassing", "ridiculous"]),
    ("facilities_comfort", ["socket", "power outlet", "plug", "lounge", "toilet", "food", "breakfast",
                            "cereal", "soup", "standing", "crowd", "carriage", "aircon", "air con",
                            "heating", "declassif", "first class"]),
    ("ticket_booking", ["ticket", "book", "reserve", "seat", "upgrade", "fare", "advance",
                        "weekly", "season", "pass", "change", "reservation"]),
    ("delay_cancellation", ["delay", "cancelled", "cancel", "status", "on time", "depart",
                            "due in", "disrupt", "strike", "resume", "scheduled", "terminate",
                            "running ok", "signalling", "points failure", "leaving"]),
]


def pre_tag(text: str) -> str:
    """First-match in priority order; fallback to 'other'."""
    t = text.lower()
    for label, kws in _KEYWORD_TAGS:
        if any(kw in t for kw in kws):
            return label
    return "other"


def is_messy(text: str) -> bool:
    """Heuristic for 'hard' cases we deliberately over-sample: very short,
    emoji-heavy, or non-Latin-heavy messages."""
    t = text.strip()
    if len(t) < 45:
        return True
    non_ascii = sum(1 for c in t if ord(c) > 127)
    if non_ascii / max(len(t), 1) > 0.05:
        return True
    return False


def load_threads(processed_path: str) -> list[dict]:
    out = []
    with open(processed_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec["n_support"] >= 1 and len(rec["first_customer"]) >= 10:
                out.append(rec)
    return out


def sample(args: list[str]) -> None:
    cfg = load_config()
    rng_seed = cfg["golden"]["random_seed"]
    target = cfg["golden"]["target_size"]
    messy_slots = int(target * 0.10)  # ~10% reserved for hard cases
    threads = load_threads(cfg["data"]["processed_path"])

    # Deterministic random order (hash-based so it's stable across machine/order).
    threads.sort(key=lambda t: hashlib.md5(t["thread_id"].encode()).hexdigest())
    import random
    rng = random.Random(rng_seed)
    rng.shuffle(threads)

    messy_pool = [t for t in threads if is_messy(t["first_customer"])]
    clean_pool = [t for t in threads if not is_messy(t["first_customer"])]

    chosen: list[dict] = []
    seen = set()

    # 1. Over-sample messy cases.
    for t in messy_pool:
        if len(chosen) >= messy_slots:
            break
        if t["thread_id"] not in seen:
            chosen.append(t)
            seen.add(t["thread_id"])

    # 2. Stratify the rest by keyword-estimated intent with a per-class floor.
    slots_left = target - len(chosen)
    per_intent_floor = max(8, slots_left // len(LABELS))
    by_intent: dict[str, list[dict]] = {l: [] for l in LABELS}
    for t in clean_pool:
        by_intent[pre_tag(t["first_customer"])].append(t)
    for label in LABELS:
        rng.shuffle(by_intent[label])

    # Round 1: take up to per_intent_floor from every intent to guarantee coverage.
    for label in LABELS:
        taken = 0
        for t in by_intent[label]:
            if taken >= per_intent_floor or len(chosen) >= target:
                break
            if t["thread_id"] not in seen:
                chosen.append(t)
                seen.add(t["thread_id"])
                taken += 1

    # Round 2: top up to target from the remaining (shuffled) clean pool.
    remainder = [t for pool in by_intent.values() for t in pool if t["thread_id"] not in seen]
    rng.shuffle(remainder)
    for t in remainder:
        if len(chosen) >= target:
            break
        chosen.append(t)
        seen.add(t["thread_id"])

    # Round 3: top up from the full frame if still short (shouldn't happen).
    for t in threads:
        if len(chosen) >= target:
            break
        if t["thread_id"] not in seen:
            chosen.append(t)
            seen.add(t["thread_id"])

    rng.shuffle(chosen)

    rows = []
    for t in chosen:
        rows.append({
            "example_id": t["thread_id"],
            "text": t["first_customer"],
            "intent": None,
            "escalate": None,      # "auto_handle" | "escalate"
            "reference_reply": None,  # short note on what a good reply must contain
            "thread_id": t["thread_id"],
            "n_turns": t["n_turns"],
        })

    out = Path(cfg["data"]["golden_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")

    # Sampling metadata (documented deliverable).
    meta = {
        "method": "hash-shuffled + stratified by keyword-estimated intent, with 10% "
                  "messy-case oversample (short/emoji/non-Latin messages).",
        "random_seed": rng_seed,
        "target_size": target,
        "actual_size": len(rows),
        "messy_slots": messy_slots,
        "frame": "VirginTrains threads with non-empty first customer message (>=10 chars) and >=1 support reply",
        "pre_tag_note": "keyword pre-tags are used ONLY for stratification, not classification",
        "intent_distribution_pre_tag": {l: len(by_intent.get(l, [])) for l in LABELS},
        "chosen_pre_tag_distribution": _count_pre_tags(rows),
    }
    meta_path = out.parent / "sampling_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[golden] wrote {len(rows)} unlabeled examples to {out}")
    print(f"[golden] sampling metadata -> {meta_path}")


def _count_pre_tags(rows: list[dict]) -> dict[str, int]:
    from collections import Counter
    return dict(Counter(pre_tag(r["text"]) for r in rows))


def merge(args: list[str]) -> None:
    """Merge a labels draft (JSONL: example_id, intent, escalate, reference_reply)
    into the golden file, validating labels."""
    cfg = load_config()
    golden_path = Path(cfg["data"]["golden_path"])
    labels_path = Path(args[0]) if args else golden_path.parent / "labels_draft.jsonl"

    labels = {}
    for line in labels_path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        labels[d["example_id"]] = d

    rows = [json.loads(l) for l in golden_path.read_text().splitlines() if l.strip()]
    n_missing = 0
    for r in rows:
        lab = labels.get(r["example_id"])
        if lab is None:
            n_missing += 1
            continue
        r["intent"] = lab["intent"]
        r["escalate"] = lab["escalate"]
        r["reference_reply"] = lab.get("reference_reply", "")
        _validate(r)

    golden_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    if n_missing:
        print(f"[golden] WARNING: {n_missing} examples still unlabeled")
    print(f"[golden] merged labels from {labels_path}")


def split(args: list[str]) -> None:
    """Write the fixed dev/test split using a stable hash of example_id."""
    cfg = load_config()
    golden_path = Path(cfg["data"]["golden_path"])
    rows = [json.loads(l) for l in golden_path.read_text().splitlines() if l.strip()]
    dev_frac = cfg["golden"]["dev_frac"]
    dev, test = [], []
    for r in rows:
        h = int(hashlib.md5(r["example_id"].encode()).hexdigest(), 16) % 100
        (dev if h < dev_frac * 100 else test).append(r)
    for name, subset in (("dev", dev), ("test", test)):
        p = golden_path.parent / f"golden_{name}.jsonl"
        p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in subset) + "\n")
        print(f"[golden] {name}: {len(subset)} examples -> {p}")
    # log the split sizes into metadata
    meta_path = golden_path.parent / "sampling_metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["dev_size"], meta["test_size"] = len(dev), len(test)
    meta_path.write_text(json.dumps(meta, indent=2))


def _validate(r: dict) -> None:
    from src.taxonomy import validate_label
    r["intent"] = validate_label(r["intent"])
    if r["escalate"] not in ("auto_handle", "escalate"):
        raise ValueError(f"bad escalate {r['escalate']!r} for {r['example_id']}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "sample"
    {"sample": sample, "merge": merge, "split": split}[cmd](sys.argv[2:])
