"""Pipeline wiring: one call per incoming message.

- Structured intermediate outputs (confidence, risk_labels, grounding stats)
- Abstention path: low confidence → auto-escalate
- Full observability of every pipeline stage

`handle_message(text)` is the single entry point both the demo and the tests
call. It returns the full structured result, which the eval harness consumes.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.classify import classify
from src.config import load_config
from src.draft import draft
from src.escalate import decide
from src.ground import Grounder
from src.llm_client import LLMClient, LLMOfflineError
from src.taxonomy import LABELS


class Pipeline:
    def __init__(self, cfg: dict | None = None, few_shot: list[dict] | None = None):
        self.cfg = cfg or load_config()
        cache_dir = self.cfg["data"]["cache_dir"]
        self.agent = LLMClient(self.cfg["llm"], cache_dir, role="agent")
        self.grounder = Grounder(self.cfg["data"]["processed_path"], k=self.cfg["grounding"]["k"])
        self.few_shot = few_shot or []
        self.escalation_rules = self.cfg["escalation"]
        # Confidence threshold for abstention
        self.confidence_threshold = self.cfg.get("classification", {}).get("confidence_threshold", 0.4)

    def handle_message(self, text: str) -> dict:
        # Stage 1: Classification with confidence
        cls = classify(self.agent, text, self.few_shot)
        intent = cls["intent"]
        confidence = cls.get("confidence", 0.7)

        # Stage 2: Grounding with similarity scores
        retrieved = self.grounder.retrieve(text)
        grounding_stats = self.grounder.retrieval_stats(text)

        # Stage 3: Drafting (grounded in retrieved precedent)
        reply = draft(self.agent, text, intent, retrieved)

        # Stage 4: Risk-based escalation (independent of intent)
        esc = decide(
            self.agent, text, intent, self.escalation_rules,
            confidence=confidence,
            confidence_threshold=self.confidence_threshold,
        )

        return {
            "intent": intent,
            "intent_raw": cls["raw"],
            "confidence": confidence,
            "reply": reply,
            "decision": esc["decision"],
            "risk_labels": esc.get("risk_labels", []),
            "reason": esc["reason"],
            "retrieved": retrieved,
            "grounding_stats": grounding_stats,
        }


def load_few_shot(dev_path: str, max_per_intent: int = 2) -> list[dict]:
    """Load few-shot examples from the dev slice only (never test)."""
    by_intent: dict[str, list[dict]] = {l: [] for l in LABELS}
    with open(dev_path, encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            by_intent[e["intent"]].append(e)
    shots = []
    for label in LABELS:
        shots.extend(by_intent[label][:max_per_intent])
    return shots


def run_demo(cfg: dict | None = None) -> list[dict]:
    """Run the pipeline over the test slice; used by the demo/repro path."""
    cfg = cfg or load_config()
    golden_dir = Path(cfg["data"]["golden_path"]).parent
    shots = load_few_shot(str(golden_dir / "golden_dev.jsonl"))
    pipe = Pipeline(cfg, few_shot=shots)
    results = []
    with open(golden_dir / "golden_test.jsonl", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            r = pipe.handle_message(e["text"])
            r["example_id"] = e["example_id"]
            results.append(r)
    return results


if __name__ == "__main__":
    import sys

    cfg = load_config()
    # A few human-readable demo messages (independent of the golden set).
    demo_messages = [
        "is the 16:28 to Preston running on time today?",
        "I was delayed 2 hours and still haven't received my Delay Repay refund.",
        "your wifi hasn't worked for the whole journey, again.",
    ]
    pipe = Pipeline(cfg, few_shot=load_few_shot(
        str(Path(cfg["data"]["golden_path"]).parent / "golden_dev.jsonl")))
    for msg in demo_messages:
        print("=" * 70)
        print("MESSAGE:", msg)
        # Retrieval is deterministic and works offline — show it regardless.
        try:
            for i, r in enumerate(pipe.grounder.retrieve(msg, k=2)):
                print(f"  [retrieved {i+1}] customer: {r['customer'][:90]}")
                print(f"               resolution: {r['resolution'][:110]}")
                print(f"               similarity: {r.get('similarity', 'N/A'):.3f}")
        except Exception as e:  # noqa: BLE001
            print("  [retrieval error]", e)
        try:
            out = pipe.handle_message(msg)
            print(json.dumps({k: out[k] for k in
                              ("intent", "confidence", "decision", "risk_labels", "reason", "reply")},
                             ensure_ascii=False, indent=2))
        except LLMOfflineError as e:
            print("  [offline] LLM stages need an API key or a populated cache.")
            print("            Set OPENAI_API_KEY in .env and re-run: make demo")
        sys.stdout.flush()
