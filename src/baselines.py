"""Baselines.

Two baselines are required by the assignment and both are evaluated on the same
golden test set as the LLM pipeline:

- TrivialBaseline: majority-class intent + one canned reply + always auto-handle
  (auto-handle is the majority escalation class, so this is the "beat this or
  you've built nothing" floor).
- SimpleBaseline: TF-IDF + logistic-regression intent (trained on the dev slice),
  template reply = nearest resolved thread by TF-IDF similarity (no generation),
  rule-based escalation (money/booking intents escalate, else auto-handle).

Everything here is deterministic and needs no LLM, so its numbers are real and
reproducible with no API key.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.ground import Grounder
from src.taxonomy import LABELS, default_escalation

CANNED_REPLY = "Thanks for your message. We've received it and will follow up shortly."


class TrivialBaseline:
    def __init__(self, majority_intent: str):
        self.majority = majority_intent

    def predict(self, text: str) -> dict:
        return {
            "intent": self.majority,
            "reply": CANNED_REPLY,
            "decision": "auto_handle",
            "reason": "trivial baseline always auto-handles",
        }


class SimpleBaseline:
    def __init__(self, train_examples: list[dict], grounder: Grounder):
        self.grounder = grounder
        texts = [e["text"] for e in train_examples]
        labels = [e["intent"] for e in train_examples]
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        X = self.vec.fit_transform(texts)
        self.clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.clf.fit(X, labels)

    def predict(self, text: str) -> dict:
        intent = self.clf.predict(self.vec.transform([text]))[0]
        retrieved = self.grounder.retrieve(text, k=1)
        reply = retrieved[0]["resolution"] if retrieved else CANNED_REPLY
        decision = default_escalation(intent)
        reason = f"rule: intent '{intent}' -> {decision}"
        return {"intent": intent, "reply": reply, "decision": decision, "reason": reason}


def majority_intent_from(examples: list[dict]) -> str:
    return Counter(e["intent"] for e in examples).most_common(1)[0][0]


def load_golden(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
