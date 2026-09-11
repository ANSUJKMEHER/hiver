"""Tests for the pipeline: wired stages, hard escalation rules, grounding."""
from __future__ import annotations

import pytest

from src import escalate, taxonomy
from src.ground import Grounder
from src.llm_client import LLMClient


class FakeClient:
    """Deterministic stub for LLMClient — no network, no cache."""
    def __init__(self, json_response=None, text_response=None):
        self.json_response = json_response or {"intent": "other"}
        self.text_response = text_response or "A canned reply."

    def complete(self, system, user, **kw):
        return self.text_response

    def complete_json(self, system, user, **kw):
        return self.json_response


def test_taxonomy_validate_normalizes_wrappers():
    assert taxonomy.validate_label("delay_cancellation") == "delay_cancellation"
    assert taxonomy.validate_label("Intent: refund_compensation") == "refund_compensation"


def test_taxonomy_rejects_hallucinated_label():
    with pytest.raises(ValueError):
        taxonomy.validate_label("free_pizza")


def test_escalation_hard_rule_fires_without_llm():
    client = FakeClient(json_response={"decision": "auto_handle", "reason": "x"})
    out = escalate.decide(client, "I want my money back", "refund_compensation",
                          {"always_escalate_intents": ["refund_compensation", "ticket_booking"]})
    assert out["decision"] == "escalate"


def test_escalation_llm_judgment_path():
    client = FakeClient(json_response={"decision": "escalate", "reason": "churn risk"})
    out = escalate.decide(client, "never travelling with you again", "complaint",
                          {"always_escalate_intents": []})
    assert out["decision"] == "escalate"
    assert out["reason"]


def test_grounder_returns_k_resolved_threads(tmp_path):
    # build a tiny processed file with resolved + unresolved threads
    lines = [
        {"thread_id": "1", "resolved": True, "n_support": 1,
         "turns": [{"role": "support", "text": "DM us your booking reference and we will look into it."}],
         "first_customer": "my train was delayed and I want compensation"},
        {"thread_id": "2", "resolved": True, "n_support": 1,
         "turns": [{"role": "support", "text": "Wifi is available in standard class, log on to VirginTrains."}],
         "first_customer": "is there wifi on board"},
        {"thread_id": "3", "resolved": False, "n_support": 0,
         "turns": [{"role": "customer", "text": "help"}],
         "first_customer": "help"},
    ]
    import json
    p = tmp_path / "threads.jsonl"
    p.write_text("\n".join(json.dumps(l) for l in lines))
    g = Grounder(str(p), k=1)
    out = g.retrieve("I was delayed and want a refund")
    assert len(out) == 1
    assert "compensation" in out[0]["resolution"] or "DM" in out[0]["resolution"]


def test_llm_cache_roundtrip(tmp_path):
    import os
    os.environ.pop("OPENAI_API_KEY", None)
    cfg = {"provider": "openai", "base_url": "https://example.com/v1",
           "agent_model": "models/gemini-3.1-flash-lite-preview", "judge_model": "models/gemini-3.1-flash-lite-preview",
           "temperature": 0.0, "max_tokens": 256, "api_key_env": "OPENAI_API_KEY"}
    c = LLMClient(cfg, str(tmp_path), role="agent")
    # no key, no cache -> should raise
    from src.llm_client import LLMOfflineError
    with pytest.raises(LLMOfflineError):
        c.complete("sys", "user")
