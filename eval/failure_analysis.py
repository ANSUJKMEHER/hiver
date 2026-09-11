"""Failure analysis: surface the worst cases for manual review.

Given aligned predictions and gold labels, this extracts concrete failure
examples (intent misclassifications, bad escalation calls, low judge scores) so
the report's "top-5 failure modes" are built from real outputs, not vibes.
"""
from __future__ import annotations

import json


def intent_errors(preds: list[dict], gold: list[dict], limit: int = 20) -> list[dict]:
    out = []
    for p, g in zip(preds, gold):
        if p["intent"] != g["intent"]:
            out.append({
                "example_id": g.get("example_id"),
                "text": g["text"],
                "true": g["intent"],
                "pred": p["intent"],
            })
    return out[:limit]


def escalation_errors(preds: list[dict], gold: list[dict]) -> dict:
    """Split into the expensive (false auto-handle) and cheap (false escalate) errors."""
    false_auto, false_esc = [], []
    for p, g in zip(preds, gold):
        if p["decision"] != g["escalate"]:
            item = {"example_id": g.get("example_id"), "text": g["text"],
                    "true": g["escalate"], "pred": p["decision"]}
            (false_auto if g["escalate"] == "escalate" else false_esc).append(item)
    return {"false_auto_handle": false_auto, "false_escalate": false_esc}


def low_judge_scores(results: list[dict], judge_dim: str = "total", limit: int = 20) -> list[dict]:
    """results entries carry a 'judge' dict with per-dimension scores."""
    scored = [r for r in results if "judge" in r and judge_dim in r.get("judge", {})]
    scored.sort(key=lambda r: r["judge"][judge_dim])
    return scored[:limit]


def write_failures(intent_err: list[dict], esc_err: dict, low: list[dict], path: str) -> None:
    import json as _json
    from pathlib import Path
    Path(path).write_text(_json.dumps({
        "intent_errors": intent_err,
        "escalation_errors": esc_err,
        "low_judge_scores": low,
    }, indent=2, ensure_ascii=False))
