"""Tests for judge calibration agreement + judge output shape."""
from __future__ import annotations

from eval import judge, judge_calibration


class FakeJudgeClient:
    def complete_json(self, system, user, **kw):
        return {"groundedness": 4, "factual_correctness": 5, "tone_match": 4, "likely_to_resolve": 4}


def test_judge_reply_shape():
    c = FakeJudgeClient()
    out = judge.judge_reply(c, "msg", "intent", "reply", [])
    assert set(out) == {"groundedness", "factual_correctness", "tone_match", "likely_to_resolve", "total"}
    assert out["total"] == 17


def test_agreement_runs_all_dimensions():
    j = [{"groundedness": 4, "factual_correctness": 5, "tone_match": 4, "likely_to_resolve": 4, "total": 17},
         {"groundedness": 2, "factual_correctness": 2, "tone_match": 3, "likely_to_resolve": 2, "total": 9}]
    h = [{"groundedness": 4, "factual_correctness": 5, "tone_match": 4, "likely_to_resolve": 5, "total": 18},
         {"groundedness": 2, "factual_correctness": 1, "tone_match": 3, "likely_to_resolve": 2, "total": 8}]
    out = judge_calibration.agreement(j, h)
    assert "groundedness" in out and "total" in out
    assert out["total"]["spearman_total"] == 1.0  # ranks perfectly aligned
