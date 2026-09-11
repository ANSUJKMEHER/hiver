"""LLM-as-judge for reply quality.

Uses a DIFFERENT, stronger model than the reply generator (configured via
`llm.judge_model`) to reduce self-grading bias. Scores each reply on a 1–5
rubric across four dimensions. The judge's own reliability is checked in
`judge_calibration.py` (agreement with a human), not assumed.
"""
from __future__ import annotations

from src.llm_client import LLMClient

DIMENSIONS = ["groundedness", "factual_correctness", "tone_match", "likely_to_resolve"]

_SYSTEM = (
    "You are an independent evaluator of customer-support replies for a UK train operator "
    "(VirginTrains). Score the reply on each dimension from 1 (poor) to 5 (excellent):\n"
    "- groundedness: does it follow the retrieved historical resolution rather than inventing policy?\n"
    "- factual_correctness: no hallucinated refunds, deadlines, or policies.\n"
    "- tone_match: matches the brand's warm, concise, personal voice.\n"
    "- likely_to_resolve: would a reasonable customer leave satisfied / with a clear next step?\n"
    "Respond with ONLY a JSON object: {\"groundedness\": int, \"factual_correctness\": int, "
    "\"tone_match\": int, \"likely_to_resolve\": int}."
)


def judge_reply(client: LLMClient, message: str, intent: str, reply: str, retrieved: list[dict]) -> dict:
    precedent = "\n".join(f"- customer: {r['customer'][:200]} -> resolution: {r['resolution'][:200]}" for r in retrieved)
    user = (
        f"Incoming message: {message}\nClassified intent: {intent}\n"
        f"Retrieved precedent:\n{precedent}\n\nGenerated reply: {reply}\n\nScore it."
    )
    parsed = client.complete_json(_SYSTEM, user)
    scores = {}
    for d in DIMENSIONS:
        try:
            scores[d] = int(parsed.get(d, 0))
        except (TypeError, ValueError):
            scores[d] = 0
    scores["total"] = sum(scores[d] for d in DIMENSIONS)
    return scores
