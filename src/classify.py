"""Intent classification with confidence score.

Implements GPT-6 Astra suggestion: structured intermediate outputs with
schema-validated confidence scores alongside intent labels. Confidence
feeds the abstention path in the pipeline (low confidence → auto-escalate).
"""
from __future__ import annotations

import logging

from src.llm_client import LLMClient, _extract_json
from src.taxonomy import taxonomy_text, validate_label

log = logging.getLogger("classify")

_SYSTEM = (
    "You are the intent-classification stage of a customer-support agent for a UK train "
    "operator (VirginTrains). Given an incoming customer message, assign exactly one intent "
    "from the taxonomy below. Be strict: prefer the more specific intent when several could "
    "apply.\n\n"
    "Respond with ONLY a JSON object of the form:\n"
    "{\"intent\": \"<label>\", \"confidence\": <0.0-1.0>}\n\n"
    "where confidence is your estimated probability that this is the correct intent "
    "(1.0 = certain, 0.5 = coin flip, <0.3 = very unsure).\n\n"
    + taxonomy_text()
)


def classify(client: LLMClient, text: str, few_shot: list[dict] | None = None) -> dict:
    """Return {'intent': canonical_label, 'confidence': float, 'raw': model output}."""
    shots = ""
    if few_shot:
        shots = "\n\nExamples:\n" + "\n".join(
            f"- message: {e['text']!r} -> intent: {e['intent']}" for e in few_shot
        )
    user = f"Message:\n{text}\n{shots}\n\nReturn the JSON."
    raw = client.complete(_SYSTEM, user)
    parsed = _extract_json(raw) if _looks_json(raw) else _parse_label(raw)
    label = validate_label(parsed.get("intent", ""))

    # Extract confidence — default to 0.7 if model didn't return one
    try:
        confidence = float(parsed.get("confidence", 0.7))
        confidence = max(0.0, min(1.0, confidence))  # clamp to [0, 1]
    except (ValueError, TypeError):
        confidence = 0.7

    log.info("classify raw=%r -> %s (confidence=%.2f)", raw, label, confidence)
    return {"intent": label, "confidence": confidence, "raw": raw}


def _looks_json(raw: str) -> bool:
    return "{" in raw and "intent" in raw


def _parse_label(raw: str) -> dict:
    # tolerate a bare label with no JSON wrapper
    return {"intent": raw.strip().strip('"').strip("'")}
