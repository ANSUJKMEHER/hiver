"""Escalation decision with explicit risk labels and hybrid routing.

Implements GPT-6 Astra suggestions:
- Independent risk model with explicit labels (safety_risk, financial,
  pii_exposure, anger_churn, low_confidence, operational_disruption)
- Hybrid routing: deterministic keyword rules for high-risk patterns
  PLUS LLM judgment for ambiguous cases
- Escalation reasons over MESSAGE CONTENT, not just intent
"""
from __future__ import annotations

import re

from src.llm_client import LLMClient

# ── Risk labels ────────────────────────────────────────────────────────
RISK_LABELS = [
    "safety_risk",           # passenger safety, trapped, overcrowding
    "financial",             # refund, compensation, delay repay, overcharged
    "pii_exposure",          # personal data, booking ref, card details
    "anger_churn",           # angry/threatening to leave, complaint escalation
    "legal_threat",          # legal action, ombudsman, GDPR
    "vulnerability",         # disabled, autistic, elderly, medical
    "operational_disruption", # stranded, no service, major disruption
    "low_confidence",        # classifier confidence below threshold
    "none",                  # no risk detected — safe to auto-handle
]

# ── Keyword-based hard rules (hybrid routing) ──────────────────────────
# These fire deterministically BEFORE the LLM, catching clear patterns
# that must always escalate regardless of model confidence.
_SAFETY_PATTERNS = re.compile(
    r"\b(unsafe|dangerous|trapped|locked\s+in|stranded|emergency|"
    r"overcrowd|can.?t\s+get\s+off|fire|injured|accident)\b", re.I
)
_FINANCIAL_PATTERNS = re.compile(
    r"\b(refund|compensat|delay\s*repay|overcharg|money\s*back|"
    r"reimburse|charged\s+twice|want\s+my\s+money)\b", re.I
)
_PII_PATTERNS = re.compile(
    r"\b(booking\s*ref|reservation\s*number|card\s*details|"
    r"account\s*number|personal\s*data|my\s+address|passport|NI\s*number)\b", re.I
)
_LEGAL_PATTERNS = re.compile(
    r"\b(lawyer|solicitor|legal\s*action|sue\s+you|court|"
    r"ombudsman|GDPR|data\s*protection|trading\s*standards)\b", re.I
)
_VULNERABILITY_PATTERNS = re.compile(
    r"\b(disab|wheelchair|autis|anxiety|panic|elderly|"
    r"medical|pregnant|child\s*alone|unaccompanied)\b", re.I
)

_KEYWORD_RULES = [
    (_SAFETY_PATTERNS,        "safety_risk",            "keyword match: safety concern detected"),
    (_FINANCIAL_PATTERNS,     "financial",              "keyword match: financial/refund request"),
    (_PII_PATTERNS,           "pii_exposure",           "keyword match: personal/booking data mentioned"),
    (_LEGAL_PATTERNS,         "legal_threat",           "keyword match: legal/regulatory language"),
    (_VULNERABILITY_PATTERNS, "vulnerability",          "keyword match: vulnerable customer signal"),
]

# ── LLM prompt for judgment-call middle ────────────────────────────────
_SYSTEM = (
    "You are the escalation stage of a customer-support agent for a UK train operator "
    "(VirginTrains). Decide whether the incoming message can be auto-handled with a public "
    "reply, or must be escalated to a human (DM/private channel).\n\n"
    "IMPORTANT: Base your decision on the MESSAGE CONTENT, not just the intent label. "
    "Escalate when the message involves:\n"
    "- Money/refunds/compensation (risk: financial)\n"
    "- Account or booking verification (risk: pii_exposure)\n"
    "- Private personal data (risk: pii_exposure)\n"
    "- Safety concerns (risk: safety_risk)\n"
    "- Legal threats (risk: legal_threat)\n"
    "- A vulnerable customer (risk: vulnerability)\n"
    "- A strongly angry/churn-risk customer (risk: anger_churn)\n"
    "- Major operational disruption (risk: operational_disruption)\n\n"
    "Auto-handle simple status questions, factual questions, mild feedback, and compliments.\n\n"
    "Respond with ONLY a JSON object:\n"
    "{\"decision\": \"auto_handle\"|\"escalate\", "
    "\"risk_labels\": [\"<zero or more labels from: safety_risk, financial, pii_exposure, "
    "anger_churn, legal_threat, vulnerability, operational_disruption, none>\"], "
    "\"reason\": \"<one short sentence>\"}"
)


def detect_keyword_risks(message: str) -> list[tuple[str, str]]:
    """Return list of (risk_label, reason) from deterministic keyword rules."""
    hits = []
    for pattern, label, reason in _KEYWORD_RULES:
        if pattern.search(message):
            hits.append((label, reason))
    return hits


def decide(client: LLMClient, message: str, intent: str, hard_rules: dict,
           confidence: float = 1.0, confidence_threshold: float = 0.5) -> dict:
    """Escalation decision with risk labels and hybrid routing.

    Order of precedence:
    1. Low classifier confidence → escalate (abstention path)
    2. Hard intent rules (from config) → escalate
    3. Keyword-based risk detection → escalate
    4. LLM judgment call → escalate or auto_handle
    """
    risk_labels = []
    reasons = []

    # 1. Abstention: low classifier confidence
    if confidence < confidence_threshold:
        risk_labels.append("low_confidence")
        reasons.append(f"classifier confidence {confidence:.2f} below threshold {confidence_threshold}")

    # 2. Hard intent rules (from config — backward compatible)
    always = set(hard_rules.get("always_escalate_intents", []))
    if intent in always:
        risk_labels.append("financial" if intent == "refund_compensation" else "pii_exposure")
        reasons.append(f"intent '{intent}' requires account/money handling")

    # 3. Keyword-based risk detection (deterministic, fast)
    kw_hits = detect_keyword_risks(message)
    for label, reason in kw_hits:
        if label not in risk_labels:
            risk_labels.append(label)
            reasons.append(reason)

    # If any deterministic rule fired, escalate immediately (no LLM call needed)
    if risk_labels:
        return {
            "decision": "escalate",
            "risk_labels": risk_labels,
            "reason": "; ".join(reasons),
        }

    # 4. LLM judgment call for the ambiguous middle
    user = f"Incoming message: {message}\nClassified intent: {intent}\n\nReturn the JSON."
    parsed = client.complete_json(_SYSTEM, user)
    decision = parsed.get("decision", "").strip().lower()
    if decision not in ("auto_handle", "escalate"):
        decision = "escalate"  # fail-safe toward human

    llm_risks = parsed.get("risk_labels", [])
    if isinstance(llm_risks, list):
        risk_labels.extend(r for r in llm_risks if r in RISK_LABELS)
    if not risk_labels:
        risk_labels = ["none"] if decision == "auto_handle" else ["anger_churn"]

    reason = parsed.get("reason", "no reason given")
    return {"decision": decision, "risk_labels": risk_labels, "reason": reason}
