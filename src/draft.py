"""Reply drafting, grounded in retrieved precedent.

The prompt is conditioned on (message, classified intent, retrieved precedent),
and is instructed to actually USE the retrieved threads — echoing the brand's
own phrasing/policy — not merely receive them as inert context. Tone/length are
constrained to match the brand's real historical replies.
"""
from __future__ import annotations

from src.llm_client import LLMClient

_SYSTEM = (
    "You are drafting the reply for a UK train operator's (VirginTrains) customer-support "
    "agent on Twitter. Write ONE short, public reply. Rules:\n"
    "- Ground the reply in the retrieved historical resolutions: reuse the brand's actual "
    "policy/practical steps rather than inventing new ones.\n"
    "- Match the brand's real voice: warm, concise, uses short sentences, signs with an "
    "agent initials-style signature (e.g. ^AB), and avoids corporate filler.\n"
    "- Keep it under ~280 characters.\n"
    "- If the retrieved precedent shows the issue was handled by asking the customer to DM "
    "details, say so instead of pretending you resolved it inline.\n"
    "- Do NOT invent refund amounts, deadlines, or policies not present in the retrieved threads.\n"
    "Reply with the message text only (no quotation marks, no preamble)."
)


def draft(client: LLMClient, message: str, intent: str, retrieved: list[dict]) -> str:
    precedent = "\n\n".join(
        f"Retrieved resolved thread {i+1}:\nCustomer: {r['customer'][:300]}\n"
        f"Brand resolution: {r['resolution'][:300]}"
        for i, r in enumerate(retrieved)
    )
    user = (
        f"Incoming customer message: {message}\nClassified intent: {intent}\n\n"
        f"Retrieved historical resolutions to ground on:\n{precedent}\n\nDraft the reply."
    )
    return client.complete(_SYSTEM, user).strip()
