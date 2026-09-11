"""Intent taxonomy for VirginTrains.

The taxonomy was derived by reading ~120 real inbound threads for the brand
before writing any pipeline code (see REPORT.md §problem framing and
DECISIONS.md). Each intent carries a one-line definition and one real
(lightly redacted) example from the data, so the classifier prompt and the
report always quote the same source of truth.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Intent:
    label: str
    definition: str
    example: str


INTENTS: list[Intent] = [
    Intent(
        label="delay_cancellation",
        definition="A question or report about a specific train's delay, cancellation, or current running status.",
        example="why is the 1845 to London from Leeds cancelled?",
    ),
    Intent(
        label="refund_compensation",
        definition="A request or follow-up about a refund, Delay Repay compensation, or getting money back.",
        example="how long does it take a delay repay refund to occur? Been 2 weeks since I submitted it.",
    ),
    Intent(
        label="ticket_booking",
        definition="Ticket purchase, changes, seat reservations, upgrades, lost tickets, or fare questions.",
        example="I've just been charged £169.99 for a single on top of my £81 return from Wilmslow to London.",
    ),
    Intent(
        label="wifi_onboard",
        definition="Issues or questions about onboard wifi, the Beam entertainment app, or connectivity.",
        example="hi there - on train to b'ham and I don't think the wifi's working so can't access beam?",
    ),
    Intent(
        label="facilities_comfort",
        definition="Reports or complaints about onboard or station facilities: seats, power sockets, food, lounge, crowding, declassification.",
        example="coach E has no power for anyone to use the sockets. Pretty rubbish for people who have been travelling.",
    ),
    Intent(
        label="complaint",
        definition="General dissatisfaction or venting, often without a clear actionable request.",
        example="going to the dogs here aren't we. Twice this week have had to ask for cereal.",
    ),
    Intent(
        label="compliment",
        definition="Praise, thanks, or positive feedback about staff or service.",
        example="shout out to your staff at Stafford! Went the extra mile to explain all my alternatives.",
    ),
    Intent(
        label="other",
        definition="Anything not covered above: jokes, media, off-topic, or ambiguous messages.",
        example="Arjun loved his wish to be a VirginTrains driver! He didn't take any of his Virgin kit off all day!",
    ),
]

LABELS: list[str] = [i.label for i in INTENTS]
INTENT_BY_LABEL: dict[str, Intent] = {i.label: i for i in INTENTS}


def taxonomy_text() -> str:
    """Render the taxonomy as a compact block for prompt injection."""
    lines = ["Intent taxonomy (choose exactly one label):"]
    for i in INTENTS:
        lines.append(f"- {i.label}: {i.definition}")
    return "\n".join(lines)


def validate_label(label: str) -> str:
    """Normalize a (possibly noisy) model label to a canonical label.

    Raises ValueError on anything that cannot be mapped unambiguously, so a
    hallucinated label fails loudly instead of silently corrupting metrics.
    """
    label = (label or "").strip().lower()
    # strip common wrappers like "intent: delay_cancellation" or "intent = delay"
    if label.startswith("intent"):
        label = label.split("intent", 1)[1].lstrip(" :=-")
    label = label.strip('"\'')
    if label in LABELS:
        return label
    # tolerate common wrappers like "intent: delay_cancellation" or "delay/cancellation"
    for canonical in LABELS:
        if label == canonical.replace("_", "") or label.replace("/", "_") == canonical:
            return canonical
    raise ValueError(f"Unrecognized intent label: {label!r}")


def default_escalation(intent_label: str) -> str:
    """Fallback escalation rule keyed on intent, used by the trivial baseline and
    as a sanity floor for the model. Money/booking intents always escalate
    (they need account verification / private details); everything else may be
    auto-handled.
    """
    if intent_label in ("refund_compensation", "ticket_booking"):
        return "escalate"
    return "auto_handle"
