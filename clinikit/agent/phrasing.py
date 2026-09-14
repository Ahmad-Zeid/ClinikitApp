"""
Letting Gemini word the reply -- and checking what it wrote before sending it.

THE SIMPLE VERSION
    The hand-written replies in responder.py are always correct, but they sound stiff,
    and they can only ever speak English. A real clinic in Beirut needs better than that.

    So we let Gemini write the reply instead. But we do not trust it. Before the patient
    sees anything, we check what Gemini wrote against the facts we gave it. If it mentions
    a time we never offered, a doctor we never named, or claims a booking that never
    happened, we throw it away and send the hand-written version.

WHY THIS IS SAFE WHEN "JUST ASK THE AI" IS NOT
    Gemini is trained to be agreeable. Ask it to reply to a patient and it will sometimes
    write "Booked! I've also sent you a reminder." Nobody sent a reminder. For a clinic
    that matters: the patient acts on something that did not happen.

    The difference here is that we already know every fact allowed to appear in the reply.
    The system produced them. So checking is possible -- we are not asking "is this true?",
    we are asking "is every detail in this sentence one of the details I handed over?"

    That is a much easier question, and it can be answered with ordinary string matching.

WHAT HAPPENS WHEN THE CHECK FAILS
    The patient gets the hand-written reply. They never see anything wrong. The failure is
    recorded so we can see how often it happens.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

from .backends.base import ExtractorUnavailable

# Words that only make sense if something really was written to the appointment book.
# If nothing changed, none of these may appear in the reply.
_COMPLETION_CLAIMS = [
    "i've booked", "i have booked", "you're booked", "you are booked",
    "has been booked", "is booked", "booking is confirmed", "i've confirmed",
    "i have confirmed", "has been cancelled", "has been canceled", "i've cancelled",
    "i have cancelled", "has been moved", "i've moved", "i have moved",
    "has been rescheduled", "appointment is set", "all set for",
]

# Promises about things this system cannot do at all.
_IMPOSSIBLE_CLAIMS = [
    "sent you a reminder", "sent a reminder", "sent you a confirmation",
    "email", "e-mail", "text message", "sms", "call you back", "ring you",
    "added to your calendar", "invoice", "payment", "insurance",
]

_TIME_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", re.I)
_TIME_24_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_DOCTOR_RE = re.compile(r"\bDr\.?\s+([A-Z][a-z]+)", re.I)


@dataclass(frozen=True)
class ReplyFacts:
    """Everything the reply is allowed to mention, plus the safe fallback."""

    template: str
    """The hand-written reply. Always correct. Used whenever the check fails."""

    action: str
    changed_the_book: bool
    doctors: tuple[str, ...] = ()
    times: tuple[datetime, ...] = ()
    reference: str | None = None
    extra_notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PhrasedReply:
    text: str
    source: str
    """'gemini' if the model's wording was used, otherwise 'template'."""

    rejected_because: str | None = None


def _allowed_times(times: tuple[datetime, ...]) -> set[str]:
    """Every way a permitted time might reasonably be written."""
    allowed: set[str] = set()
    for t in times:
        h24, h12, m = t.hour, (t.hour % 12) or 12, t.minute
        allowed.update({
            f"{h24}:{m:02d}", f"{h12}:{m:02d}",
            f"{h12}:{m:02d}am" if h24 < 12 else f"{h12}:{m:02d}pm",
        })
        if m == 0:
            allowed.add(f"{h12}am" if h24 < 12 else f"{h12}pm")
    return allowed


def verify(candidate: str, facts: ReplyFacts) -> str | None:
    """
    Check a proposed reply. Returns a reason to reject it, or None if it is fine.

    Deliberately strict. Sending the hand-written reply is a small loss; sending a reply
    that invents a time is a real one.
    """
    low = candidate.lower()

    # 1. Does it claim something happened that did not?
    if not facts.changed_the_book:
        for claim in _COMPLETION_CLAIMS:
            if claim in low:
                return f"claims completion ({claim!r}) but nothing was written"

    # 2. Does it promise something this system cannot do?
    for claim in _IMPOSSIBLE_CLAIMS:
        if claim in low:
            return f"promises something the system cannot do ({claim!r})"

    # 3. Does every time it mentions come from the facts we supplied?
    allowed = _allowed_times(facts.times)
    mentioned: set[str] = set()
    for h, m, ampm in _TIME_RE.findall(candidate):
        mentioned.add(f"{int(h)}:{int(m or 0):02d}{ampm.lower()}")
        mentioned.add(f"{int(h)}{ampm.lower()}" if not m else f"{int(h)}:{int(m):02d}{ampm.lower()}")
    for h, m in _TIME_24_RE.findall(candidate):
        mentioned.add(f"{int(h)}:{int(m):02d}")

    for token in mentioned:
        bare = token.replace("am", "").replace("pm", "")
        if token not in allowed and bare not in {a.replace("am", "").replace("pm", "")
                                                 for a in allowed}:
            return f"mentions a time we did not offer ({token})"

    # 4. Does every doctor it names come from the facts we supplied?
    permitted = {d.lower() for name in facts.doctors for d in name.replace("Dr.", "").split()}
    for surname in _DOCTOR_RE.findall(candidate):
        if facts.doctors and surname.lower() not in permitted:
            return f"names a doctor we did not mention (Dr. {surname})"

    # 5. Sanity: a reply that is far longer than the template is probably padded out.
    if len(candidate) > max(400, len(facts.template) * 3):
        return "much longer than the approved reply; likely padded with invented detail"

    return None


SYSTEM_PROMPT = """\
You write short replies from a medical clinic in Beirut to a patient messaging them.

You will be given an approved reply and the facts behind it. Rewrite the approved reply so
it sounds like a warm, capable human receptionist.

HARD RULES
- Use ONLY the facts given. Never add a time, a date, a doctor or a detail that is not
  there.
- Never say something was booked, cancelled or moved unless the facts say it was.
- Never promise reminders, emails, texts, calls, calendar invites or payments. The clinic
  cannot do these.
- Keep it to two or three short sentences.
- If the patient wrote in Arabizi (Lebanese Arabic in Latin letters), reply the same way.
  Otherwise reply in English.
- Do not add a greeting unless the approved reply has one.

Return only the reply text. No quotes, no explanation."""


class Phraser:
    """Rewrites approved replies in natural language, and checks the result."""

    def __init__(self, extractor=None) -> None:
        """`extractor` is a GeminiExtractor -- we reuse its client, throttle and retries."""
        self._extractor = extractor
        self.rejections: list[str] = []

    @property
    def available(self) -> bool:
        return self._extractor is not None and hasattr(self._extractor, "write_text")

    def rephrase(self, facts: ReplyFacts, patient_message: str = "") -> PhrasedReply:
        if not self.available:
            return PhrasedReply(facts.template, "template", "no language model available")

        prompt = self._prompt(facts, patient_message)
        try:
            candidate = self._extractor.write_text(SYSTEM_PROMPT, prompt).strip().strip('"')
        except Exception as exc:  # noqa: BLE001
            return PhrasedReply(facts.template, "template",
                                f"model unavailable: {type(exc).__name__}")

        if not candidate:
            return PhrasedReply(facts.template, "template", "model returned nothing")

        problem = verify(candidate, facts)
        if problem:
            self.rejections.append(problem)
            return PhrasedReply(facts.template, "template", problem)

        return PhrasedReply(candidate, getattr(self._extractor, "name", "model"))

    @staticmethod
    def _prompt(facts: ReplyFacts, patient_message: str) -> str:
        lines = []
        if patient_message:
            lines.append(f"The patient wrote: {patient_message}")
        lines.append(f"\nApproved reply (rewrite this):\n{facts.template}")
        lines.append("\nFacts you may use:")
        lines.append(f"- action taken: {facts.action}")
        lines.append(f"- appointment book changed: {'yes' if facts.changed_the_book else 'NO'}")
        if facts.doctors:
            lines.append(f"- doctors mentioned: {', '.join(facts.doctors)}")
        if facts.times:
            lines.append("- times mentioned: " +
                         ", ".join(f"{t:%A %d %B %H:%M}" for t in facts.times))
        if facts.reference:
            lines.append(f"- booking reference: {facts.reference}")
        lines.append("\nRewrite the approved reply. Add nothing.")
        return "\n".join(lines)
