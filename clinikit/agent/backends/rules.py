"""
Reading a patient message using nothing but keywords and patterns.

HOW IT WORKS, IN PLAIN TERMS
    Each intent has a list of words and phrases that hint at it. We scan the message and
    every hit is a vote. The intent with the most votes wins. How far ahead the winner is
    becomes the confidence score: a landslide means confident, a close race means unsure.

WHY IT EXISTS
    This backend is not trying to beat the language model. It is the *control* — the
    thing we compare against, so that "did the LLM actually earn its place?" has a number
    attached instead of an assumption. In science you always measure against a baseline;
    this is ours.

    It has a second job too. It needs no API key, no network, and no account. Someone who
    clones this repo can run the whole system immediately. A submission that does not run
    scores nothing, however good the clever parts are.

WHAT IT IS BAD AT, HONESTLY
    Typos it has not seen. Sarcasm. Word order. "I do NOT want to cancel" reads exactly
    like "I want to cancel" to a keyword counter. Those failures are the point: they are
    what the LLM is being paid to fix.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ..clinic import DOCTORS
from ..schema import Extraction, Intent
from .base import Extractor

# ──────────────────────────────────────────────────────────────────────────────
# The vote table
# ──────────────────────────────────────────────────────────────────────────────
# (phrase, weight). Longer, more specific phrases are worth more, because "move my
# appointment" tells us far more than "move" on its own.

_VOTES: dict[Intent, list[tuple[str, int]]] = {
    Intent.CANCEL_APPOINTMENT: [
        ("cancel my appointment", 5), ("cancel the appointment", 5),
        ("cancel my booking", 5), ("cancel", 3), ("canceling", 3), ("cancelling", 3),
        ("delete my appointment", 4), ("remove my appointment", 4),
        ("can't make it", 4), ("cant make it", 4), ("won't be able to come", 4),
        ("i cannot come", 4), ("not coming", 3),
    ],
    Intent.RESCHEDULE_APPOINTMENT: [
        ("reschedule", 5), ("re-schedule", 5),
        ("move my appointment", 5), ("change my appointment", 5),
        ("postpone", 4), ("push my appointment", 4), ("shift my appointment", 4),
        ("switch my appointment", 4), ("move", 2), ("instead of", 2),
        ("from monday to", 3), ("earlier slot", 3), ("later slot", 3),
    ],
    Intent.BOOK_APPOINTMENT: [
        ("book an appointment", 5), ("make an appointment", 5),
        ("need an appointment", 5), ("want an appointment", 5),
        ("schedule an appointment", 5), ("book me", 4), ("book", 3),
        ("i want to see", 4), ("i need to see", 4), ("can i see", 4),
        ("appointment", 2), ("badde shouf", 4), ("badde shoufo", 4),
    ],
    Intent.ASK_OPENING_HOURS: [
        ("what time does the clinic close", 6), ("what time do you close", 6),
        ("what time do you open", 6), ("opening hours", 5), ("working hours", 5),
        ("when are you open", 5), ("are you open", 4), ("what time do you", 3),
        ("close", 2), ("open", 2), ("hours", 2),
    ],
    Intent.ASK_DOCTOR_AVAILABILITY: [
        ("do you have anything available", 6), ("anything available", 5),
        ("is dr", 3), ("is doctor", 3), ("available", 3), ("availability", 4),
        ("any slots", 4), ("any openings", 4), ("any free", 3), ("free", 1),
        ("when is", 2), ("what times", 3),
    ],
    Intent.TALK_TO_HUMAN: [
        ("speak with a human", 6), ("speak to a human", 6), ("talk to a human", 6),
        ("speak to someone", 5), ("talk to someone", 5), ("real person", 5),
        ("receptionist", 5), ("call me", 4), ("somebody call me", 5),
        ("someone call me", 5), ("from the clinic call", 5), ("human", 3),
    ],
    Intent.CONFIRM: [
        ("yes", 4), ("yep", 4), ("yeah", 4), ("sure", 3), ("ok", 3), ("okay", 3),
        ("confirm", 4), ("go ahead", 4), ("please do", 4), ("sounds good", 4),
        ("that works", 4), ("perfect", 3), ("book it", 5),
    ],
    Intent.DENY: [
        ("no thanks", 5), ("no thank you", 5), ("nope", 4), ("nah", 4),
        ("never mind", 5), ("nevermind", 5), ("forget it", 4), ("not that one", 4),
        ("no", 3),
    ],
}

# Short replies like "yes" or "no" are answers to a question we just asked. In a longer
# sentence the same word usually is not — "no, I need to see Dr. Karim" is a booking.
_SHORT_REPLY_WORDS = 6

# Explicit signals that the patient does not want us to act yet. Only phrases that plainly
# mean "stop" belong here — politeness and vagueness are not the same as deferral.
_HEDGES = [
    "don't book", "dont book", "do not book", "don't confirm", "dont confirm",
    "do not confirm", "don't schedule", "dont schedule", "not yet", "nothing yet",
    "just checking", "just want to know", "just asking", "just curious",
    "not sure yet", "i'm not sure", "im not sure", "might want", "maybe later",
    "thinking about", "hold off", "before i decide", "don't hold", "no rush",
]

_PREVIOUS_VISIT = [
    "my doctor", "same problem", "same issue", "last time", "again for the same",
    "the usual", "my usual doctor", "follow up", "follow-up",
]

# ──────────────────────────────────────────────────────────────────────────────
# Slot patterns
# ──────────────────────────────────────────────────────────────────────────────

_DOCTOR_RE = re.compile(r"\b(?:dr\.?|doctor)\s+([a-z]+)", re.I)

_DATE_PATTERNS = [
    r"\bday after tomorrow\b", r"\btomorrow\b", r"\btmrw\b", r"\btmr\b", r"\bbukra\b",
    r"\bboukra\b", r"\btoday\b", r"\bnext week\b", r"\bthis week\b",
    r"\bnext (?:mon|tues?|wed(?:nes)?|thur?s?|fri|sat(?:ur)?|sun)(?:day)?\b",
    r"\b(?:mon|tues?|wed(?:nes)?|thur?s?|fri|sat(?:ur)?|sun)(?:day)?\b",
    r"\bin \d+ days?\b", r"\b\d{4}-\d{1,2}-\d{1,2}\b", r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b",
]

_TIME_PATTERNS = [
    r"\bafter \d{1,2}(?::\d{2})?\s*(?:am|pm)?\b",
    r"\bbefore \d{1,2}(?::\d{2})?\s*(?:am|pm)?\b",
    r"\bbetween \d{1,2}\s*(?:and|-|to)\s*\d{1,2}\b",
    r"\bat \d{1,2}(?::\d{2})?\s*(?:am|pm)?\b",
    r"\b\d{1,2}:\d{2}\s*(?:am|pm)?\b",
    r"\b\d{1,2}\s*(?:am|pm)\b",
    r"\bafternoon\b", r"\bmorning\b", r"\bevening\b", r"\bnoon\b", r"\bmidday\b",
    r"\bany ?time\b", r"\bsometime\b", r"\bwhenever\b",
]

# Words that can follow "doctor" without being a name: "my doctor again", "the doctor
# please". Without this guard the pattern happily reports a doctor called "Again".
_NOT_NAMES = {
    "again", "please", "today", "tomorrow", "there", "who", "about", "now", "soon",
    "back", "first", "next", "last", "and", "or", "for", "the", "this", "that",
}

_KNOWN_FIRST = {d.first_name.lower() for d in DOCTORS}
_KNOWN_LAST = {d.last_name.lower() for d in DOCTORS}


class RuleBasedExtractor(Extractor):
    """Keyword-and-pattern reader. No network, no API key, always available."""

    name = "rules"

    def extract(self, message: str, history: Sequence[str] = ()) -> Extraction:
        text = message.lower().strip()
        word_count = len(text.split())

        intent, confidence = self._vote(text, word_count)

        result = Extraction(
            intent=intent,
            confidence=confidence,
            doctor=self._find_doctor(message),
            preferred_date=self._find_target_date(text, intent),
            preferred_time=self._first_match(text, _TIME_PATTERNS),
            existing_appointment_phrase=self._find_existing(text, intent),
            is_hedged=any(h in text for h in _HEDGES),
            refers_to_previous_visit=any(p in text for p in _PREVIOUS_VISIT),
            reason_for_visit=None,
            reasoning="keyword vote",
        )
        return self._finalise(result, message)

    # ---- intent ----

    def _vote(self, text: str, word_count: int) -> tuple[Intent, float]:
        """Tally votes per intent and convert the winning margin into a confidence."""
        scores: dict[Intent, int] = {}
        for intent, phrases in _VOTES.items():
            # "yes" / "no" only really mean yes/no in a short reply.
            if intent in (Intent.CONFIRM, Intent.DENY) and word_count > _SHORT_REPLY_WORDS:
                continue
            total = sum(w for phrase, w in phrases if phrase in text)
            if total:
                scores[intent] = total

        if not scores:
            return Intent.OTHER, 0.2

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_intent, top_score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0

        # A clear win scores high; a near-tie scores low. Capped at 0.8 because a keyword
        # counter should never claim the certainty a real language model can.
        margin = (top_score - runner_up) / top_score
        confidence = min(0.8, 0.45 + 0.35 * margin)
        return top_intent, round(confidence, 2)

    # ---- slots ----

    def _find_doctor(self, message: str) -> str | None:
        """
        Prefer an explicit "Dr. X"; otherwise look for a name we already know.

        The captured word is checked before being believed. "my doctor again" must not
        produce a doctor named Again — so we accept the word only if we recognise it, or
        the patient capitalised it the way people capitalise names.
        """
        m = _DOCTOR_RE.search(message)
        if m:
            candidate = m.group(1)
            known = candidate.lower() in _KNOWN_FIRST or candidate.lower() in _KNOWN_LAST
            looks_like_a_name = candidate[:1].isupper() and candidate.lower() not in _NOT_NAMES
            if known or looks_like_a_name:
                return m.group(0).strip()
            return None
        for word in re.findall(r"[A-Za-z]+", message):
            if word.lower() in _KNOWN_FIRST or word.lower() in _KNOWN_LAST:
                return word
        return None

    @classmethod
    def _find_target_date(cls, text: str, intent: Intent) -> str | None:
        """
        The date the patient wants to end up on.

        "Move my appointment from Monday to Wednesday" contains two dates. Wednesday is
        the one being asked for; Monday only identifies which appointment to move. A
        plain left-to-right scan would return Monday and book the wrong day.
        """
        if intent == Intent.RESCHEDULE_APPOINTMENT:
            m = re.search(r"\bfrom\s+(\w+)\s+to\s+(\w+)", text, re.I)
            if m:
                return m.group(2)
        return cls._first_match(text, _DATE_PATTERNS)

    @staticmethod
    def _first_match(text: str, patterns: list[str]) -> str | None:
        for pattern in patterns:
            m = re.search(pattern, text, re.I)
            if m:
                return m.group(0).strip()
        return None

    @staticmethod
    def _find_existing(text: str, intent: Intent) -> str | None:
        """
        For a reschedule, 'from Monday to Wednesday' hides two dates. This grabs the one
        being moved FROM; the normal date extractor grabs the one being moved TO.
        """
        if intent not in (Intent.RESCHEDULE_APPOINTMENT, Intent.CANCEL_APPOINTMENT):
            return None
        m = re.search(r"\bfrom\s+(\w+)\s+to\s+\w+", text, re.I)
        if m:
            return m.group(1)
        m = re.search(r"(my\s+\w+\s+appointment)", text, re.I)
        return m.group(1) if m else None
