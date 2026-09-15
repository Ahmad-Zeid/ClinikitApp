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


# Words that only make sense if something really was written to the appointment book.
# If nothing changed, none of these may appear in the reply.
_COMPLETION_CLAIMS = [
    "i've booked", "i have booked", "you're booked", "you are booked",
    "has been booked", "is booked", "booking is confirmed", "i've confirmed",
    "i have confirmed", "has been cancelled", "has been canceled", "i've cancelled",
    "i have cancelled", "has been moved", "i've moved", "i have moved",
    "has been rescheduled", "appointment is set", "all set for",
    # Added after the model wrote "Your appointment with Dr. Karim Nassar is scheduled
    # for Wednesday" on a turn that was only ASKING for confirmation. Nothing had been
    # written to the appointment book. The patient would have believed they had an
    # appointment they did not have -- exactly the failure this check exists to prevent,
    # slipping through because the wording was not on the list.
    # Kept deliberately narrow. "Would you like me to cancel your appointment with Dr.
    # Karim?" is a perfectly good question, so phrases like "your appointment with" must
    # NOT be banned -- over-blocking sends us back to stiff templates, which is the thing
    # this whole layer exists to fix. Only wordings that assert a completed action.
    "is scheduled for", "is set for", "is confirmed", "is reserved",
    "we have you down", "we've got you down", "you're all set", "you are all set",
]

# Promises about things this system cannot do at all.
# Arabic script and common Arabizi markers. The clinic is English-only for now, and a
# reply in the wrong language is a bad reply however accurate it is.
_NON_ENGLISH_MARKERS = ["3a", "7a", "2a", "5a", "9a", "badd", "shou", "kif", "bukra",
                        "ma3", "3ando", "fadi", "el "]

# Wordings that assess a patient's condition rather than pointing at a department.
#
# The system routes: "Dr. Karim Nassar is our cardiologist." It must never assess: what
# the symptom might mean, how serious it is, or how quickly it needs attention. Those are
# clinical judgements and nothing here is qualified to make them. A reply that drifts
# into one is thrown away and the approved wording is sent instead.
#
# Note what is NOT banned: "you should see our cardiologist" is directory information in
# a receptionist's mouth, and blocking it would cost naturalness for no safety gain.
_CLINICAL_ADVICE = [
    "sounds like", "could be a sign", "may have", "might have", "probably have",
    "you likely", "this is likely", "diagnos", "symptoms suggest", "indicates",
    "seek immediate", "seek urgent", "go to a&e", "go to the er", "emergency room",
    "this is serious", "this is urgent", "right away", "as soon as possible you should",
    "i recommend you", "i'd recommend you", "you need to be seen",
]

_IMPOSSIBLE_CLAIMS = [
    # Phrases, not bare words. Banning "insurance" outright meant we could not even say
    # "I'm passing on your insurance question" -- the reply was rejected for mentioning
    # the topic it was about. What must be blocked is PROMISING to do something the
    # clinic cannot do, not naming the subject.
    "sent you a reminder", "sent a reminder", "i'll send you a reminder",
    "we'll send you a reminder", "sent you a confirmation", "i'll email",
    "we'll email", "i will email", "we will email", "i'll text", "we'll text",
    "send you an email", "send you a text", "send you an sms",
    "i'll call you", "we'll call you", "i will call you", "we will call you",
    "added to your calendar", "add it to your calendar",
    "we'll invoice", "your insurance covers", "covered by your insurance",
    "we accept your insurance", "we take your insurance",
]

# Ways of saying "we will not act yet".
#
# When the patient wrote "don't book anything yet", the approved reply opens with
# "Understood -- I won't book anything yet." That one sentence is the whole point of the
# turn: it tells the patient we heard the instruction. A reworded reply that quietly
# drops it looks helpful and breaks the promise.
#
# So on those turns the promise is not optional. If the model's wording does not contain
# one of these, the wording is thrown away and the hand-written version is sent.
_NO_ACTION_PROMISES = [
    "won't book", "will not book", "wont book", "not book anything",
    "won't confirm", "will not confirm", "wont confirm", "not confirm anything",
    "won't reserve", "will not reserve", "not reserve anything",
    "nothing is booked", "nothing has been booked", "nothing booked",
    "haven't booked", "have not booked", "hold off", "holding off",
    "leave it unbooked", "without booking",
]

# Statements about what is (or is not) in the patient's records.
#
# These are claims about the clinic's own files, and the model has no way to check them.
# On one turn it wrote "we do not have an active appointment on our schedule for you" to
# a patient who did have one. The patient could have acted on that.
#
# The rule: a claim like this may appear only if OUR OWN approved reply made it. We are
# not asking the model to be right, only to not add records claims we did not make.
_RECORD_CLAIMS = [
    "do not have an active appointment", "don't have an active appointment",
    "do not have an appointment", "don't have an appointment",
    "no appointment on", "no active appointment", "no upcoming appointment",
    "you have no appointment", "you don't have any appointment",
    "you do not have any appointment", "nothing on our schedule",
    "nothing in our system", "not on our schedule", "no record of",
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
    situation: str = ""
    """What just happened, in plain words. Comes from the policy layer's own reason."""

    goal: str = ""
    """What this reply needs to achieve. Comes from the responder's branch."""

    needs: tuple[str, ...] = ()
    """What we still need from the patient, if anything."""

    has_list: bool = False
    """True when a numbered list sits between the lead and the closing, so the model can
    be told not to repeat or summarise something it cannot see."""

    must_promise_no_action: bool = False
    """True on a hedged turn (G1). The reply MUST still say we are not booking anything.
    Without this the reworded version can drop the promise and sound like a sales push."""

    extra_notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PhrasedReply:
    text: str
    source: str
    """'gemini' if the model's wording was used, otherwise 'template'."""

    rejected_because: str | None = None


def _parse(raw: str) -> tuple[str, str]:
    """
    Pull the two rewritten parts out of the model's answer.

    Expected shape is "LEAD: ...\\nCLOSING: ...". Models sometimes drop a label or add a
    stray quote, so anything unlabelled before a CLOSING line is treated as the lead --
    a slightly wrong split is still usable, whereas failing outright is not.
    """
    lead_lines: list[str] = []
    closing_lines: list[str] = []
    target = lead_lines

    for line in (raw or "").splitlines():
        stripped = line.strip().strip('"')
        upper = stripped.upper()
        if upper.startswith("LEAD:"):
            target = lead_lines
            stripped = stripped[5:].strip()
        elif upper.startswith("CLOSING:"):
            target = closing_lines
            stripped = stripped[8:].strip()
        if stripped and stripped.lower() != "(blank)":
            target.append(stripped)

    return " ".join(lead_lines).strip(), " ".join(closing_lines).strip()


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


# Words that flip a claim into its opposite.
#
# "Nothing is booked" contains "is booked", which is on the completion list -- and it is
# the exact sentence we most want to be able to send. Without this, the reassurance that
# nothing happened was rejected for claiming that something had.
_NEGATORS = ("nothing", "not", "no ", "never", "n't", "without", "isn", "wasn", "hasn")


def _is_negated(low: str, start: int) -> bool:
    """Is the claim just before position `start` cancelled by a negative word?"""
    window = low[max(0, start - 24):start]
    return any(word in window for word in _NEGATORS)


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
            start = low.find(claim)
            if start != -1 and not _is_negated(low, start):
                return f"claims completion ({claim!r}) but nothing was written"

    # 2. On a hedged turn, is the promise still there?
    #    "don't book anything yet" must be answered with "I won't". A reply that drops
    #    it and goes straight to "here are the times, shall I secure one?" reads as a
    #    push, which is the opposite of what the patient asked for.
    if facts.must_promise_no_action:
        if not any(promise in low for promise in _NO_ACTION_PROMISES):
            return "hedged turn, but the reply never promises not to book"

    # 3. Does it make a claim about the patient's records that we did not make?
    template_low = facts.template.lower()
    for claim in _RECORD_CLAIMS:
        if claim in low and claim not in template_low:
            return f"claims something about the patient's records we did not say ({claim!r})"

    # 4. Is it actually in English?
    if any(ch for ch in candidate if "\u0600" <= ch <= "\u06ff"):
        return "reply contains Arabic script"
    words = set(low.replace(",", " ").replace(".", " ").split())
    hits = [m for m in _NON_ENGLISH_MARKERS if m.strip() in words or
            any(w.startswith(m) for w in words if len(m) > 2)]
    if len(hits) >= 2:
        return f"reply looks like Arabizi rather than English ({', '.join(hits[:3])})"

    # 5. Has it strayed into assessing the patient rather than routing them?
    for phrase in _CLINICAL_ADVICE:
        if phrase in low:
            return f"strays into clinical advice ({phrase!r})"

    # 6. Does it promise something this system cannot do?
    for claim in _IMPOSSIBLE_CLAIMS:
        if claim in low:
            return f"promises something the system cannot do ({claim!r})"

    # 7. Does every time it mentions come from the facts we supplied?
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

    # 8. Does every doctor it names come from the facts we supplied?
    permitted = {d.lower() for name in facts.doctors for d in name.replace("Dr.", "").split()}
    for surname in _DOCTOR_RE.findall(candidate):
        if facts.doctors and surname.lower() not in permitted:
            return f"names a doctor we did not mention (Dr. {surname})"

    # 9. Sanity: a reply that is far longer than the template is probably padded out.
    if len(candidate) > max(400, len(facts.template) * 3):
        return "much longer than the approved reply; likely padded with invented detail"

    return None


SYSTEM_PROMPT = """\
You are the receptionist at a medical clinic in Beirut, writing back to a patient.

You will be given the SITUATION, the GOAL for this reply, and the FACTS you may use.
Write the reply yourself. A safe version is included -- use it as a guide to what is
true, not as a sentence to paraphrase.

Your reply has two parts:
  LEAD     what comes before any list
  CLOSING  what comes after it

A numbered list of times or doctors may sit between them. You will not see it and must
not reproduce, summarise, or count it.

TONE
- Warm but professional. A real receptionist, not a chatbot.
- Two or three short sentences across the whole reply. Often one is enough.
- If they mentioned a symptom or a problem, acknowledge it briefly first.
- Answer what they actually asked. If they asked a yes/no question, answer it.
- No emoji. No exclamation marks. Never "Absolutely", "I'd be delighted", "Great news".
- Greet them only if the SITUATION says they greeted you.
- Write times the way a receptionist says them out loud: "4:00 pm", never "16:00". The
  FACTS give you times in 24-hour form; convert them.

HARD RULES
- Use ONLY the facts given. Never invent a time, date, doctor or detail.
- Never say anything was booked, moved or cancelled unless the FACTS say it was.
- Never promise reminders, emails, texts, calls, or calendar invites.
- Never say what a symptom might mean, how serious it is, or how urgent. You are a
  receptionist, not a clinician. Name the department and stop.
- Always write in English.

Reply in exactly this format and nothing else:
LEAD: <your lead, or blank>
CLOSING: <your closing, or blank>"""



class Phraser:
    """Writes the patient-facing wording, and checks it before it is sent."""

    def __init__(self, extractor=None) -> None:
        """`extractor` is any provider backend -- we reuse its client, throttle and retries."""
        self._extractor = extractor
        self.rejections: list[str] = []

    @property
    def available(self) -> bool:
        return self._extractor is not None and hasattr(self._extractor, "write_text")

    def rephrase(self, reply, facts: ReplyFacts, patient_message: str = "") -> PhrasedReply:
        """
        Write the prose around a reply, leaving the list of facts untouched.

        The model is given the situation and the goal and composes its own sentences. It
        used to be handed our sentence and asked to reword it, which could not recover
        from a bad one -- "No problem, I won't book that" was reworded faithfully to a
        patient who was looking at a list nobody was booking.

        The model never sees the body. That is what makes this safe to do on every reply,
        including the ones with a list, which is most of them.
        """
        if not self.available:
            return PhrasedReply(reply.as_text(), "template", "no language model available")

        lead = " ".join(p for p in (reply.acknowledgement, reply.opening) if p).strip()
        if not lead and not reply.closing:
            return PhrasedReply(reply.as_text(), "template", "nothing to reword")

        try:
            raw = self._extractor.write_text(
                SYSTEM_PROMPT, self._prompt(lead, reply.closing, facts, patient_message)
            )
        except Exception as exc:  # noqa: BLE001
            return PhrasedReply(reply.as_text(), "template",
                                f"model unavailable: {type(exc).__name__}")

        new_lead, new_closing = _parse(raw)
        if not new_lead and not new_closing:
            return PhrasedReply(reply.as_text(), "template", "model returned nothing usable")

        # Check only the prose. The body is ours and was never sent.
        prose = " ".join(p for p in (new_lead, new_closing) if p)
        problem = verify(prose, facts)
        if problem:
            self.rejections.append(problem)
            return PhrasedReply(reply.as_text(), "template", problem)

        blocks = []
        if new_lead:
            blocks.append(new_lead)
        if reply.body:
            blocks.append("\n".join(reply.body))
        if new_closing:
            blocks.append(new_closing)

        return PhrasedReply("\n\n".join(blocks), getattr(self._extractor, "name", "model"))

    @staticmethod
    def _prompt(lead: str, closing: str, facts: ReplyFacts, patient_message: str) -> str:
        """
        The situation brief.

        Deliberately not "here is a sentence, rewrite it". That was the old approach and
        it could not recover from a bad sentence: when the template said "I won't book
        that" about some times nobody was booking, the model reworded the nonsense
        faithfully. Given the situation instead, it can write something that makes sense.
        """
        lines = []
        if patient_message:
            lines.append(f'The patient wrote: "{patient_message}"')
        if facts.situation:
            lines.append(f"SITUATION: {facts.situation}")
        if facts.goal:
            lines.append(f"GOAL: {facts.goal}")

        lines.append("")
        lines.append("FACTS you may use:")
        if facts.changed_the_book:
            lines.append("- The appointment book WAS changed. You may say it is done.")
        else:
            lines.append("- NOTHING was booked, moved or cancelled. The appointment book "
                         "is unchanged. Do not imply otherwise.")
        if facts.doctors:
            lines.append(f"- Doctors involved: {', '.join(facts.doctors)}")
        if facts.times:
            lines.append("- Times involved: " +
                         ", ".join(f"{t:%A %d %B %H:%M}" for t in facts.times))
        if facts.reference:
            lines.append(f"- Booking reference: {facts.reference}")
        if facts.needs:
            lines.append(f"- Still needed from them: {', '.join(facts.needs)}")
        if facts.has_list:
            lines.append("- A numbered list sits between LEAD and CLOSING. Do not repeat "
                         "it, summarise it, or say how many items it has.")
        if facts.must_promise_no_action:
            # Stated as a requirement rather than a preference, because verify() will
            # reject the reply if it is missing and the patient will get the stiff one.
            lines.append("- REQUIRED: the patient asked us NOT to act yet. Your reply "
                         "must say plainly that you will not book anything. Say it "
                         "first, and do not follow it with 'but'. You may still show "
                         "the times and invite them to come back to you.")

        lines.append("")
        lines.append(f"A safe version (true, but stiff):\n  LEAD: {lead or '(blank)'}"
                     f"\n  CLOSING: {closing or '(blank)'}")
        lines.append("")
        lines.append("Now write your own LEAD and CLOSING.")
        return "\n".join(lines)
