"""
The agent's memory.

THE SIMPLE VERSION
    Until now the agent has been forgetful. Each message was read on its own, with no
    idea what came before. That does not work for a conversation:

        Agent:   "Tuesday 11am is free. Shall I book it?"
        Patient: "yes"

    The word "yes" means nothing by itself. No doctor, no day, no time. It only makes
    sense if you remember what was just offered.

    This file remembers.

WHY IT IS ITS OWN FILE
    This is the ONLY file in the project that stores anything that changes between
    messages. Everything else reads what it is given and hands back an answer.

    That matters when something goes wrong. If the agent "remembers" the wrong thing,
    there is exactly one place that could have written it -- the `_remember` method
    below. Not ten places across five files.

HOW ONE MESSAGE FLOWS THROUGH
    1. Take a copy of what we currently remember       (_snapshot)
    2. Read the message                                 (the backend)
    3. Decide what is allowed                           (policy.decide)
    4. Do it                                            (tools.execute)
    5. Write the reply                                  (responder.respond)
    6. Update what we remember                          (_remember)

    Steps 2 to 5 never touch memory. They are handed a copy and give back an answer.
    Only step 6 writes anything down.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

from . import responder, tools
from .phrasing import Phraser, ReplyFacts
from .backends import CHAIN_ORDER, get_extractor
from .backends.base import ExtractorUnavailable
from .clinic import TIMEZONE, ClinicDB, seeded_db
from .policy import (
    WRITE_ACTIONS,
    Action,
    Context,
    Decision,
    Offer,
    PendingQuestion,
    decide,
)
from .schema import Extraction, Intent
from .tools import ToolResult


@dataclass
class Turn:
    """
    A complete record of one message and everything that happened because of it.

    Kept in full rather than just the reply, because this is what the inspector panel in
    the web interface shows, and what an audit log for a real clinic would need: what was
    said, what the system understood, what it decided, why, and what it did.
    """

    message: str
    extraction: Extraction
    decision: Decision
    result: ToolResult
    reply: str
    seconds: float
    backend: str
    changed_the_book: bool
    reply_source: str = "template"
    """'gemini' if the model's wording was used, 'template' if the hand-written one was."""

    reply_rejected_because: Optional[str] = None
    """Why the model's wording was thrown away, when it was. Empty if it was accepted."""

    degraded_reason: Optional[str] = None
    """Set when the chosen reader was unreachable and the keyword reader stood in."""

    def as_dict(self) -> dict:
        """Flat form, for the API and the audit log."""
        return {
            "message": self.message,
            "reply": self.reply,
            "backend": self.backend,
            "seconds": round(self.seconds, 3),
            "extraction": self.extraction.model_dump(mode="json"),
            "action": self.decision.action,
            "reason": self.decision.reason,
            "guarantee": self.decision.guarantee,
            "was_write": self.decision.action in WRITE_ACTIONS,
            "changed_the_book": self.changed_the_book,
            "detail": self.result.detail,
            "reply_source": self.reply_source,
            "reply_rejected_because": self.reply_rejected_because,
            "degraded_reason": self.degraded_reason,
            "slots": [
                {
                    "when": getattr(s, "start", s).isoformat(),
                    "label": f"{getattr(s, 'start', s):%A %d %B, %H:%M}",
                    "doctor": getattr(getattr(s, "doctor", None), "full_name", None),
                }
                for s in self.result.slots
            ],
        }


class Session:
    """One ongoing conversation with one patient."""

    def __init__(
        self,
        backend: str = "rules",
        patient_id: str = "p_001",
        db: Optional[ClinicDB] = None,
        clock: Optional[Callable[[], datetime]] = None,
        natural_replies: bool = False,
    ) -> None:
        self.extractor = get_extractor(backend)

        # When on, Gemini rewrites each reply in natural language and the result is
        # checked against the facts before the patient sees it. When the check fails, or
        # no model is available, the hand-written reply is sent instead -- so turning this
        # on can make replies nicer but can never make them wrong.
        self.natural_replies = natural_replies
        self.phraser = Phraser(self.extractor if natural_replies else None)

        # NOTE: there is deliberately no keyword-matching stand-in here.
        #
        # There used to be. When the language model was unreachable, the keyword reader
        # answered instead -- and it answers badly. "I need to see a doctor" came back as
        # "I couldn't find that doctor". Patients got nonsense during every outage.
        #
        # Now, if no provider can be reached, we say so and fetch a person. A clinic would
        # far rather hear "we are having trouble, someone will help you" than a confident
        # wrong answer about their appointment.
        self.patient_id = patient_id
        self.db = db if db is not None else seeded_db(patient_id)

        # Passing the clock in keeps tests reproducible: pin it and the same conversation
        # gives the same answers every time, forever.
        self._clock = clock or (lambda: datetime.now(TIMEZONE))

        # --- everything below here is the memory ---
        self.known_slots: dict[str, str] = {}
        self.pending_offer: Optional[Offer] = None
        self.pending_question: Optional[PendingQuestion] = None
        self.offered_options: tuple = ()
        self.last_touched_appointment_id: Optional[str] = None
        self.clarifications: int = 0
        self.history: list[str] = []
        self.turns: list[Turn] = []

    # ---- the main entry point ------------------------------------------------

    def handle(self, message: str) -> Turn:
        """Process one patient message and return everything that happened."""
        started = time.monotonic()

        ctx = self._snapshot()

        try:
            extraction = self.extractor.extract(message, self._reader_history())
        except ExtractorUnavailable as exc:
            # Nobody could read the message. Do not guess -- hand over to a person.
            turn = self._outage_turn(message, str(exc), time.monotonic() - started)
            self._remember(turn)
            return turn

        degraded = None
        used = getattr(self.extractor, "last_used", None)
        if used and used != CHAIN_ORDER[0] and self.extractor.name == "auto":
            degraded = f"first choice unavailable — answered by {used}"

        decision = decide(extraction, ctx)
        result = tools.execute(decision, ctx)
        structured = responder.respond(decision, result, ctx, extraction)

        changed = result.ok and decision.action in WRITE_ACTIONS
        reply = structured.as_text()
        reply_source, rejected = "template", None
        if self.natural_replies and self.phraser.available and degraded is None:
            phrased = self.phraser.rephrase(
                structured,
                _facts_for(decision, result, structured, changed, ctx),
                message,
            )
            reply, reply_source, rejected = phrased.text, phrased.source, phrased.rejected_because

        turn = Turn(
            message=message,
            extraction=extraction,
            decision=decision,
            result=result,
            reply=reply,
            seconds=time.monotonic() - started,
            backend=extraction.backend or self.extractor.name,
            changed_the_book=changed,
            reply_source=reply_source,
            reply_rejected_because=rejected,
            degraded_reason=degraded,
        )
        self._remember(turn)
        return turn

    # ---- memory in ------------------------------------------------------------

    def _reader_history(self) -> list[str]:
        """
        What the reader is told about the conversation so far.

        If an offer is waiting for a yes or no, that is stated outright rather than left
        for the model to infer from our last reply. Without it, "yes please book it" was
        read as a brand-new booking request, and the patient had to start again.
        """
        lines = list(self.history[-4:])

        if self.offered_options:
            shown = "; ".join(
                f"{i}. " + (
                    getattr(o, "full_name", None)
                    or (f"{o.start:%A %d %B at %H:%M}" if getattr(o, "start", None) else str(o))
                )
                for i, o in enumerate(self.offered_options[:6], start=1)
            )
            lines.append(f"clinic just showed this numbered list: {shown}")

        if self.pending_offer is not None:
            lines.append(
                f"clinic is waiting for a YES or NO on this exact offer: "
                f"{self.pending_offer.summary}"
            )
        elif self.pending_question is not None:
            lines.append(
                f"clinic is waiting for a YES or NO on this question: "
                f"{self.pending_question.summary}"
            )
        return lines

    def _outage_turn(self, message: str, problem: str, seconds: float) -> Turn:
        """
        What the patient sees when no AI provider can be reached.

        Honest, and safe: handing over to a person cannot book, move or cancel anything.
        """
        from .policy import Action, Decision

        decision = Decision(
            action=Action.HANDOFF_TO_HUMAN,
            reason=f"No message reader was available, so nothing was interpreted. {problem}",
        )
        return Turn(
            message=message,
            extraction=Extraction(
                intent=Intent.OTHER, confidence=0.0,
                reasoning="the message was never read — every provider was unavailable",
                raw_message=message, backend="none",
            ),
            decision=decision,
            result=ToolResult(ok=True, action=Action.HANDOFF_TO_HUMAN,
                              detail="Handed to a person: no reader available."),
            reply=("Sorry — I'm having trouble on my end right now. "
                   "I've passed this to the clinic team and someone will get back to you shortly."),
            seconds=seconds,
            backend="none",
            changed_the_book=False,
            degraded_reason=f"all providers unavailable — {problem[:110]}",
        )

    def _snapshot(self) -> Context:
        """
        Take a copy of what we remember, to hand to the decision-making code.

        `dict(...)` makes a copy rather than passing the original. The policy layer
        therefore cannot change our memory even by accident -- it is working from a copy.
        """
        return Context(
            now=self._clock(),
            db=self.db,
            patient_id=self.patient_id,
            known_slots=dict(self.known_slots),
            pending_offer=self.pending_offer,
            pending_question=self.pending_question,
            clarifications_so_far=self.clarifications,
            offered_options=self.offered_options,
            last_touched_appointment_id=self.last_touched_appointment_id,
        )

    # ---- memory out -----------------------------------------------------------

    def _remember(self, turn: Turn) -> None:
        """
        The only method in the project that changes what the agent remembers.

        If the agent ever remembers something wrong, the bug is in here.
        """
        self.turns.append(turn)

        # A yes/no question is only live for the turn immediately after we ask it.
        # Keeping it longer would mean a "yes" three messages later silently answers
        # something the patient has long since moved on from.
        self.pending_question = turn.decision.question

        # Remember the numbered choices we just showed, so "the first one" resolves next
        # turn. Only replaced when we actually showed something -- otherwise a follow-up
        # question would wipe out the list the patient is still looking at.
        if turn.decision.candidates:
            self.offered_options = tuple(turn.decision.candidates)

        # Record BOTH sides. The reader used to see only the patient's own messages, so
        # it had no idea a question had just been asked -- and read "yes please book it"
        # as a brand-new booking request rather than an answer. A conversation the model
        # can only half-see is a conversation it will half-understand.
        self.history.append(f"patient: {turn.message}")
        self.history.append(f"clinic: {' '.join(turn.reply.split())[:200]}")

        # Forget earlier details once a request is over.
        #
        # Details are remembered so a patient does not have to repeat themselves while we
        # are filling in the gaps of one request. They should NOT leak into the next one.
        # Without this, someone who asked about Dr. George and then said "I need to see a
        # doctor" was quietly assumed to still mean Dr. George.
        #
        # We are mid-request only while we are asking a question, or while an offer is
        # waiting for a yes. Anything else ends the thread.
        still_gathering = (
            turn.decision.action == Action.ASK_FOR_MORE_INFORMATION
            or turn.decision.offer is not None
            or self.pending_offer is not None
        )
        if not still_gathering:
            self.known_slots.clear()

        # Hold on to details the patient gave, so they do not have to repeat themselves.
        if turn.extraction.doctor:
            self.known_slots["doctor"] = turn.extraction.doctor
        if turn.extraction.preferred_date:
            self.known_slots["date"] = turn.extraction.preferred_date
        if turn.extraction.preferred_time:
            self.known_slots["time"] = turn.extraction.preferred_time

        # Count only questions that mean "I did not understand you" -- not the ordinary
        # back-and-forth of gathering details.
        #
        # This used to count every question, so a normal conversation ("which doctor?",
        # "which day?", "what time?") hit the limit and the agent handed the patient to a
        # person on the fourth message. Asking which doctor is the agent working, not the
        # agent failing.
        confused = (
            turn.decision.guarantee == "G6"
            or "clarification" in turn.decision.missing
            or "what_they_need" in turn.decision.missing
        )
        self.clarifications = self.clarifications + 1 if confused else 0

        # A finished booking ends the thread: clear the offer and the gathered details so
        # the next request starts clean.
        if turn.changed_the_book:
            if turn.result.appointment is not None:
                self.last_touched_appointment_id = turn.result.appointment.id
            self.pending_offer = None
            self.known_slots.clear()
            return

        # The patient said no -- drop the offer rather than leaving it lying around for a
        # later "yes" to pick up by accident.
        if turn.extraction.intent == Intent.DENY:
            self.pending_offer = None
            return

        # Otherwise carry forward whatever the policy layer offered this turn. If it
        # offered nothing, keep the previous offer alive so the patient can still say yes.
        if turn.decision.offer is not None:
            self.pending_offer = turn.decision.offer

    # ---- convenience ----------------------------------------------------------

    def reset(self) -> None:
        """Start a fresh conversation, keeping the same appointment book."""
        self.known_slots.clear()
        self.pending_offer = None
        self.pending_question = None
        self.offered_options = ()
        self.clarifications = 0
        self.last_touched_appointment_id = None
        self.history.clear()
        self.turns.clear()

    def audit_log(self) -> list[dict]:
        """Every turn so far, in a form that could be written to a file or a database."""
        return [t.as_dict() for t in self.turns]

    @property
    def writes_so_far(self) -> int:
        return sum(1 for t in self.turns if t.changed_the_book)


def _facts_for(decision, result, structured, changed: bool, ctx) -> ReplyFacts:
    """
    Collect exactly what the reply is allowed to mention.

    Anything not gathered here will be rejected if it appears in the model's wording, so
    this list is the boundary of what the patient can be told.
    """
    doctors: list[str] = []
    if decision.doctor is not None:
        doctors.append(decision.doctor.full_name)
    for c in decision.candidates:
        full = getattr(c, "full_name", None)
        if full:
            doctors.append(full)

    times: list[datetime] = []
    for s in result.slots:
        start = getattr(s, "start", s)
        times.append(start)
        slot_doctor = getattr(s, "doctor", None)
        if slot_doctor is not None:
            doctors.append(slot_doctor.full_name)
    if decision.slot is not None:
        times.append(decision.slot)
    if result.appointment is not None:
        times.append(result.appointment.start)
        doctor = ctx.db.doctor(result.appointment.doctor_id)
        if doctor is not None:
            doctors.append(doctor.full_name)
    for c in decision.candidates:
        start = getattr(c, "start", None)
        if isinstance(start, datetime):
            times.append(start)

    # The appointment or offer being ASKED about has not been executed yet, so it is not
    # in result.appointment -- but its time is right there in the reply we are about to
    # reword. Leaving it out made the checker reject a perfectly correct confirmation
    # question for "mentioning a time we did not offer".
    if decision.appointment is not None:
        times.append(decision.appointment.start)
        doctor = ctx.db.doctor(decision.appointment.doctor_id)
        if doctor is not None:
            doctors.append(doctor.full_name)
    if decision.offer is not None and decision.offer.start is not None:
        times.append(decision.offer.start)

    return ReplyFacts(
        template=structured.as_text(),
        action=decision.action,
        changed_the_book=changed,
        doctors=tuple(dict.fromkeys(doctors)),
        times=tuple(dict.fromkeys(times)),
        reference=result.appointment.id if result.appointment else None,
        # The policy layer already wrote a plain-language account of what it decided and
        # why. That is exactly the situation brief the model needs -- no second
        # description to write and keep in step with the first.
        situation=decision.reason,
        goal=structured.goal,
        needs=tuple(decision.missing),
        has_list=bool(structured.body),
    )
