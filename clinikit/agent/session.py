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
from .backends import get_extractor
from .clinic import TIMEZONE, ClinicDB, seeded_db
from .policy import Action, Context, Decision, Offer, WRITE_ACTIONS, decide
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
            "slots": [s.isoformat() for s in self.result.slots],
        }


class Session:
    """One ongoing conversation with one patient."""

    def __init__(
        self,
        backend: str = "rules",
        patient_id: str = "p_001",
        db: Optional[ClinicDB] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.extractor = get_extractor(backend)
        self.patient_id = patient_id
        self.db = db if db is not None else seeded_db(patient_id)

        # Passing the clock in keeps tests reproducible: pin it and the same conversation
        # gives the same answers every time, forever.
        self._clock = clock or (lambda: datetime.now(TIMEZONE))

        # --- everything below here is the memory ---
        self.known_slots: dict[str, str] = {}
        self.pending_offer: Optional[Offer] = None
        self.clarifications: int = 0
        self.history: list[str] = []
        self.turns: list[Turn] = []

    # ---- the main entry point ------------------------------------------------

    def handle(self, message: str) -> Turn:
        """Process one patient message and return everything that happened."""
        started = time.monotonic()

        ctx = self._snapshot()
        extraction = self.extractor.extract(message, self.history)
        decision = decide(extraction, ctx)
        result = tools.execute(decision, ctx)
        reply = responder.respond(decision, result, ctx)

        turn = Turn(
            message=message,
            extraction=extraction,
            decision=decision,
            result=result,
            reply=reply,
            seconds=time.monotonic() - started,
            backend=self.extractor.name,
            changed_the_book=(result.ok and decision.action in WRITE_ACTIONS),
        )
        self._remember(turn)
        return turn

    # ---- memory in ------------------------------------------------------------

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
            clarifications_so_far=self.clarifications,
        )

    # ---- memory out -----------------------------------------------------------

    def _remember(self, turn: Turn) -> None:
        """
        The only method in the project that changes what the agent remembers.

        If the agent ever remembers something wrong, the bug is in here.
        """
        self.turns.append(turn)
        self.history.append(turn.message)

        # Hold on to details the patient gave, so they do not have to repeat themselves.
        if turn.extraction.doctor:
            self.known_slots["doctor"] = turn.extraction.doctor
        if turn.extraction.preferred_date:
            self.known_slots["date"] = turn.extraction.preferred_date
        if turn.extraction.preferred_time:
            self.known_slots["time"] = turn.extraction.preferred_time

        # Count clarifying questions in a row. Enough of them and the policy layer
        # hands the conversation to a person instead of asking again.
        if turn.decision.action == Action.ASK_FOR_MORE_INFORMATION:
            self.clarifications += 1
        else:
            self.clarifications = 0

        # A finished booking ends the thread: clear the offer and the gathered details so
        # the next request starts clean.
        if turn.changed_the_book:
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
        self.clarifications = 0
        self.history.clear()
        self.turns.clear()

    def audit_log(self) -> list[dict]:
        """Every turn so far, in a form that could be written to a file or a database."""
        return [t.as_dict() for t in self.turns]

    @property
    def writes_so_far(self) -> int:
        return sum(1 for t in self.turns if t.changed_the_book)
