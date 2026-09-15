"""
Turning a decision into something a patient reads.

THE SIMPLE VERSION
    By now the system knows what it is doing and why. This file writes it down as English.

REPLIES COME IN PARTS, AND THAT IS THE WHOLE POINT
    A reply used to be one block of text. That forced an impossible choice: either let the
    model reword the whole thing -- and watch it flatten a tidy list of appointment times
    into a run-on sentence -- or let it reword nothing, and sound like a form.

    So a reply is now four pieces:

        acknowledgement   "Sorry to hear that."                    prose
        opening           "Dr. Karim Nassar is our cardiologist."   prose
        body              1. Wednesday 16 September, 11:00          DATA
                          2. Wednesday 16 September, 11:30
        closing           "Would either of those suit?"             prose

    The model may reword the prose. It never sees the body, so it cannot damage it. The
    patient gets human wording wrapped around facts that are exactly what the clinic's
    own records say.

WHY THE FACTS ARE NEVER LEFT TO THE MODEL
    A model asked to be helpful will sometimes write "I've booked you in for Tuesday" when
    nothing was booked. It is being agreeable. In a clinic that means somebody stays home
    believing they have an appointment. The body is built here, from the appointment book,
    and handed over untouched.

THE OPTIONS ARE NUMBERED ON PURPOSE
    Bullets cannot be referred to. Numbers can: "the first one", "option 2". The numbering
    is what makes that phrase answerable -- see _resolve_option in policy.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .clinic import (
    CLINIC_ADDRESS,
    CLINIC_NAME,
    CLINIC_PHONE,
    DOCTORS,
    Appointment,
    Doctor,
    describe_opening_hours,
)
from .policy import Action, Decision, FreeSlot
from .tools import ToolResult


@dataclass(frozen=True)
class Reply:
    """One reply, split into the parts a model may touch and the part it may not."""

    acknowledgement: str = ""
    """A short human response to what they said. "Sorry to hear that." Prose."""

    opening: str = ""
    """The useful sentence. "Dr. Karim Nassar is our cardiologist." Prose."""

    body: tuple[str, ...] = ()
    """Numbered facts from the clinic's records. NEVER reworded, never paraphrased."""

    closing: str = ""
    """The question that moves things along. "Would either of those suit?" Prose."""

    def as_text(self) -> str:
        """Assemble the pieces into the message the patient sees."""
        blocks: list[str] = []
        lead = " ".join(p for p in (self.acknowledgement, self.opening) if p).strip()
        if lead:
            blocks.append(lead)
        if self.body:
            blocks.append("\n".join(self.body))
        if self.closing:
            blocks.append(self.closing)
        return "\n\n".join(blocks)


# ──────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ──────────────────────────────────────────────────────────────────────────────

def _when(moment: datetime) -> str:
    return f"{moment:%A %-d %B at %-I:%M %p}".replace("AM", "am").replace("PM", "pm")


def _numbered(lines: list[str]) -> tuple[str, ...]:
    return tuple(f"{i}. {line}" for i, line in enumerate(lines, start=1))


def _slot_lines(slots, limit: int = 4, show_doctor: bool = False) -> tuple[str, ...]:
    out = []
    for s in list(slots)[:limit]:
        if isinstance(s, FreeSlot):
            out.append(f"{_when(s.start)} — {s.doctor.full_name}" if show_doctor
                       else _when(s.start))
        else:
            out.append(_when(s))
    return _numbered(out)


def _doctor_lines(doctors, with_hours: bool = True) -> tuple[str, ...]:
    out = []
    for d in doctors:
        out.append(f"{d.full_name} — {d.specialty}, {d.hours_summary}" if with_hours
                   else f"{d.full_name} — {d.specialty}")
    return _numbered(out)


def _acknowledge(extraction) -> str:
    """
    A short human response to a stated problem.

    Only fires when the patient actually described something. Saying "sorry to hear that"
    to somebody asking about parking would be strange.
    """
    if extraction is None or not getattr(extraction, "reason_for_visit", None):
        return ""
    return "Sorry to hear that."


# ──────────────────────────────────────────────────────────────────────────────
# The reply
# ──────────────────────────────────────────────────────────────────────────────

def respond(decision: Decision, result: ToolResult, ctx, extraction=None) -> Reply:
    """Build the reply for one turn."""

    # --- a person takes over -------------------------------------------------
    if decision.action == Action.HANDOFF_TO_HUMAN:
        if decision.topic and decision.topic not in ("greeting",):
            return Reply(
                opening=f"That one is better answered by our team — I'm passing on your "
                        f"question about {decision.topic}.",
                closing="Someone will come back to you shortly.",
            )
        return Reply(
            opening="Of course — I'm passing you to a member of our team now.",
            closing="Someone will be with you shortly.",
        )

    # --- information ---------------------------------------------------------
    if decision.action == Action.PROVIDE_INFORMATION:
        return _inform(decision, extraction)

    # --- something actually happened ----------------------------------------
    if decision.action == Action.CREATE_APPOINTMENT:
        if not result.ok:
            return Reply(opening=f"I couldn't complete that booking: {result.detail.lower()}",
                         closing="Shall we try another time?")
        appt = result.appointment
        doctor = ctx.db.doctor(appt.doctor_id)
        return Reply(
            opening=f"That's booked — {doctor.full_name}, {_when(appt.start)}.",
            closing=f"Your reference is {appt.id}.",
        )

    if decision.action == Action.RESCHEDULE_APPOINTMENT:
        if not result.ok:
            return Reply(opening=f"I couldn't move that appointment: {result.detail.lower()}",
                         closing="Would another time work?")
        appt = result.appointment
        doctor = ctx.db.doctor(appt.doctor_id)
        return Reply(opening=f"Moved — you're now seeing {doctor.full_name} on "
                             f"{_when(appt.start)}.")

    if decision.action == Action.CANCEL_APPOINTMENT:
        if not result.ok:
            return Reply(opening=f"I couldn't cancel that: {result.detail.lower()}")
        appt = result.appointment
        doctor = ctx.db.doctor(appt.doctor_id)
        return Reply(
            opening=f"Cancelled — your appointment with {doctor.full_name} on "
                    f"{_when(appt.start)} has been removed.",
            closing="Let me know if you'd like to rebook.",
        )

    # --- showing what is free ------------------------------------------------
    if decision.action == Action.CHECK_AVAILABILITY:
        return _availability(decision, result, extraction)

    # --- asking a question ---------------------------------------------------
    if decision.action == Action.ASK_FOR_MORE_INFORMATION:
        return _ask(decision, ctx, extraction)

    return Reply(opening="Sorry, I didn't quite follow that.",
                 closing="Could you put it another way?")


def _inform(decision: Decision, extraction) -> Reply:
    topic = decision.topic or ""

    if topic == "greeting":
        return Reply(
            opening=f"Hello — you've reached {CLINIC_NAME}.",
            closing="How can I help today?",
        )

    if topic == "opening_hours":
        return Reply(
            opening="Here are our opening hours.",
            body=tuple(describe_opening_hours().splitlines()),
            closing="Would you like to book something?",
        )

    if topic == "which_doctor" and decision.candidates:
        # Naming a department, and only that. Not a word about what the symptom means.
        return Reply(
            acknowledgement=_acknowledge(extraction),
            opening=f"That's handled by our {decision.specialty.lower()} team.",
            body=_doctor_lines(decision.candidates),
            closing="Would you like me to check when they're free?",
        )

    # Anything else factual we hold about the clinic.
    return Reply(
        opening=f"{CLINIC_NAME} is on {CLINIC_ADDRESS}. You can reach us on {CLINIC_PHONE}.",
        body=tuple(describe_opening_hours().splitlines()),
        closing="Anything else I can help with?",
    )


def _availability(decision: Decision, result: ToolResult, extraction) -> Reply:
    who = decision.doctor.full_name if decision.doctor else None
    show_doctor = decision.doctor is None

    if not result.slots:
        return Reply(
            acknowledgement=_acknowledge(extraction),
            opening=(f"I can't see anything free for {who} then." if who
                     else "I can't see anything free around then."),
            closing="Would a different day work?",
        )

    # The patient said not to act yet. Say so plainly so they know we listened.
    if decision.guarantee == "G1":
        opening = "Understood — I won't book anything yet."
    elif decision.guarantee in ("G4", "G5"):
        opening = decision.reason
    elif who:
        opening = f"{who} has these free:"
    else:
        opening = "Here's what's free soon:"

    return Reply(
        acknowledgement=_acknowledge(extraction),
        opening=opening,
        body=_slot_lines(result.slots, show_doctor=show_doctor),
        closing="Would any of those suit?",
    )


def _ask(decision: Decision, ctx, extraction) -> Reply:
    missing = decision.missing

    # --- confirming before anything is written ------------------------------
    if "confirmation" in missing and decision.offer:
        offer = decision.offer
        if offer.kind == "create":
            return Reply(opening=f"{offer.summary}.", closing="Shall I lock that in?")
        if offer.kind == "cancel":
            return Reply(opening=f"Just to check — you'd like me to {offer.summary}?",
                         closing="Say yes and I'll take care of it.")
        if offer.kind == "reschedule":
            return Reply(opening=f"I can {offer.summary}.", closing="Shall I go ahead?")

    # --- which doctor --------------------------------------------------------
    if "doctor" in missing:
        if decision.problem == "doctor_suggested" and decision.candidates:
            return Reply(
                acknowledgement=_acknowledge(extraction),
                opening=f"That's handled by our {decision.specialty.lower()} team.",
                body=_doctor_lines(decision.candidates),
                closing="Shall I check when they're free?",
            )
        if decision.problem == "doctor_ambiguous" and decision.candidates:
            return Reply(
                opening="We have more than one doctor by that name.",
                body=_doctor_lines(decision.candidates, with_hours=False),
                closing="Which one did you mean?",
            )
        if decision.problem == "doctor_unknown":
            return Reply(
                opening="I couldn't find a doctor by that name. Here's the team.",
                body=_doctor_lines(DOCTORS, with_hours=False),
                closing="Who would you like to see?",
            )
        if decision.problem == "doctor_previous":
            return Reply(
                opening="I can't see your previous notes from here, so I'm not sure who "
                        "you saw. Here's the team.",
                body=_doctor_lines(DOCTORS, with_hours=False),
                closing="Who would you like to see? I can also pass you to the clinic if "
                        "you'd rather someone checked your records.",
            )
        return Reply(
            acknowledgement=_acknowledge(extraction),
            opening="Here's the team.",
            body=_doctor_lines(DOCTORS, with_hours=False),
            closing="Who would you like to see?",
        )

    # --- which appointment ---------------------------------------------------
    if "which_appointment" in missing:
        appointments = [a for a in decision.candidates if isinstance(a, Appointment)]
        if appointments:
            lines = [f"{ctx.db.doctor(a.doctor_id).full_name} — {_when(a.start)}"
                     for a in appointments]
            return Reply(opening="You have these coming up.",
                         body=_numbered(lines),
                         closing="Which one did you mean?")
        return Reply(opening="I don't have an upcoming appointment on file for you.",
                     closing="Would you like to book one?")

    # --- which of the options we listed --------------------------------------
    if "which_option" in missing:
        return Reply(opening="Sorry, I'm not sure which one you meant.",
                     closing="Could you give me the number, or the time?")

    # --- when ----------------------------------------------------------------
    if "new_date" in missing:
        return Reply(opening="Happy to move that.", closing="When would suit you instead?")
    if "date" in missing:
        return Reply(opening=decision.reason, closing="What day would suit you instead?")
    if "time" in missing:
        return Reply(opening=decision.reason,
                     closing="Could you give me the time a bit more precisely — 9am or 4pm, say?")

    # --- confirmation with nothing pending -----------------------------------
    if "what_to_confirm" in missing:
        return Reply(opening="I don't have anything waiting for a yes at the moment.",
                     closing="What would you like to do?")
    if "what_they_would_prefer" in missing:
        return Reply(opening="No problem, I won't book that.",
                     closing="What would work better for you?")

    return Reply(closing="Could you tell me a little more about what you need?")
