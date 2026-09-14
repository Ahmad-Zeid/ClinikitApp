"""
Turning a decision into a sentence the patient reads.

THE SIMPLE VERSION
    By this point the system knows what it is doing and why. This file writes that down
    as normal English.

WHY THE REPLIES ARE WRITTEN BY HAND, NOT BY THE AI
    It would be easy to ask Gemini to write these replies. We do not, on purpose.

    A language model asked to be helpful will sometimes say "I've booked you in for
    Tuesday" when nothing was booked. It is trying to be pleasant. For a clinic that is
    a serious problem: the patient stays home on Tuesday believing they have an
    appointment.

    Because these sentences are fixed templates, the words the patient reads always match
    what actually happened in the appointment book. The reply cannot promise something
    the system did not do.

    The trade-off is honest: the replies sound slightly more formal than a chatbot's. For
    a medical clinic that is the right way round.
"""

from __future__ import annotations

from datetime import datetime

from .clinic import Appointment, describe_opening_hours
from .policy import Action, Decision, FreeSlot
from .tools import ToolResult


def _when(moment: datetime) -> str:
    return f"{moment:%A %-d %B at %-I:%M %p}".replace("AM", "am").replace("PM", "pm")


def _slot_list(slots, limit: int = 4, show_doctor: bool = False) -> str:
    """
    Format free times for the patient.

    When no particular doctor was asked for, the doctor's name is shown beside each time --
    otherwise a list of bare times is meaningless, because the patient has no idea who
    they would be seeing.
    """
    lines = []
    for s in list(slots)[:limit]:
        if isinstance(s, FreeSlot):
            when = _when(s.start)
            lines.append(f"  • {when} — {s.doctor.full_name}" if show_doctor else f"  • {when}")
        else:
            lines.append(f"  • {_when(s)}")
    return "\n".join(lines)


def respond(decision: Decision, result: ToolResult, ctx) -> str:
    """Write the patient-facing reply for one turn."""

    # --- a person is taking over -------------------------------------------
    if decision.action == Action.HANDOFF_TO_HUMAN:
        return ("Of course — I'm passing you to a member of our team now. "
                "Someone will be with you shortly.")

    # --- a factual answer ---------------------------------------------------
    if decision.action == Action.PROVIDE_INFORMATION:
        return f"Our opening hours are:\n\n{describe_opening_hours()}"

    # --- something was actually done ---------------------------------------
    if decision.action == Action.CREATE_APPOINTMENT:
        if not result.ok:
            return f"Sorry — I couldn't complete that booking: {result.detail.lower()}"
        appt = result.appointment
        doctor = ctx.db.doctor(appt.doctor_id)
        return (f"Done. You're booked with {doctor.full_name} on {_when(appt.start)}.\n"
                f"Reference: {appt.id}")

    if decision.action == Action.RESCHEDULE_APPOINTMENT:
        if not result.ok:
            return f"Sorry — I couldn't move that appointment: {result.detail.lower()}"
        appt = result.appointment
        doctor = ctx.db.doctor(appt.doctor_id)
        return f"Moved. Your appointment with {doctor.full_name} is now {_when(appt.start)}."

    if decision.action == Action.CANCEL_APPOINTMENT:
        if not result.ok:
            return f"Sorry — I couldn't cancel that: {result.detail.lower()}"
        appt = result.appointment
        doctor = ctx.db.doctor(appt.doctor_id)
        return (f"Cancelled. Your appointment with {doctor.full_name} on "
                f"{_when(appt.start)} has been removed.")

    # --- showing what is free ----------------------------------------------
    if decision.action == Action.CHECK_AVAILABILITY:
        who = decision.doctor.full_name if decision.doctor else "our doctors"
        if not result.slots:
            return (f"I couldn't find anything free for {who} then. "
                    "Would a different day work?")

        show_doctor = decision.doctor is None
        opening = (f"{who} has these times free:" if decision.doctor
                   else "Here's what's free soon:")
        if decision.guarantee == "G1":
            # The patient asked us not to book. Say so plainly, so they know we listened.
            opening = (f"Understood — I won't book anything yet.\n\n"
                       f"{who} has these times free:")
        elif decision.guarantee == "G4":
            opening = f"{decision.reason}\n\nThe nearest times available are:"
        elif decision.guarantee == "G5":
            opening = f"{decision.reason}\n\nThese are still free:"

        return (f"{opening}\n{_slot_list(result.slots, show_doctor=show_doctor)}"
                "\n\nWould you like one of these?")

    # --- asking a question --------------------------------------------------
    if decision.action == Action.ASK_FOR_MORE_INFORMATION:
        return _ask(decision, ctx)

    return "Sorry, I didn't quite follow that. Could you say it another way?"


def _ask(decision: Decision, ctx) -> str:
    """Word the question, based on what is missing."""
    missing = decision.missing

    if "confirmation" in missing and decision.offer:
        offer = decision.offer
        if offer.kind == "create":
            return f"I can book {offer.summary}. Shall I confirm that?"
        if offer.kind == "cancel":
            return f"Just to check — you'd like me to {offer.summary}?"
        if offer.kind == "reschedule":
            return f"I can {offer.summary}. Shall I go ahead?"

    if "doctor" in missing:
        from .clinic import DOCTORS
        roster = "\n".join(f"  • {d.full_name} ({d.specialty})" for d in DOCTORS)

        if decision.problem == "doctor_ambiguous" and decision.candidates:
            names = "\n".join(f"  • {d.full_name} ({d.specialty})"
                              for d in decision.candidates)
            return (f"We have more than one doctor by that name:\n{names}"
                    "\n\nWhich one did you mean?")

        if decision.problem == "doctor_unknown":
            return (f"I couldn't find a doctor by that name. Here's who's at the clinic:"
                    f"\n{roster}\n\nWho would you like to see?")

        if decision.problem == "doctor_previous":
            return ("I can't see which doctor you saw before from here. Here's the team:"
                    f"\n{roster}\n\nWho would you like to see? If you'd rather someone "
                    "checked your records, I can pass you to the clinic.")

        # doctor_missing: the patient simply has not said who yet. Offer the list rather
        # than telling them we could not find somebody they never named.
        return f"Of course — who would you like to see?\n{roster}"

    if "which_appointment" in missing:
        if decision.candidates:
            listed = "\n".join(
                f"  • {ctx.db.doctor(a.doctor_id).full_name} — {_when(a.start)}"
                for a in decision.candidates if isinstance(a, Appointment)
            )
            if listed:
                return f"You have these appointments coming up:\n{listed}\n\nWhich one did you mean?"
        return "Which appointment did you mean?"

    if "new_date" in missing:
        return "When would you like to move it to?"

    if "date" in missing:
        return f"{decision.reason} What day would suit you instead?"

    if "time" in missing:
        return f"{decision.reason} Could you give me the time more precisely — for example 9am or 4pm?"

    if "what_to_confirm" in missing:
        return "Sorry — I don't have anything pending to confirm. What would you like to do?"

    if "what_they_would_prefer" in missing:
        return "No problem, I won't book that. What would work better for you?"

    return "Could you tell me a little more about what you need?"
