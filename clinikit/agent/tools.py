"""
The things the clinic can actually do.

THE SIMPLE VERSION
    Up to now, nothing in this project has changed anything. The reader worked out what
    the patient said. The policy layer decided what should happen. Both of them only
    produced words.

    This file is the hands. It is the only place in the whole project where an
    appointment is really created, moved or cancelled.

WHY THAT IS USEFUL
    If you want to know everything that can change a patient's booking, you read one
    file. Not ten.

A DELIBERATE DOUBLE-CHECK
    The policy layer already checked that a booking is allowed. The three functions here
    that change something check again anyway: is the doctor real, is the clinic open, is
    the slot still free.

    That looks like wasted work, and normally it would be. It is here on purpose. If
    somebody later edits the policy layer and makes a mistake, these checks stop a bad
    booking from reaching the appointment book. Two locks on the same door.

NOTE ON "MOCKED"
    The assessment says these actions can be fake, and they are: everything lives in
    memory and disappears when the program stops. But the function names and arguments
    are shaped the way real ones would be, so swapping in a real database later would
    mean rewriting the inside of these functions and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

from .clinic import (
    Appointment,
    ClinicDB,
    Doctor,
    clinic_hours_on,
    describe_opening_hours,
    find_doctors,
)


@dataclass(frozen=True)
class ToolResult:
    """What happened when an action ran."""

    ok: bool
    action: str
    detail: str
    """Plain sentence describing the outcome. Goes into the audit log."""

    appointment: Optional[Appointment] = None
    slots: tuple = ()
    """Free times to offer. Each carries the doctor it belongs to (policy.FreeSlot)."""
    info: str = ""
    extra: dict = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Actions that only look
# ──────────────────────────────────────────────────────────────────────────────

def check_availability(
    db: ClinicDB,
    now: datetime,
    doctor: Optional[Doctor] = None,
    days: tuple[date, ...] = (),
    earliest=None,
    latest=None,
    limit: int = 6,
) -> ToolResult:
    """Find free appointment slots. Changes nothing."""
    doctors = [doctor] if doctor else list(find_doctors("") or [])
    if not doctors:
        from .clinic import DOCTORS
        doctors = list(DOCTORS)

    from .policy import FreeSlot

    found: list = []
    seen: set = set()
    for d in days[:5]:
        for doc in doctors:
            for slot in db.free_slots(doc, d, now):
                if earliest and slot.time() < earliest:
                    continue
                if latest and slot.time() >= latest:
                    continue
                if doctor is None and slot in seen:
                    continue
                seen.add(slot)
                found.append(FreeSlot(slot, doc))
        if len(found) >= limit:
            break

    found = sorted(found, key=lambda f: f.start)[:limit]
    who = doctor.full_name if doctor else "any doctor"
    return ToolResult(
        ok=bool(found),
        action="check_availability",
        detail=f"Found {len(found)} free slot(s) for {who}.",
        slots=tuple(found),
    )


def provide_information(topic: str = "opening_hours") -> ToolResult:
    """Answer a factual question. Changes nothing."""
    return ToolResult(
        ok=True,
        action="provide_information",
        detail=f"Provided information about {topic}.",
        info=describe_opening_hours(),
    )


def handoff_to_human(reason: str = "") -> ToolResult:
    """Pass the conversation to a member of staff. Changes nothing in the book."""
    return ToolResult(
        ok=True,
        action="handoff_to_human",
        detail=f"Handed to a human. Reason: {reason or 'patient requested it'}.",
    )


def ask_for_more_information(missing: tuple[str, ...] = ()) -> ToolResult:
    """Ask the patient a question. Changes nothing."""
    return ToolResult(
        ok=True,
        action="ask_for_more_information",
        detail=f"Asked the patient for: {', '.join(missing) or 'clarification'}.",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Actions that change the appointment book
# ──────────────────────────────────────────────────────────────────────────────
# Each one re-checks its own conditions. See the note at the top of the file.

def _slot_is_bookable(
    db: ClinicDB, doctor: Doctor, start: datetime, now: datetime
) -> Optional[str]:
    """Return a reason the slot cannot be booked, or None if it is fine."""
    if start <= now:
        return "that time has already passed"
    hours = clinic_hours_on(start.date())
    if hours is None:
        return "the clinic is closed that day"
    if not (hours[0] <= start.time() < hours[1]):
        return "that time is outside the clinic's opening hours"
    if not doctor.works_on(start.date()):
        return f"{doctor.full_name} does not work that day"
    if not (doctor.start <= start.time() < doctor.end):
        return f"{doctor.full_name} does not see patients at that hour"
    if db.is_slot_taken(doctor.id, start):
        return "that slot is already booked"
    return None


def create_appointment(
    db: ClinicDB,
    patient_id: str,
    doctor: Doctor,
    start: datetime,
    now: datetime,
    reason: Optional[str] = None,
) -> ToolResult:
    """Book a new appointment."""
    blocked = _slot_is_bookable(db, doctor, start, now)
    if blocked:
        return ToolResult(
            ok=False,
            action="create_appointment",
            detail=f"Refused to book: {blocked}.",
        )

    appointment = db.add(patient_id, doctor.id, start, reason)
    return ToolResult(
        ok=True,
        action="create_appointment",
        detail=f"Booked {doctor.full_name} on {start:%A %d %B at %H:%M}.",
        appointment=appointment,
    )


def reschedule_appointment(
    db: ClinicDB,
    appointment: Appointment,
    new_start: datetime,
    now: datetime,
) -> ToolResult:
    """Move an existing appointment to a new time."""
    doctor = db.doctor(appointment.doctor_id)
    if doctor is None:
        return ToolResult(False, "reschedule_appointment", "That doctor no longer exists.")
    if appointment.status != "booked":
        return ToolResult(False, "reschedule_appointment",
                          "That appointment is not active, so it cannot be moved.")

    blocked = _slot_is_bookable(db, doctor, new_start, now)
    if blocked:
        return ToolResult(False, "reschedule_appointment", f"Refused to move it: {blocked}.")

    was = appointment.start
    appointment.start = new_start
    return ToolResult(
        ok=True,
        action="reschedule_appointment",
        detail=(f"Moved {doctor.full_name} from {was:%A %d %B %H:%M} "
                f"to {new_start:%A %d %B %H:%M}."),
        appointment=appointment,
    )


def cancel_appointment(db: ClinicDB, appointment: Appointment) -> ToolResult:
    """Cancel an existing appointment."""
    if appointment.status != "booked":
        return ToolResult(False, "cancel_appointment", "That appointment was already cancelled.")

    doctor = db.doctor(appointment.doctor_id)
    appointment.status = "cancelled"
    return ToolResult(
        ok=True,
        action="cancel_appointment",
        detail=(f"Cancelled {doctor.full_name if doctor else 'appointment'} on "
                f"{appointment.start:%A %d %B at %H:%M}."),
        appointment=appointment,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Running a decision
# ──────────────────────────────────────────────────────────────────────────────

def execute(decision, ctx) -> ToolResult:
    """
    Carry out whatever the policy layer decided.

    This is a simple switchboard. It does not think about whether the decision was a
    good one -- that already happened. It just calls the matching function.

    Keeping the thinking and the doing in separate files is the point. If you want to
    know what the system is allowed to do, read policy.py. If you want to know what
    happens when it does it, read this file.
    """
    from .policy import Action

    a = decision.action

    if a == Action.HANDOFF_TO_HUMAN:
        return handoff_to_human(decision.reason)

    if a == Action.PROVIDE_INFORMATION:
        return provide_information()

    if a == Action.ASK_FOR_MORE_INFORMATION:
        return ask_for_more_information(decision.missing)

    if a == Action.CHECK_AVAILABILITY:
        # The policy layer already worked out which slots to show, so reuse them rather
        # than searching twice.
        if decision.candidates:
            return ToolResult(
                ok=True, action=a,
                detail=f"Found {len(decision.candidates)} free slot(s).",
                slots=tuple(decision.candidates),
            )
        return check_availability(ctx.db, ctx.now, doctor=decision.doctor)

    if a == Action.CREATE_APPOINTMENT:
        return create_appointment(
            ctx.db, ctx.patient_id, decision.doctor, decision.slot, ctx.now
        )

    if a == Action.RESCHEDULE_APPOINTMENT:
        return reschedule_appointment(ctx.db, decision.appointment, decision.slot, ctx.now)

    if a == Action.CANCEL_APPOINTMENT:
        return cancel_appointment(ctx.db, decision.appointment)

    return ToolResult(False, a, f"No handler for action {a!r}.")
