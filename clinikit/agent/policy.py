"""
The safety layer. This is the most important file in the project.

WHAT IT DOES, IN PLAIN TERMS
    Everything so far has been about *understanding* the patient. This file decides what
    the clinic is allowed to *do* about it. It looks at what was extracted, checks it
    against the clinic's real calendar and rules, and returns the name of an action.

WHAT MAKES IT SAFE
    It returns the name of an action as a piece of text. It cannot perform one. There is
    no database write anywhere in this file, and nothing here can reach the internet. So
    even if the language model is completely wrong, or somebody writes a malicious
    message trying to hijack it, the worst that happens is this file is asked to consider
    a bad suggestion -- and the rules below reject it.

    That is why the rules are ordinary `if` statements. You can read every one of them,
    and so can an interviewer or an auditor. Safety you can read beats safety you hope for.

THE SEVEN GUARANTEES
    G1  No write when the patient said not to act yet (is_hedged).
    G2  No write without explicit confirmation of an offer WE made.
    G3  No booking for a doctor who does not exist or whose name is ambiguous.
    G4  No booking outside clinic hours or the doctor's own hours.
    G5  No booking in a slot already taken, or in the past.
    G6  Low confidence never writes -- it asks a question or fetches a human.
    G7  The model cannot invoke tools. True by construction: see above.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from typing import Optional

from .clinic import (
    Appointment,
    ClinicDB,
    Doctor,
    clinic_hours_on,
    find_doctors,
    suggest_doctors_for,
)
from .schema import Extraction, Intent
from .temporal import AMBIGUOUS_HOUR, CLOSED_DAY, PAST_DATE, Resolved, resolve


# ──────────────────────────────────────────────────────────────────────────────
# Actions
# ──────────────────────────────────────────────────────────────────────────────

class Action:
    """
    The things the system may decide to do.

    The first six are the brief's list, spelled exactly as the brief spells them.
    PROVIDE_INFORMATION is an addition: "what time do you close?" needs an answer, and
    none of the six fit. The brief says "possible actions may include", so extending the
    list is allowed -- but the addition is marked rather than quietly mixed in.
    """

    CHECK_AVAILABILITY = "check_availability"
    CREATE_APPOINTMENT = "create_appointment"
    RESCHEDULE_APPOINTMENT = "reschedule_appointment"
    CANCEL_APPOINTMENT = "cancel_appointment"
    HANDOFF_TO_HUMAN = "handoff_to_human"
    ASK_FOR_MORE_INFORMATION = "ask_for_more_information"

    PROVIDE_INFORMATION = "provide_information"  # added -- see docstring


# The three actions that change the appointment book. Everything else only looks.
# The safety tests import this set directly, so "what counts as dangerous" is defined
# in exactly one place and can never drift out of step with the tests.
WRITE_ACTIONS: frozenset[str] = frozenset({
    Action.CREATE_APPOINTMENT,
    Action.RESCHEDULE_APPOINTMENT,
    Action.CANCEL_APPOINTMENT,
})

# Below this, we ask instead of acting.
CONFIDENCE_THRESHOLD = 0.55

# After this many clarifying questions in a row, stop asking and fetch a human.
MAX_CLARIFICATIONS = 3


# ──────────────────────────────────────────────────────────────────────────────
# Inputs and outputs
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FreeSlot:
    """
    One bookable time, and the doctor it belongs to.

    Used to be a bare time. That produced replies like "8:00 am, 8:00 am, 8:30 am" --
    two different doctors free at the same moment, listed twice, with no way for the
    patient to tell them apart. Carrying the doctor alongside the time fixes both the
    duplicate and the missing name.
    """

    start: datetime
    doctor: "Doctor"


@dataclass(frozen=True)
class Offer:
    """
    A specific thing the SYSTEM proposed and the patient may accept.

    This is the mechanism behind G2. A patient saying "yes" can only ever confirm an
    Offer that we created and stored ourselves. It is impossible for a patient message
    to invent one, so "yes" can never book something that was never proposed.
    """

    kind: str                       # "create" | "reschedule" | "cancel"
    summary: str                    # readable description, used in the question we ask
    doctor_id: Optional[str] = None
    start: Optional[datetime] = None
    appointment_id: Optional[str] = None


@dataclass(frozen=True)
class PendingQuestion:
    """
    What a plain "yes" would mean, when the last thing we said was a yes/no question.

    An Offer covers "shall I book this?" -- a specific appointment waiting for consent.
    But plenty of our questions are not offers: "Would you like me to check when Dr.
    Nassar is free?" is a yes/no question with a clear follow-up, and "yes please" to it
    used to come back as "I'm not sure which request you're confirming" -- while the
    patient was answering the question we had just asked them.

    Deliberately NOT an Offer, and deliberately cannot lead to a write. Saying yes to
    "shall I look something up" must never book anything.
    """

    kind: str                       # currently only "check_availability"
    summary: str = ""
    doctor_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Context:
    """
    A read-only snapshot of everything the decision may depend on.

    This is the photocopy. `decide()` reads it and never changes it -- which is why the
    same Context and the same Extraction always produce the same Decision, and why the
    tests can pin the clock and get identical results forever.
    """

    now: datetime
    db: ClinicDB
    patient_id: str = "p_001"
    known_slots: dict = field(default_factory=dict)
    pending_offer: Optional[Offer] = None
    clarifications_so_far: int = 0

    pending_question: Optional[PendingQuestion] = None
    """The yes/no question we asked last turn, if it had a concrete follow-up."""

    offered_options: tuple = ()
    """
    The numbered choices we showed the patient last turn, in the order shown.

    Needed so "the first one" means something. Without it, we list four times and then
    cannot accept a reference to any of them -- which reads as the assistant not having
    been paying attention to its own message.
    """

    last_touched_appointment_id: Optional[str] = None
    """
    The appointment we booked, moved or cancelled on the previous turn.

    Needed so that "cancel that" straight after booking means the thing we just booked.
    Without it, the patient books an appointment, says "actually cancel that", and gets
    a list of all their appointments asking which one -- which reads as the assistant
    not having been paying attention.
    """


@dataclass(frozen=True)
class Decision:
    """What the system has decided to do, and why."""

    action: str
    reason: str
    """Plain-language explanation. Shown in the UI, written to the audit log, and the
    thing a reviewer reads to understand what happened."""

    guarantee: Optional[str] = None
    """Which guarantee (G1..G7) blocked a write, when one did. Empty otherwise."""

    problem: Optional[str] = None
    """
    A short code for what went wrong, e.g. "doctor_missing", "doctor_unknown",
    "doctor_ambiguous". The wording of the reply is chosen from this.

    It used to be chosen by searching for words inside `reason`, which broke: the reason
    "No doctor was named" contains "No doctor", so a patient who said "I need to see a
    doctor" was told "I couldn't find that doctor". Codes cannot misfire that way.
    """

    doctor: Optional[Doctor] = None
    slot: Optional[datetime] = None
    appointment: Optional[Appointment] = None
    offer: Optional[Offer] = None
    """An offer to remember for the next turn. The session layer stores this."""

    question: Optional[PendingQuestion] = None
    """A yes/no question we are asking, so next turn's "yes" has something to mean."""

    missing: tuple[str, ...] = ()
    """What we still need from the patient, e.g. ("doctor", "date")."""

    candidates: tuple = ()
    """Options to present: ambiguous doctors, or free slots to choose from."""

    specialty: Optional[str] = None
    """
    The department a described problem belongs to, when we are suggesting a doctor.

    Set only by the routing path. It names a department -- it never carries anything
    about what the symptom might mean. See the note above SYMPTOM_ROUTES in clinic.py.
    """

    topic: Optional[str] = None
    """
    What the patient wanted, in the model's own words, carried through for a human.

    This is what stops "other" being a dead end: instead of "could you tell me more",
    the person who picks up the conversation sees "asking about their insurance".
    """

    @property
    def is_write(self) -> bool:
        return self.action in WRITE_ACTIONS


# ──────────────────────────────────────────────────────────────────────────────
# The decision
# ──────────────────────────────────────────────────────────────────────────────

def decide(extraction: Extraction, ctx: Context) -> Decision:
    """
    Choose an action. Pure: reads its two arguments, changes nothing.

    The order of these rules is deliberate and is the safety argument. Rules that BLOCK
    come before rules that ACT, so a blocking condition can never be skipped over by an
    earlier rule that decided to do something.
    """

    # --- 0. A greeting is answered, not investigated -------------------------
    # Placed before the confidence check on purpose. "hi" carries almost no information,
    # so it scores low -- and the confidence rule then treated it as a failure to
    # understand and replied "could you tell me more about what you need". Saying hello
    # back is not a risk; it is the entire correct response.
    if extraction.intent == Intent.GREETING:
        return Decision(
            action=Action.PROVIDE_INFORMATION,
            reason="The patient said hello. Greeting them back and offering help.",
            topic="greeting",
        )

    # --- 1. An explicit request for a person always wins ---------------------
    if extraction.intent == Intent.TALK_TO_HUMAN:
        return Decision(
            action=Action.HANDOFF_TO_HUMAN,
            reason="The patient asked to speak to a person.",
        )

    # --- 1b. Picking one of the choices we listed ----------------------------
    # Deliberately BEFORE the confidence gate AND the confirmation rule.
    #
    # Before the confidence gate, because "the first one" carries almost no words and
    # therefore scores low -- and the gate then replied "could you tell me more about
    # what you need" to somebody who had just answered our own numbered list. This is
    # safe to let through: the reference is resolved by plain pattern matching against
    # options WE stored, and the result still needs a confirmation before anything is
    # written.
    #
    # Before the confirmation rule, "The second one" is often read as a
    # confirmation, and if the confirm rule ran first it would find nothing pending and
    # reply "I don't have anything waiting for a yes" -- while the patient is looking at
    # the numbered list we just sent them. A specific choice beats a bare yes.
    if _is_really_picking_an_option(extraction, ctx):
        picked = _resolve_option(extraction.option_reference, ctx)
        if picked is None:
            return Decision(
                action=Action.ASK_FOR_MORE_INFORMATION,
                reason=(f"Could not tell which option {extraction.option_reference!r} "
                        f"refers to among the {len(ctx.offered_options)} shown."),
                missing=("which_option",),
                candidates=tuple(ctx.offered_options),
            )
        if isinstance(picked, Doctor):
            with_doctor = extraction.model_copy(update={"doctor": picked.full_name})
            resolved = resolve(extraction.preferred_date, extraction.preferred_time, ctx.now)
            if not resolved.has_date:
                # They picked a doctor and said nothing about when. Asking "what day?" is
                # what a form does. Showing that doctor's next free times is what a
                # receptionist does, and it moves things forward instead of back a step.
                return _remember_single_slot(Decision(
                    action=Action.CHECK_AVAILABILITY,
                    reason=f"Chose {picked.full_name} from the list; showing their next "
                           f"free times.",
                    **_availability_lookup(with_doctor, ctx, doctor=picked),
                ))
            return _propose_booking(with_doctor, ctx)
        if isinstance(picked, Appointment):
            return _propose_cancel(
                extraction.model_copy(
                    update={"existing_appointment_phrase": picked.id}), ctx
            )
        return _validate_exact_slot(picked.doctor, picked.start, ctx)


    # --- 2. Not confident enough to act -------------------------- G6 --------
    # Checked early, and before anything that could write. A low-confidence reading is
    # exactly the situation where acting on it is most likely to be wrong.
    if extraction.confidence < CONFIDENCE_THRESHOLD:
        if ctx.clarifications_so_far >= MAX_CLARIFICATIONS:
            return Decision(
                action=Action.HANDOFF_TO_HUMAN,
                reason=(
                    f"Still unclear after {ctx.clarifications_so_far} attempts "
                    f"(confidence {extraction.confidence:.2f}). Handing to a person."
                ),
                guarantee="G6",
            )
        return Decision(
            action=Action.ASK_FOR_MORE_INFORMATION,
            reason=(
                f"Confidence {extraction.confidence:.2f} is below the "
                f"{CONFIDENCE_THRESHOLD} threshold, so no action is taken."
            ),
            guarantee="G6",
            missing=("clarification",),
        )

    # --- 3. "Yes" -------------------------------------------------- G2 ------
    # A confirmation is only meaningful against an offer we ourselves made.
    if extraction.intent == Intent.CONFIRM:
        if ctx.pending_offer is not None:
            return _execute_offer(ctx.pending_offer, ctx)

        # Not an offer, but we did ask them a yes/no question. Answer it.
        # This path can only ever look things up -- see PendingQuestion.
        if ctx.pending_question is not None and ctx.pending_question.kind == "check_availability":
            doctors = [d for d in (ctx.db.doctor(i) for i in ctx.pending_question.doctor_ids)
                       if d is not None]
            only = doctors[0] if len(doctors) == 1 else None
            return _remember_single_slot(Decision(
                action=Action.CHECK_AVAILABILITY,
                reason=f"Said yes to: {ctx.pending_question.summary}",
                **_availability_lookup(extraction, ctx, doctor=only),
            ))

        # A list is on screen and they said yes. They mean one of those -- we just do
        # not know which. "I don't have anything waiting for a yes" is technically true
        # and completely useless when the patient is looking at four options we sent.
        if ctx.offered_options:
            return Decision(
                action=Action.ASK_FOR_MORE_INFORMATION,
                reason="Said yes while a list of options was on screen; asking which one.",
                missing=("which_option",),
                candidates=tuple(ctx.offered_options),
            )

        return Decision(
            action=Action.ASK_FOR_MORE_INFORMATION,
            reason="The patient agreed to something, but nothing was offered to agree to.",
            guarantee="G2",
            missing=("what_to_confirm",),
        )

    # --- 4. "No" -------------------------------------------------------------
    if extraction.intent == Intent.DENY:
        return Decision(
            action=Action.ASK_FOR_MORE_INFORMATION,
            reason="The patient declined. The pending offer is dropped.",
            missing=("what_they_would_prefer",),
        )

    # --- 5. Opening hours ----------------------------------------------------
    if extraction.intent == Intent.ASK_OPENING_HOURS:
        return Decision(
            action=Action.PROVIDE_INFORMATION,
            reason="The patient asked about opening hours. No booking involved.",
            topic="opening_hours",
        )

    # --- 5b. A question about the clinic itself ------------------------------
    # "Who do I see about my heart?" is answered by naming the department -- never by
    # saying anything about the symptom. See the note above SYMPTOM_ROUTES.
    if extraction.intent == Intent.ASK_CLINIC_INFO:
        # Both fields, joined. "My son has a fever" often arrives as
        # reason_for_visit="fever" with the "my son" part only in the summary -- and "my
        # son" is what points at the paediatrician. Looking at one field alone sent a
        # child to a general practitioner.
        described = " ".join(p for p in (extraction.reason_for_visit,
                                         extraction.request_summary) if p)

        # fall_back_to_gp=False matters here. With it on, ANY question produced a doctor
        # suggestion -- "do you take my insurance?" came back recommending a
        # dermatologist. Only route when a real symptom was recognised.
        specialty, doctors = suggest_doctors_for(described, fall_back_to_gp=False)
        if doctors and specialty:
            return Decision(
                action=Action.PROVIDE_INFORMATION,
                reason=f"Asked which doctor covers this. Naming our {specialty.lower()}.",
                specialty=specialty,
                candidates=tuple(doctors),
                topic="which_doctor",
                question=PendingQuestion(
                    kind="check_availability",
                    summary=f"checking when our {specialty.lower()} is free",
                    doctor_ids=tuple(d.id for d in doctors),
                ),
            )

        # Things we actually hold: where we are, how to reach us, when we are open.
        question = (extraction.request_summary or "").lower()
        if any(word in question for word in
               ("where", "location", "address", "find", "parking", "phone", "number",
                "contact", "reach", "hours", "open", "close", "directions")):
            return Decision(
                action=Action.PROVIDE_INFORMATION,
                reason="A question about the clinic that we hold the answer to.",
                topic="clinic_info",
            )

        # Everything else -- insurance, billing, prescriptions, referrals. We do not
        # hold these answers, and guessing at them would be worse than useless.
        return Decision(
            action=Action.HANDOFF_TO_HUMAN,
            reason=f"Not something this assistant can answer: {extraction.request_summary}.",
            topic=extraction.request_summary or "a question about the clinic",
        )

    # --- 6. The patient told us not to act yet --------------------- G1 ------
    # Deliberately placed BEFORE every booking rule below. It does not matter which
    # intent won, or how confident the model was: if the message contains an explicit
    # "not yet", nothing may be written. This is the rule that handles the brief's
    # ambiguous example, and it holds regardless of what the model decided the intent was.
    if extraction.is_hedged:
        return _remember_single_slot(Decision(
            action=Action.CHECK_AVAILABILITY,
            reason=(
                "The patient explicitly asked us not to act yet, so this is treated as a "
                "look-up only. Nothing will be booked."
            ),
            guarantee="G1",
            **_availability_lookup(extraction, ctx),
        ))

    # --- 7. Asking what is free ----------------------------------------------
    if extraction.intent == Intent.ASK_DOCTOR_AVAILABILITY:
        return _remember_single_slot(Decision(
            action=Action.CHECK_AVAILABILITY,
            reason="The patient asked what is available. Looking up, not booking.",
            **_availability_lookup(extraction, ctx),
        ))

    # --- 8. Cancel -----------------------------------------------------------
    if extraction.intent == Intent.CANCEL_APPOINTMENT:
        return _propose_cancel(extraction, ctx)

    # --- 9. Reschedule -------------------------------------------------------
    if extraction.intent == Intent.RESCHEDULE_APPOINTMENT:
        return _propose_reschedule(extraction, ctx)

    # --- 10. Book ------------------------------------------------------------
    if extraction.intent == Intent.BOOK_APPOINTMENT:
        return _propose_booking(extraction, ctx)

    # --- 11. Anything else ---------------------------------------------------
    # This used to reply "could you tell me more about what you need" -- a dead end for
    # anyone asking about insurance, prescriptions or parking.
    #
    # The reader now always writes a plain-language summary of what was wanted, whatever
    # it was, so we can hand the conversation to a person WITH that context instead of
    # asking the patient to rephrase something we were never going to understand.
    if extraction.request_summary:
        return Decision(
            action=Action.HANDOFF_TO_HUMAN,
            reason=f"Outside what this assistant covers: {extraction.request_summary}.",
            topic=extraction.request_summary,
        )

    return Decision(
        action=Action.ASK_FOR_MORE_INFORMATION,
        reason="The message was not a recognisable clinic request.",
        missing=("what_they_need",),
    )


def _remember_single_slot(decision: Decision) -> Decision:
    """
    If we showed the patient exactly one free time, remember it as an offer.

    Without this, the conversation dead-ends. We say "Tuesday 11am is free" and the
    patient says "yes, book it" -- and there is nothing stored for that "yes" to attach
    to, so we have to ask them to start again. Annoying, and it makes the agent look
    broken.

    This is still safe. The offer is one WE created from the clinic's real calendar, and
    it still needs an explicit yes before anything is written. G2 is untouched.

    Only fires when there is exactly one slot. With several on screen, "yes" is genuinely
    unclear, and picking one for the patient is the kind of quiet guess this system does
    not make.
    """
    if decision.offer is not None:
        return decision

    slots = [c for c in decision.candidates if isinstance(c, FreeSlot)]
    if len(slots) != 1:
        return decision

    slot, doctor_for_slot = slots[0].start, slots[0].doctor
    decision = replace(decision, doctor=decision.doctor or doctor_for_slot)
    if decision.appointment is not None:
        summary = (f"move {decision.doctor.full_name} to "
                   f"{slot:%A %d %B at %H:%M}")
        offer = Offer(kind="reschedule", summary=summary, doctor_id=decision.doctor.id,
                      start=slot, appointment_id=decision.appointment.id)
    else:
        summary = f"{decision.doctor.full_name} on {slot:%A %d %B at %H:%M}"
        offer = Offer(kind="create", summary=summary,
                      doctor_id=decision.doctor.id, start=slot)
    return replace(decision, offer=offer)


# ──────────────────────────────────────────────────────────────────────────────
# Turning a confirmed offer into a write
# ──────────────────────────────────────────────────────────────────────────────

def _execute_offer(offer: Offer, ctx: Context) -> Decision:
    """
    The only route to a write action in this entire file.

    Reached only from rule 3, which requires an explicit confirmation AND an offer the
    system itself created. Both conditions together are G2.
    """
    mapping = {
        "create": Action.CREATE_APPOINTMENT,
        "reschedule": Action.RESCHEDULE_APPOINTMENT,
        "cancel": Action.CANCEL_APPOINTMENT,
    }
    action = mapping.get(offer.kind)
    if action is None:
        return Decision(
            action=Action.ASK_FOR_MORE_INFORMATION,
            reason=f"Unrecognised offer type {offer.kind!r}; refusing to act on it.",
            guarantee="G2",
        )

    # The slot may have been taken by someone else between the offer and the "yes".
    if offer.kind in ("create", "reschedule") and offer.doctor_id and offer.start:
        if ctx.db.is_slot_taken(offer.doctor_id, offer.start):
            return Decision(
                action=Action.CHECK_AVAILABILITY,
                reason="That slot was taken while we were waiting for confirmation.",
                guarantee="G5",
                doctor=ctx.db.doctor(offer.doctor_id),
                **_slots_for_doctor(offer.doctor_id, offer.start.date(), ctx),
            )

    return Decision(
        action=action,
        reason=f"The patient confirmed an offer the system made: {offer.summary}.",
        doctor=ctx.db.doctor(offer.doctor_id) if offer.doctor_id else None,
        slot=offer.start,
        appointment=_appointment_by_id(ctx, offer.appointment_id),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Booking
# ──────────────────────────────────────────────────────────────────────────────

def _propose_booking(extraction: Extraction, ctx: Context) -> Decision:
    """
    Work towards a booking, stopping at the first thing that is missing or wrong.

    Never returns a write action. The best outcome is an Offer plus a question --
    "shall I book this?" -- which the patient must then accept. That is G2 in practice.
    """
    doctor, problem = _resolve_doctor(extraction, ctx)
    if problem is not None:
        return problem

    resolved = resolve(extraction.preferred_date, extraction.preferred_time, ctx.now)

    if PAST_DATE in resolved.problems:
        return Decision(
            action=Action.ASK_FOR_MORE_INFORMATION,
            reason="The time requested has already passed.",
            guarantee="G5", doctor=doctor, missing=("date",),
        )
    if CLOSED_DAY in resolved.problems:
        return Decision(
            action=Action.ASK_FOR_MORE_INFORMATION,
            reason="The clinic is closed on the day requested.",
            guarantee="G4", doctor=doctor, missing=("date",),
        )
    if AMBIGUOUS_HOUR in resolved.problems:
        return Decision(
            action=Action.ASK_FOR_MORE_INFORMATION,
            reason="The time given could mean two different hours.",
            doctor=doctor, missing=("time",),
        )
    if not resolved.has_date:
        return Decision(
            action=Action.ASK_FOR_MORE_INFORMATION,
            reason="No usable date was given.",
            doctor=doctor, missing=("date",),
        )

    # An exact slot was named: validate it properly before offering it.
    if resolved.is_exact and doctor is not None:
        return _validate_exact_slot(doctor, resolved.exact, ctx)

    # Otherwise we have a range. Show what is free and let them choose.
    return _remember_single_slot(Decision(
        action=Action.CHECK_AVAILABILITY,
        reason="A date range was given rather than a specific time, so we offer options.",
        **_availability_lookup(extraction, ctx, doctor=doctor, resolved=resolved),
    ))


def _validate_exact_slot(doctor: Doctor, slot: datetime, ctx: Context) -> Decision:
    """Check one specific slot against every rule, then offer it. G4 and G5 live here."""
    day = slot.date()

    if slot <= ctx.now:
        return Decision(
            action=Action.ASK_FOR_MORE_INFORMATION,
            reason="That time is in the past.",
            guarantee="G5", doctor=doctor, missing=("date",),
        )

    hours = clinic_hours_on(day)
    if hours is None:
        return Decision(
            action=Action.ASK_FOR_MORE_INFORMATION,
            reason="The clinic is closed that day.",
            guarantee="G4", doctor=doctor, missing=("date",),
        )
    if not (hours[0] <= slot.time() < hours[1]):
        return Decision(
            action=Action.CHECK_AVAILABILITY,
            reason="That time is outside the clinic's opening hours.",
            guarantee="G4", doctor=doctor,
            **_slots_for_doctor(doctor.id, day, ctx),
        )
    if not doctor.works_on(day):
        return Decision(
            action=Action.CHECK_AVAILABILITY,
            reason=f"{doctor.full_name} does not work that day.",
            guarantee="G4", doctor=doctor,
            **_slots_for_doctor(doctor.id, _next_working_day(doctor, day), ctx),
        )
    if not (doctor.start <= slot.time() < doctor.end):
        return Decision(
            action=Action.CHECK_AVAILABILITY,
            reason=f"{doctor.full_name} does not see patients at that hour.",
            guarantee="G4", doctor=doctor,
            **_slots_for_doctor(doctor.id, day, ctx),
        )
    if ctx.db.is_slot_taken(doctor.id, slot):
        return Decision(
            action=Action.CHECK_AVAILABILITY,
            reason="That slot is already booked.",
            guarantee="G5", doctor=doctor,
            **_slots_for_doctor(doctor.id, day, ctx),
        )

    # Everything checks out. Offer it -- and still do not book it.
    summary = f"{doctor.full_name} on {slot:%A %d %B at %H:%M}"
    return Decision(
        action=Action.ASK_FOR_MORE_INFORMATION,
        reason="The slot is valid and free. Asking the patient to confirm before booking.",
        guarantee="G2",
        doctor=doctor, slot=slot,
        offer=Offer(kind="create", summary=summary, doctor_id=doctor.id, start=slot),
        missing=("confirmation",),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Cancel and reschedule
# ──────────────────────────────────────────────────────────────────────────────

def _propose_cancel(extraction: Extraction, ctx: Context) -> Decision:
    appointment, problem = _identify_appointment(extraction, ctx)
    if problem is not None:
        return problem

    doctor = ctx.db.doctor(appointment.doctor_id)
    summary = f"cancel {doctor.full_name} on {appointment.start:%A %d %B at %H:%M}"
    return Decision(
        action=Action.ASK_FOR_MORE_INFORMATION,
        reason="Found the appointment. Asking the patient to confirm before cancelling.",
        guarantee="G2",
        doctor=doctor, appointment=appointment,
        offer=Offer(kind="cancel", summary=summary, appointment_id=appointment.id),
        missing=("confirmation",),
    )


def _propose_reschedule(extraction: Extraction, ctx: Context) -> Decision:
    appointment, problem = _identify_appointment(extraction, ctx)
    if problem is not None:
        return problem

    doctor = ctx.db.doctor(appointment.doctor_id)
    resolved = resolve(extraction.preferred_date, extraction.preferred_time, ctx.now)

    if not resolved.has_date:
        return Decision(
            action=Action.ASK_FOR_MORE_INFORMATION,
            reason="We know which appointment to move, but not what to move it to.",
            doctor=doctor, appointment=appointment, missing=("new_date",),
        )

    if resolved.is_exact:
        check = _validate_exact_slot(doctor, resolved.exact, ctx)
        if check.offer is None:
            return check  # something was wrong with the new slot; surface it
        summary = (f"move {doctor.full_name} from {appointment.start:%A %d %B %H:%M} "
                   f"to {resolved.exact:%A %d %B %H:%M}")
        return Decision(
            action=Action.ASK_FOR_MORE_INFORMATION,
            reason="The new slot is valid and free. Asking the patient to confirm.",
            guarantee="G2",
            doctor=doctor, slot=resolved.exact, appointment=appointment,
            offer=Offer(kind="reschedule", summary=summary, doctor_id=doctor.id,
                        start=resolved.exact, appointment_id=appointment.id),
            missing=("confirmation",),
        )

    return Decision(
        action=Action.CHECK_AVAILABILITY,
        reason="A day was given but not a time. Showing what is free that day.",
        doctor=doctor, appointment=appointment,
        **_slots_for_doctor(doctor.id, resolved.days[0], ctx),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Small helpers
# ──────────────────────────────────────────────────────────────────────────────

_ORDINALS = {
    "first": 1, "1st": 1, "earliest": 1,
    "second": 2, "2nd": 2,
    "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4,
    "fifth": 5, "5th": 5,
    "last": -1, "latest": -1,
}


def _is_really_picking_an_option(extraction: Extraction, ctx: Context) -> bool:
    """
    Is this message actually choosing from the list we showed?

    The reader sometimes fills option_reference when the patient is doing nothing of the
    sort. It happened with "which doctor should I see if I have problems with my heart":
    a list of doctors was on screen, the reader decided that counted as picking one, and
    the patient was shown the first doctor's diary instead of the cardiologist.

    So the reference has to survive two checks, both deterministic:

      1. The words must ACTUALLY APPEAR in the patient's message. Same rule we use for
         dates -- the reader reports what it read, and we verify we can find it. A
         reference the patient never typed is an invention.
      2. The message must not also be describing a problem. "The first one" is a
         selection; "which doctor for my heart" is a question that happens to contain a
         number-ish word.
    """
    if not extraction.option_reference or not ctx.offered_options:
        return False

    message = (extraction.raw_message or "").lower()
    reference = extraction.option_reference.lower().strip()
    if message and reference not in message:
        # Allow a shortened form: the reader may report "the first one" for "first".
        core = reference.replace("the ", "").replace(" one", "").strip()
        if not core or core not in message:
            return False

    # Describing a symptom is not picking an appointment slot.
    if extraction.reason_for_visit:
        return False

    return True


def _resolve_option(reference: str, ctx: Context):
    """
    Work out which of the listed choices the patient meant.

    Handles "the first one", "option 2", "the 9am one", "the last one". Returns the
    chosen item, or None when it cannot be worked out -- in which case the caller asks
    rather than picking something. Guessing here would book the wrong appointment.

    Times are checked BEFORE numbers, because "the 9am one" contains a 9 that is a time,
    not a position in the list.
    """
    text = reference.lower().strip()
    options = list(ctx.offered_options)
    if not options:
        return None

    # --- a time, e.g. "the 9am one", "the 11:30" ---
    clock = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", text)
    if clock and (clock.group(3) or clock.group(2)):
        hour, minute, meridiem = int(clock.group(1)), int(clock.group(2) or 0), clock.group(3)
        if meridiem == "pm" and hour != 12:
            hour += 12
        if meridiem == "am" and hour == 12:
            hour = 0
        for option in options:
            start = getattr(option, "start", None)
            if start is not None and start.hour == hour and start.minute == minute:
                return option

    # --- a word, e.g. "the first one", "the last one" ---
    for word, position in _ORDINALS.items():
        if re.search(rf"\b{word}\b", text):
            index = len(options) - 1 if position == -1 else position - 1
            return options[index] if 0 <= index < len(options) else None

    # --- a bare number, e.g. "option 2", "number 3", "2" ---
    number = re.search(r"\b(\d{1,2})\b", text)
    if number:
        index = int(number.group(1)) - 1
        if 0 <= index < len(options):
            return options[index]

    return None


def _resolve_doctor(
    extraction: Extraction, ctx: Context
) -> tuple[Optional[Doctor], Optional[Decision]]:
    """
    Turn a written name into one real doctor. G3 lives here.

    Returns (doctor, None) on success or (None, Decision) when we must ask instead.
    Never picks one of several matches: "Dr. Khoury" matches two real doctors, and
    guessing would book a dermatologist instead of a paediatrician.
    """
    name = extraction.doctor or ctx.known_slots.get("doctor")
    if not name:
        if extraction.refers_to_previous_visit:
            return None, Decision(
                action=Action.ASK_FOR_MORE_INFORMATION,
                reason="The patient referred to a previous doctor we cannot identify.",
                guarantee="G3", problem="doctor_previous", missing=("doctor",),
            )
        # No doctor named -- but if they told us what is wrong, point them at the right
        # department rather than handing over the whole staff list and making them guess.
        described = " ".join(p for p in (extraction.reason_for_visit,
                                         extraction.request_summary) if p)
        specialty, suggested = suggest_doctors_for(described)
        if suggested and specialty and extraction.reason_for_visit:
            return None, Decision(
                action=Action.ASK_FOR_MORE_INFORMATION,
                reason=f"No doctor named, but they described something our "
                       f"{specialty.lower()} covers.",
                guarantee="G3", problem="doctor_suggested", missing=("doctor",),
                specialty=specialty, candidates=tuple(suggested),
                question=PendingQuestion(
                    kind="check_availability",
                    summary=f"checking when our {specialty.lower()} is free",
                    doctor_ids=tuple(d.id for d in suggested),
                ),
            )

        return None, Decision(
            action=Action.ASK_FOR_MORE_INFORMATION,
            reason="No doctor was named.",
            guarantee="G3", problem="doctor_missing", missing=("doctor",),
        )

    matches = find_doctors(name)
    if not matches:
        return None, Decision(
            action=Action.ASK_FOR_MORE_INFORMATION,
            reason=f"No doctor at this clinic matches {name!r}.",
            guarantee="G3", problem="doctor_unknown", missing=("doctor",), candidates=(),
        )
    if len(matches) > 1:
        return None, Decision(
            action=Action.ASK_FOR_MORE_INFORMATION,
            reason=f"{name!r} matches {len(matches)} doctors. Asking which one.",
            guarantee="G3", problem="doctor_ambiguous", missing=("doctor",),
            candidates=tuple(matches),
        )
    return matches[0], None


def _identify_appointment(
    extraction: Extraction, ctx: Context
) -> tuple[Optional[Appointment], Optional[Decision]]:
    """Work out which existing appointment the patient means."""
    existing = ctx.db.active_for_patient(ctx.patient_id, ctx.now)

    if not existing:
        return None, Decision(
            action=Action.ASK_FOR_MORE_INFORMATION,
            reason="The patient has no upcoming appointments to change.",
            missing=("which_appointment",),
        )

    phrase = (extraction.existing_appointment_phrase or "").lower()
    named_doctor = (extraction.doctor or "").lower()

    # Picked straight off a numbered list: the option resolver puts the exact id here,
    # so there is nothing to match on and nothing to guess.
    exact = next((a for a in existing if a.id.lower() == phrase), None)
    if exact is not None:
        return exact, None

    # "that", "it", "this one" straight after we acted on something means that thing.
    vague = phrase.strip() in ("", "that", "it", "this", "this one", "that one",
                               "the appointment", "my appointment")
    if vague and not named_doctor and ctx.last_touched_appointment_id:
        just_done = next((a for a in existing
                          if a.id == ctx.last_touched_appointment_id), None)
        if just_done is not None:
            return just_done, None

    matches = existing
    if phrase or named_doctor:
        narrowed = []
        for appt in existing:
            doctor = ctx.db.doctor(appt.doctor_id)
            haystack = (
                f"{appt.start:%A} {appt.start:%d} {appt.start:%B} "
                f"{doctor.first_name} {doctor.last_name}"
            ).lower()
            if (phrase and any(w in haystack for w in phrase.split() if len(w) > 2)) or (
                named_doctor and any(w in haystack for w in named_doctor.split() if len(w) > 2)
            ):
                narrowed.append(appt)
        if narrowed:
            matches = narrowed

    if len(matches) == 1:
        return matches[0], None

    return None, Decision(
        action=Action.ASK_FOR_MORE_INFORMATION,
        reason=f"The patient has {len(matches)} upcoming appointments; unclear which is meant.",
        missing=("which_appointment",), candidates=tuple(matches),
    )


def _availability_lookup(
    extraction: Extraction,
    ctx: Context,
    doctor: Optional[Doctor] = None,
    resolved: Optional[Resolved] = None,
) -> dict:
    """Gather free slots to show. Read-only: this never changes the appointment book."""
    if doctor is None:
        matches = find_doctors(extraction.doctor or ctx.known_slots.get("doctor"))
        doctor = matches[0] if len(matches) == 1 else None
    if resolved is None:
        resolved = resolve(extraction.preferred_date, extraction.preferred_time, ctx.now)

    days = resolved.days or _upcoming_days(ctx.now.date(), 5)
    doctors = [doctor] if doctor else _all_doctors()

    found: list[FreeSlot] = []
    seen_times: set[datetime] = set()
    for d in days[:5]:
        for doc in doctors:
            for s in ctx.db.free_slots(doc, d, ctx.now):
                if resolved.earliest and s.time() < resolved.earliest:
                    continue
                if resolved.latest and s.time() >= resolved.latest:
                    continue
                # When no particular doctor was asked for, show each time once. Listing
                # "8:00 am" five times because five doctors are free then is useless.
                if doctor is None and s in seen_times:
                    continue
                seen_times.add(s)
                found.append(FreeSlot(s, doc))
        if len(found) >= 6:
            break

    found.sort(key=lambda f: f.start)
    return {"doctor": doctor, "candidates": tuple(found[:6])}


def _slots_for_doctor(doctor_id: str, day: date, ctx: Context) -> dict:
    """
    Free slots for one doctor on one day, as a fragment to splat into a Decision.

    Returns only `candidates`, never `doctor`. Callers already know which doctor they
    are talking about and pass it themselves; returning it here too caused a duplicate
    keyword argument.
    """
    doctor = ctx.db.doctor(doctor_id)
    if doctor is None:
        return {"candidates": ()}
    return {"candidates": tuple(
        FreeSlot(s, doctor) for s in ctx.db.free_slots(doctor, day, ctx.now)[:6]
    )}


def _next_working_day(doctor: Doctor, start: date) -> date:
    from datetime import timedelta
    d = start
    for _ in range(14):
        d += timedelta(days=1)
        if doctor.works_on(d) and clinic_hours_on(d) is not None:
            return d
    return start


def _upcoming_days(start: date, count: int) -> tuple[date, ...]:
    from datetime import timedelta
    out, d = [], start
    while len(out) < count:
        d += timedelta(days=1)
        if clinic_hours_on(d) is not None:
            out.append(d)
    return tuple(out)


def _all_doctors() -> list[Doctor]:
    from .clinic import DOCTORS
    return list(DOCTORS)


def _appointment_by_id(ctx: Context, appointment_id: Optional[str]) -> Optional[Appointment]:
    if not appointment_id:
        return None
    return next((a for a in ctx.db.appointments if a.id == appointment_id), None)
