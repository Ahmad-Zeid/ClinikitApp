"""
Turning a conversation's memory into text, and back again.

THE PROBLEM, IN PLAIN ENGLISH
    Most web hosts today do not keep one computer running for your site. They start a
    small worker when a message arrives, let it answer, and throw it away. The next
    message may be answered by a completely different worker that has never heard of
    this patient.

    That breaks the most important moment in the whole conversation. The clinic asks
    "shall I book Wednesday at 11?", the patient says "yes" -- and the worker answering
    "yes" has no idea what was offered.

THE FIX
    Stop keeping the memory on the server. Write it down, hand it to the patient's
    browser, and have the browser hand it back with the next message. The conversation
    travels with the patient, like a paper ticket rather than a name on a list.

    Every worker can then answer any message, because everything it needs arrives with
    the message.

WHY IT IS STILL SAFE
    An obvious worry: if the patient's browser holds the memory, can the patient edit it?
    Could they write "the clinic offered me Dr. George at 11" into the ticket, say "yes",
    and get an appointment the clinic never offered? That would break G2, the rule that
    nothing is ever booked except against an offer WE made.

    So the ticket is stamped. Before sending it out we work out a short code from the
    ticket's contents and a secret only the server knows, and we attach the code. When a
    ticket comes back, we work the code out again. If somebody changed so much as one
    character, the codes do not match and the ticket is thrown away.

    The patient can read the ticket. They cannot write one. The technical name for that
    stamp is an *HMAC signature* -- a fingerprint of some text that can only be produced
    by someone holding the secret.

WHAT IS NOT CARRIED
    The list of past turns. That is display material for the inspector panel, it grows
    without limit, and no decision depends on it. The browser already has its own copy of
    the transcript on screen.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from datetime import datetime

from .clinic import DOCTORS, Appointment, ClinicDB, TIMEZONE
from .policy import FreeSlot, Offer, PendingQuestion

# The secret used to stamp tickets.
#
# In production this MUST be set, and must be the same on every worker or tickets issued
# by one are rejected by the next. Locally we fall back to a fixed development value so
# the project runs with no setup -- that value is public, so it protects nothing, which
# is exactly why production has to override it.
_DEV_SECRET = "clinikit-development-secret-not-for-production"
SECRET = os.environ.get("CLINIKIT_STATE_SECRET", _DEV_SECRET)

STATE_VERSION = 1

# Lines of transcript carried forward. Two lines per turn (patient, then clinic), so this
# is the last ten turns.
HISTORY_CARRIED = 20


class TamperedState(Exception):
    """The ticket did not match its stamp, or could not be read at all."""


# ──────────────────────────────────────────────────────────────────────────────
# One object at a time
# ──────────────────────────────────────────────────────────────────────────────

def _doctor_id(doctor) -> str | None:
    return getattr(doctor, "id", None)


def _find_doctor(doctor_id: str | None):
    return next((d for d in DOCTORS if d.id == doctor_id), None)


def _dump_candidate(item) -> dict | None:
    """
    Numbered choices we showed the patient. Three kinds can appear in that list, so each
    one records which kind it is; without that we could not rebuild it.
    """
    if isinstance(item, FreeSlot):
        return {"kind": "slot", "start": item.start.isoformat(),
                "doctor_id": _doctor_id(item.doctor)}
    if isinstance(item, Appointment):
        return {"kind": "appointment", "id": item.id, "patient_id": item.patient_id,
                "doctor_id": item.doctor_id, "start": item.start.isoformat(),
                "reason": item.reason, "status": item.status}
    if _doctor_id(item) is not None:
        return {"kind": "doctor", "doctor_id": item.id}
    return None


def _load_candidate(raw: dict):
    kind = raw.get("kind")
    if kind == "slot":
        return FreeSlot(start=datetime.fromisoformat(raw["start"]),
                        doctor=_find_doctor(raw["doctor_id"]))
    if kind == "appointment":
        return _load_appointment(raw)
    if kind == "doctor":
        return _find_doctor(raw["doctor_id"])
    return None


def _dump_appointment(a: Appointment) -> dict:
    return {"id": a.id, "patient_id": a.patient_id, "doctor_id": a.doctor_id,
            "start": a.start.isoformat(), "reason": a.reason, "status": a.status}


def _load_appointment(raw: dict) -> Appointment:
    return Appointment(
        id=raw["id"], patient_id=raw["patient_id"], doctor_id=raw["doctor_id"],
        start=datetime.fromisoformat(raw["start"]),
        reason=raw.get("reason"), status=raw.get("status", "booked"),
    )


def _dump_offer(offer: Offer | None) -> dict | None:
    if offer is None:
        return None
    return {"kind": offer.kind, "summary": offer.summary, "doctor_id": offer.doctor_id,
            "start": offer.start.isoformat() if offer.start else None,
            "appointment_id": offer.appointment_id}


def _load_offer(raw: dict | None) -> Offer | None:
    if not raw:
        return None
    return Offer(
        kind=raw["kind"], summary=raw["summary"], doctor_id=raw.get("doctor_id"),
        start=datetime.fromisoformat(raw["start"]) if raw.get("start") else None,
        appointment_id=raw.get("appointment_id"),
    )


# ──────────────────────────────────────────────────────────────────────────────
# The whole conversation
# ──────────────────────────────────────────────────────────────────────────────

def snapshot(session) -> dict:
    """Everything a later message needs in order to continue this conversation."""
    return {
        "v": STATE_VERSION,
        "patient_id": session.patient_id,
        "known_slots": dict(session.known_slots),
        "pending_offer": _dump_offer(session.pending_offer),
        "pending_question": (
            {"kind": session.pending_question.kind,
             "summary": session.pending_question.summary,
             "doctor_ids": list(session.pending_question.doctor_ids)}
            if session.pending_question else None
        ),
        "offered_options": [c for c in
                            (_dump_candidate(o) for o in session.offered_options)
                            if c is not None],
        "last_touched_appointment_id": session.last_touched_appointment_id,
        "clarifications": session.clarifications,
        # Only the last few lines are ever shown to the reader (see _reader_history),
        # so carrying the whole transcript would grow the ticket forever for no gain.
        "history": list(session.history)[-HISTORY_CARRIED:],
        "appointments": [_dump_appointment(a) for a in session.db.appointments],
        "counter": session.db._counter,
    }


def restore(session, state: dict) -> None:
    """Put a snapshot back into a fresh Session, in place."""
    session.patient_id = state.get("patient_id", session.patient_id)
    session.known_slots = dict(state.get("known_slots") or {})
    session.pending_offer = _load_offer(state.get("pending_offer"))

    pq = state.get("pending_question")
    session.pending_question = (
        PendingQuestion(kind=pq["kind"], summary=pq.get("summary", ""),
                        doctor_ids=tuple(pq.get("doctor_ids") or ()))
        if pq else None
    )

    session.offered_options = tuple(
        c for c in (_load_candidate(o) for o in state.get("offered_options") or [])
        if c is not None
    )
    session.last_touched_appointment_id = state.get("last_touched_appointment_id")
    session.clarifications = int(state.get("clarifications") or 0)
    session.history = list(state.get("history") or [])

    session.db = ClinicDB(
        appointments=[_load_appointment(a) for a in state.get("appointments") or []],
        _counter=int(state.get("counter") or 0),
    )


# ──────────────────────────────────────────────────────────────────────────────
# The stamp
# ──────────────────────────────────────────────────────────────────────────────

def _stamp(payload: bytes) -> str:
    """The short code proving this text came from us and was not edited."""
    digest = hmac.new(SECRET.encode(), payload, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).decode().rstrip("=")


def seal(session) -> str:
    """Snapshot a conversation and stamp it, ready to hand to the browser."""
    payload = json.dumps(snapshot(session), separators=(",", ":")).encode()
    body = base64.urlsafe_b64encode(payload).decode().rstrip("=")
    return f"{body}.{_stamp(payload)}"


def open_sealed(token: str) -> dict:
    """
    Check a returned ticket and read it. Raises TamperedState if anything is wrong.

    `hmac.compare_digest` rather than `==` on purpose: it always takes the same amount of
    time, so an attacker cannot learn the correct code one character at a time by
    measuring how long each guess took. That attack is called a *timing attack*.
    """
    try:
        body, given = token.rsplit(".", 1)
        payload = base64.urlsafe_b64decode(body + "=" * (-len(body) % 4))
    except Exception as exc:
        raise TamperedState("state could not be read") from exc

    if not hmac.compare_digest(_stamp(payload), given):
        raise TamperedState("state failed its signature check")

    state = json.loads(payload)
    if state.get("v") != STATE_VERSION:
        raise TamperedState("state was written by a different version")
    return state
