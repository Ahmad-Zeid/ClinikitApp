"""
Carrying a conversation's memory to the browser and back.

WHY THIS EXISTS
    On a host like Vercel, each message may be answered by a different worker with no
    memory of the last one. So the memory travels with the patient instead. These tests
    check the two things that must be true for that to be safe:

      1. A conversation survives the round trip unchanged.
      2. A patient cannot edit their own memory to invent an offer we never made.

    The second one matters most. G2 says nothing is ever booked except against an offer
    the clinic itself made. If the memory could be forged, G2 would be forgeable too.
"""

from datetime import datetime

import pytest

from clinikit.agent.clinic import TIMEZONE, seeded_db
from clinikit.agent.policy import Offer
from clinikit.agent.schema import Extraction, Intent
from clinikit.agent.session import Session
from clinikit.agent.state import TamperedState, open_sealed, restore, seal, snapshot

FROZEN_NOW = datetime(2026, 9, 14, 10, 0, tzinfo=TIMEZONE)


def _session() -> Session:
    return Session(backend="rules", clock=lambda: FROZEN_NOW)


# ══════════════════════════════════════════════════════════════════════════════
# The round trip keeps the conversation intact
# ══════════════════════════════════════════════════════════════════════════════

def test_a_pending_offer_survives_the_round_trip():
    """The moment that matters: the clinic offered something, the patient says yes."""
    first = _session()
    first.pending_offer = Offer(
        kind="create",
        summary="Dr. George Haddad on Wednesday 16 September at 11:00",
        doctor_id="d_george",
        start=datetime(2026, 9, 16, 11, 0, tzinfo=TIMEZONE),
    )
    first.known_slots = {"doctor": "Dr. George", "date": "tomorrow"}
    first.history = ["patient: book me tomorrow at 11", "clinic: shall I book that?"]

    second = _session()
    restore(second, open_sealed(seal(first)))

    assert second.pending_offer == first.pending_offer
    assert second.known_slots == first.known_slots
    assert second.history == first.history


def test_the_appointment_book_survives_the_round_trip():
    first = _session()
    first.db = seeded_db("p_001")
    before = [(a.id, a.doctor_id, a.start, a.status) for a in first.db.appointments]

    second = _session()
    restore(second, open_sealed(seal(first)))

    after = [(a.id, a.doctor_id, a.start, a.status) for a in second.db.appointments]
    assert after == before


def test_a_whole_conversation_continues_across_workers():
    """
    Simulate the real hosting problem: every message answered by a brand-new worker
    that has never seen this patient before.
    """
    booking = Extraction(intent=Intent.BOOK_APPOINTMENT, confidence=0.95,
                         doctor="Dr. George", preferred_date="tomorrow",
                         preferred_time="11")
    yes = Extraction(intent=Intent.CONFIRM, confidence=1.0)

    worker_one = _session()
    worker_one.extractor = _Fixed(booking)
    worker_one.handle("book me with Dr. George tomorrow at 11")
    assert worker_one.pending_offer is not None, "the clinic should have made an offer"

    ticket = seal(worker_one)

    worker_two = _session()                      # a different machine entirely
    worker_two.extractor = _Fixed(yes)
    restore(worker_two, open_sealed(ticket))
    turn = worker_two.handle("yes")

    assert turn.changed_the_book, "the 'yes' should have booked the offer"


class _Fixed:
    """A stand-in reader that always returns the same reading, so no API call is made."""

    name = "test"
    last_used = "test"

    def __init__(self, extraction):
        self._extraction = extraction

    def extract(self, message, history=()):
        return self._extraction.model_copy(update={"raw_message": message,
                                                   "backend": "test"})


# ══════════════════════════════════════════════════════════════════════════════
# A patient cannot forge their own memory
# ══════════════════════════════════════════════════════════════════════════════

def test_an_edited_ticket_is_refused():
    """
    The attack: take a real ticket, change one character, send it back. If this were
    allowed, a patient could invent an offer and then accept it.
    """
    ticket = seal(_session())
    body, stamp = ticket.rsplit(".", 1)
    tampered = f"{body[:-2]}XY.{stamp}"

    with pytest.raises(TamperedState):
        open_sealed(tampered)


def test_a_ticket_with_a_forged_stamp_is_refused():
    ticket = seal(_session())
    body, _ = ticket.rsplit(".", 1)

    with pytest.raises(TamperedState):
        open_sealed(f"{body}.notarealstamp")


def test_rubbish_is_refused_rather_than_crashing():
    for rubbish in ("", "no-dot-here", "....", "hello.world"):
        with pytest.raises(TamperedState):
            open_sealed(rubbish)


def test_a_forged_offer_cannot_be_smuggled_in():
    """
    The specific G2 attack, written out in full.

    An attacker builds the exact memory that would make "yes" book an appointment,
    encodes it the same way we do, and sends it. Without the right secret they cannot
    produce a matching stamp, so the ticket never gets as far as the policy layer.
    """
    import base64, json

    forged = snapshot(_session())
    forged["pending_offer"] = {
        "kind": "create", "summary": "free appointment with Dr. George",
        "doctor_id": "d_george", "start": "2026-09-16T11:00:00+03:00",
        "appointment_id": None,
    }
    payload = json.dumps(forged, separators=(",", ":")).encode()
    body = base64.urlsafe_b64encode(payload).decode().rstrip("=")

    with pytest.raises(TamperedState):
        open_sealed(f"{body}.{'A' * 43}")
