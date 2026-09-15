"""
Proof that the seven safety guarantees hold.

These are the most important tests in the project. Everything else checks that the agent
is helpful; these check that it cannot cause harm.

A guarantee written in a README is a promise. A guarantee with a test is a fact.
"""

import itertools
import json
from datetime import timedelta
from pathlib import Path

import pytest

from clinikit.agent.backends import available_backends, get_extractor
from clinikit.agent.backends.base import ExtractorUnavailable
from clinikit.agent.policy import (
    WRITE_ACTIONS,
    Action,
    Context,
    Offer,
    decide,
)
from clinikit.agent.schema import Extraction, Intent
from clinikit.agent.session import Session

TESTSET = Path(__file__).resolve().parents[1] / "eval" / "testset.jsonl"
CASES = [json.loads(l) for l in TESTSET.read_text().splitlines() if l.strip()]


# ══════════════════════════════════════════════════════════════════════════════
# G1 — a patient who says "not yet" never gets a booking
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("intent", list(Intent))
def test_G1_hedged_never_writes_whatever_the_intent(intent, now, db):
    """
    The hedge blocks writes no matter what the reader decided the intent was.

    This is the heart of the design. Two Gemini models disagreed about the intent of the
    brief's ambiguous example but both spotted the hedge, so safety must not depend on
    the intent. Running every intent through proves it does not.
    """
    extraction = Extraction(intent=intent, confidence=0.99, doctor="Dr. George",
                            preferred_date="tomorrow", preferred_time="11", is_hedged=True)

    # WITH AN OFFER ALREADY WAITING.
    #
    # This used to pass an empty Context, and passing it proved less than it looked.
    # With nothing pending, a hedged "yes" fell through the confirmation branch to
    # "there is nothing to confirm" -- so the single dangerous combination in the whole
    # policy layer, a hedged confirmation against a live offer, was never once exercised.
    # It wrote, and this test watched it happen and reported success.
    offer = Offer(
        kind="create",
        summary="Dr. George Haddad tomorrow at 11:00",
        doctor_id="d_george",
        start=now.replace(hour=11, minute=0) + timedelta(days=1),
    )
    for context in (Context(now=now, db=db),
                    Context(now=now, db=db, pending_offer=offer)):
        decision = decide(extraction, context)
        assert decision.action not in WRITE_ACTIONS, (
            f"{intent.value} + is_hedged wrote {decision.action}"
        )


def test_G1_hedge_beats_maximum_confidence(now, db):
    """Even a totally confident booking request is blocked by an explicit 'not yet'."""
    extraction = Extraction(intent=Intent.BOOK_APPOINTMENT, confidence=1.0,
                            doctor="Dr. George", preferred_date="tomorrow",
                            preferred_time="11", is_hedged=True)
    assert decide(extraction, Context(now=now, db=db)).action not in WRITE_ACTIONS


# ══════════════════════════════════════════════════════════════════════════════
# G2 — nothing is written without confirming an offer WE made
# ══════════════════════════════════════════════════════════════════════════════

def test_G2_yes_with_nothing_pending_does_not_write(now, db):
    ctx = Context(now=now, db=db, pending_offer=None)
    decision = decide(Extraction(intent=Intent.CONFIRM, confidence=0.99), ctx)
    assert decision.action not in WRITE_ACTIONS
    assert decision.guarantee == "G2"


def test_G2_a_first_request_never_books_immediately(now, db):
    """A perfectly valid booking request still only produces an offer, never a booking."""
    extraction = Extraction(intent=Intent.BOOK_APPOINTMENT, confidence=0.99,
                            doctor="Dr. George", preferred_date="tomorrow",
                            preferred_time="11")
    decision = decide(extraction, Context(now=now, db=db))
    assert decision.action not in WRITE_ACTIONS
    assert decision.offer is not None


def test_G2_confirmation_after_our_offer_does_write(now, db):
    """The other side of the rule: with a real offer pending, 'yes' must work."""
    ctx = Context(now=now, db=db)
    first = decide(Extraction(intent=Intent.BOOK_APPOINTMENT, confidence=0.99,
                              doctor="Dr. George", preferred_date="tomorrow",
                              preferred_time="11"), ctx)
    second = decide(Extraction(intent=Intent.CONFIRM, confidence=0.99),
                    Context(now=now, db=db, pending_offer=first.offer))
    assert second.action == Action.CREATE_APPOINTMENT


def test_G2_offer_of_an_unknown_kind_is_refused(now, db):
    """A malformed offer must be rejected rather than acted on."""
    bogus = Offer(kind="drop_all_tables", summary="nope")
    ctx = Context(now=now, db=db, pending_offer=bogus)
    assert decide(Extraction(intent=Intent.CONFIRM, confidence=0.99), ctx).action \
        not in WRITE_ACTIONS


# ══════════════════════════════════════════════════════════════════════════════
# G3 — never book a doctor who is unknown or ambiguous
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("name", ["Dr. Khoury", "khoury", "Dr. House", "nobody", None])
def test_G3_unknown_or_ambiguous_doctor_never_writes(name, now, db):
    extraction = Extraction(intent=Intent.BOOK_APPOINTMENT, confidence=0.99,
                            doctor=name, preferred_date="tomorrow", preferred_time="11")
    decision = decide(extraction, Context(now=now, db=db))
    assert decision.action not in WRITE_ACTIONS
    assert decision.offer is None


# ══════════════════════════════════════════════════════════════════════════════
# G4 / G5 — never book outside hours, in the past, or on top of another booking
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("date_phrase,time_phrase,why", [
    ("sunday", "11", "clinic closed on Sunday"),
    ("tomorrow", "7", "nobody is open at 7"),
    ("today", "9", "09:00 has already passed at 10:00"),
    ("01/01/2020", "11", "date in the past"),
])
def test_G4_G5_invalid_times_never_produce_an_offer(date_phrase, time_phrase, why, now, db):
    extraction = Extraction(intent=Intent.BOOK_APPOINTMENT, confidence=0.99,
                            doctor="Dr. George", preferred_date=date_phrase,
                            preferred_time=time_phrase)
    decision = decide(extraction, Context(now=now, db=db))
    assert decision.action not in WRITE_ACTIONS, why
    assert decision.offer is None, why


def test_G4_doctor_own_hours_are_respected(now, db):
    """Dr. George finishes at 16:00, so 16:00 must be refused even though the clinic is open."""
    extraction = Extraction(intent=Intent.BOOK_APPOINTMENT, confidence=0.99,
                            doctor="Dr. George", preferred_date="tomorrow",
                            preferred_time="4pm")
    decision = decide(extraction, Context(now=now, db=db))
    assert decision.offer is None
    assert decision.guarantee == "G4"


# ══════════════════════════════════════════════════════════════════════════════
# G6 — low confidence never acts
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("confidence", [0.0, 0.1, 0.3, 0.54])
@pytest.mark.parametrize("intent", [Intent.BOOK_APPOINTMENT, Intent.CANCEL_APPOINTMENT,
                                    Intent.RESCHEDULE_APPOINTMENT, Intent.CONFIRM])
def test_G6_low_confidence_never_writes(confidence, intent, now, db):
    extraction = Extraction(intent=intent, confidence=confidence, doctor="Dr. George",
                            preferred_date="tomorrow", preferred_time="11")
    ctx = Context(now=now, db=db,
                  pending_offer=Offer(kind="create", summary="x",
                                      doctor_id="d_george", start=now))
    assert decide(extraction, ctx).action not in WRITE_ACTIONS


# ══════════════════════════════════════════════════════════════════════════════
# G7 — the policy layer cannot act, only name an action
# ══════════════════════════════════════════════════════════════════════════════

def test_G7_deciding_never_changes_the_appointment_book(now, db):
    """
    Run every intent through the policy layer and check the book is untouched.

    decide() returns the word "create_appointment". Something else has to choose to act
    on it. This test proves that separation actually holds rather than merely being the
    intention.
    """
    before = [(a.id, a.start, a.status) for a in db.appointments]
    for intent, hedged in itertools.product(list(Intent), [True, False]):
        decide(Extraction(intent=intent, confidence=0.99, doctor="Dr. George",
                          preferred_date="tomorrow", preferred_time="11",
                          is_hedged=hedged), Context(now=now, db=db))
    assert [(a.id, a.start, a.status) for a in db.appointments] == before


# ══════════════════════════════════════════════════════════════════════════════
# The whole test set, through every backend
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.live
@pytest.mark.parametrize("backend", available_backends())
def test_no_test_case_ever_changes_the_appointment_book(backend, now):
    """
    Every case in eval/testset.jsonl is marked must_not_write.

    None of them contains a confirmation, so none may result in a booking, a move or a
    cancellation -- including the ten adversarial ones that actively try to cause one.
    This runs against each available backend, so a change to the prompt cannot quietly
    break safety.
    """
    extractor = get_extractor(backend)
    offenders = []
    for case in CASES:
        session = Session(backend=backend, clock=lambda: now)
        session.extractor = extractor
        try:
            turn = session.handle(case["message"])
        except ExtractorUnavailable as exc:
            # Google's free tier returns 503 fairly often. A test that fails because
            # somebody else's servers are busy is noise, and noisy tests get ignored
            # until a real failure is ignored with them. Skip and say why -- this is
            # deliberately NOT treated as a pass.
            pytest.skip(f"{backend} unreachable, cannot verify: {str(exc)[:80]}")
        if turn.changed_the_book:
            offenders.append(f"[{case['id']}] {turn.decision.action}: {case['message'][:60]}")
    assert not offenders, "appointment book changed on:\n" + "\n".join(offenders)
