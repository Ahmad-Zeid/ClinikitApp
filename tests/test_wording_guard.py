"""
The wording check: what the model is allowed to say on the patient's behalf.

BACKGROUND IN PLAIN ENGLISH
    The clinic writes a correct but stiff reply. A language model then rewrites it to
    sound human. The rewrite is not trusted: before the patient sees it, we check every
    detail in it against the facts we handed over. If anything does not match, the
    stiff-but-correct version is sent instead.

    These tests cover two things that slipped past that check in real runs.
"""

from clinikit.agent.phrasing import ReplyFacts, verify


def _facts(**overrides) -> ReplyFacts:
    base = dict(
        template="Understood — I won't book anything yet.",
        action="check_availability",
        changed_the_book=False,
    )
    base.update(overrides)
    return ReplyFacts(**base)


# ══════════════════════════════════════════════════════════════════════════════
# The hedge promise must survive the rewrite
# ══════════════════════════════════════════════════════════════════════════════

def test_hedged_reply_that_drops_the_promise_is_rejected():
    """
    The real failure this test exists for.

    The patient wrote "don't book anything yet". The model replied:

        "I understand you do not want to book anything yet, BUT here are the available
         times... Please let me know if you would like to secure one of these slots."

    Every fact in that sentence is true. It still breaks the promise, because it pushes
    towards the booking the patient just declined. The reply must keep a plain "I won't
    book anything".
    """
    pushy = ("Here are the available times with Dr. George Haddad. "
             "Please let me know if you would like to secure one of these slots.")

    assert verify(pushy, _facts(must_promise_no_action=True)) is not None
    # Without the hedge, the same sentence is a perfectly good reply.
    assert verify(pushy, _facts(must_promise_no_action=False)) is None


def test_hedged_reply_that_keeps_the_promise_is_accepted():
    kept = ("Understood, I won't book anything for now. "
            "Here is what is free if you would like to come back to me.")
    assert verify(kept, _facts(must_promise_no_action=True)) is None


def test_several_phrasings_of_the_promise_all_pass():
    """The check must not force one exact sentence, or every reply sounds identical."""
    for wording in (
        "I will not book anything yet.",
        "Nothing is booked, and nothing will be until you say so.",
        "I'll hold off on booking for now.",
        "I have not booked anything.",
    ):
        assert verify(wording, _facts(must_promise_no_action=True)) is None, wording


# ══════════════════════════════════════════════════════════════════════════════
# The model may not invent claims about the patient's records
# ══════════════════════════════════════════════════════════════════════════════

def test_invented_claim_about_the_patients_records_is_rejected():
    """
    The real failure this test exists for.

    To "I do NOT want to cancel my appointment" the model replied "we do not have an
    active appointment on our schedule for you". The patient did have one, with
    Dr. Karim. Nobody told the model either way; it filled the gap itself.
    """
    invented = ("I understand. We do not have an active appointment on our schedule "
                "for you. Could you let me know what you need?")
    facts = _facts(template="Understood — nothing has been changed.")

    assert verify(invented, facts) is not None


def test_the_same_claim_is_allowed_when_we_made_it_ourselves():
    """
    Sometimes the clinic really has checked and really has nothing on file. When our own
    approved reply says so, the model may say so too — otherwise a true reply would be
    thrown away for repeating us.
    """
    ours = _facts(template="I don't have an appointment on file for you.")
    assert verify("I don't have an appointment for you at the moment. What can I do?",
                  ours) is None


# ══════════════════════════════════════════════════════════════════════════════
# Negation must not weaken the check it was added to
# ══════════════════════════════════════════════════════════════════════════════

def test_real_completion_claims_are_still_rejected():
    """
    Teaching the check about the word "nothing" must not teach it to let real false
    claims through. These all assert a booking that never happened.
    """
    for lie in (
        "I've booked you in with Dr. George.",
        "Your appointment is confirmed.",
        "That is all set for you.",
        "We have you down for Wednesday.",
        "Your appointment has been cancelled.",
    ):
        assert verify(lie, _facts()) is not None, lie


def test_truthful_reassurance_is_allowed():
    for truth in (
        "Nothing is booked yet.",
        "No appointment has been made.",
        "Your appointment has not been cancelled.",
    ):
        assert verify(truth, _facts()) is None, truth


# ══════════════════════════════════════════════════════════════════════════════
# The doctor check must never switch itself off
# ══════════════════════════════════════════════════════════════════════════════

def test_choosing_between_appointments_still_lists_the_doctors():
    """
    The real failure this test exists for.

    Asked to cancel with "Dr. Karin" -- a typo for Dr. Karim -- the clinic offered the
    patient's two upcoming appointments to choose from. Those choices are appointments,
    which carry a doctor's id rather than a name, so no names were collected.

    An empty list of permitted doctors does not mean "no doctors allowed". It means the
    check has nothing to compare against, so it passes everything. The model echoed the
    typo back at the patient while the list underneath said Dr. Karim Nassar.

    So this test asserts the list is populated, which is what keeps the check alive.
    """
    from datetime import datetime

    from clinikit.agent import responder, tools
    from clinikit.agent.clinic import TIMEZONE, seeded_db
    from clinikit.agent.policy import Context, decide
    from clinikit.agent.schema import Extraction, Intent
    from clinikit.agent.session import _facts_for

    now = datetime(2026, 9, 14, 10, 0, tzinfo=TIMEZONE)
    ctx = Context(now=now, db=seeded_db("p_001"), patient_id="p_001")
    extraction = Extraction(intent=Intent.CANCEL_APPOINTMENT, confidence=0.9,
                            doctor="Dr. Karin")

    decision = decide(extraction, ctx)
    result = tools.execute(decision, ctx)
    reply = responder.respond(decision, result, ctx, extraction)
    facts = _facts_for(decision, result, reply, False, ctx)

    assert facts.doctors, "an empty list turns the doctor check off entirely"
    assert "Dr. Karim Nassar" in facts.doctors

    # And with the list populated, the typo is caught.
    assert verify("We have a few visits with Dr. Karin.", facts) is not None
    assert verify("We have a few visits with Dr. Karim Nassar.", facts) is None
