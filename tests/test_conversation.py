"""
Tests for the things that make it feel like a conversation rather than a form.

Every one of these exists because the behaviour it checks was broken at some point and
somebody noticed while using the thing. The comment on each says what went wrong.

All fast: they drive the policy layer directly with a pinned clock, so no network and no
AI provider is involved.
"""

import pytest

from clinikit.agent.clinic import DOCTORS, specialty_for, suggest_doctors_for
from clinikit.agent.policy import (
    WRITE_ACTIONS,
    Action,
    Context,
    Doctor,
    FreeSlot,
    decide,
)
from clinikit.agent.responder import Reply, respond
from clinikit.agent.schema import Extraction, Intent
from clinikit.agent.tools import ToolResult


def E(**kw) -> Extraction:
    return Extraction(**{"confidence": 0.9, **kw})


# ══════════════════════════════════════════════════════════════════════════════
# A greeting is a greeting, not a failure
# ══════════════════════════════════════════════════════════════════════════════

def test_greeting_is_answered_not_investigated(now, db):
    """
    "hi" used to score 0.2 confidence, trip the low-confidence rule, and come back as
    "Please let us know a bit more about what you need" with a yellow warning banner.
    Saying hello is not a failure to understand.
    """
    decision = decide(E(intent=Intent.GREETING, confidence=0.2), Context(now=now, db=db))
    assert decision.action == Action.PROVIDE_INFORMATION
    assert decision.guarantee is None, "a greeting must not trip a safety rule"


def test_greeting_reply_actually_greets(now, db):
    ctx = Context(now=now, db=db)
    decision = decide(E(intent=Intent.GREETING, confidence=0.3), ctx)
    reply = respond(decision, ToolResult(True, decision.action, ""), ctx)
    assert "hello" in reply.as_text().lower()


# ══════════════════════════════════════════════════════════════════════════════
# Pointing at the right department
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("described,expected", [
    ("my heart has been racing", "Cardiologist"),
    ("chest feels tight", "Cardiologist"),
    ("a rash on my arm", "Dermatologist"),
    ("his son has a fever", "Paediatrician"),
    ("my daughter needs to be seen", "Paediatrician"),
    ("my back hurts", "General Practitioner"),
])
def test_symptoms_route_to_the_right_department(described, expected):
    assert specialty_for(described) == expected


@pytest.mark.parametrize("text", [
    "the person has a reason to come in",
    "this season i keep getting colds",
])
def test_matching_is_on_whole_words(text):
    """
    Matching was substring-based, so "son" inside "person", "reason" and "season" sent
    adults to the paediatrician. Whole words only.
    """
    assert specialty_for(text) == "General Practitioner"


def test_a_question_with_no_symptom_gets_no_doctor():
    """
    "Do you take my insurance?" came back recommending a dermatologist, because the
    router fell back to a doctor for any text at all.
    """
    specialty, doctors = suggest_doctors_for(
        "their insurance coverage", fall_back_to_gp=False
    )
    assert specialty is None and doctors == []


def test_suggesting_a_doctor_never_books(now, db):
    """Routing is read-only. Naming a department must never write to the book."""
    decision = decide(
        E(intent=Intent.BOOK_APPOINTMENT, reason_for_visit="my heart has been racing"),
        Context(now=now, db=db),
    )
    assert decision.action not in WRITE_ACTIONS
    assert decision.offer is None


def test_suggestion_names_a_department_and_says_nothing_clinical(now, db):
    """
    The one genuinely medical decision in the project. We name who covers it. We never
    say what the symptom might mean, how serious it is, or what to do about it.
    """
    ctx = Context(now=now, db=db)
    decision = decide(
        E(intent=Intent.ASK_CLINIC_INFO, request_summary="who treats heart problems"), ctx
    )
    text = respond(decision, ToolResult(True, decision.action, ""), ctx).as_text().lower()

    assert "cardiolog" in text
    for forbidden in ("you should", "you may have", "this could be", "sounds like",
                      "urgent", "serious", "diagnos"):
        assert forbidden not in text, f"reply gives clinical advice: {forbidden!r}"


# ══════════════════════════════════════════════════════════════════════════════
# Picking one of the options we listed
# ══════════════════════════════════════════════════════════════════════════════

def _four_slots(now):
    doctor = next(d for d in DOCTORS if d.id == "d_karim")
    base = now.replace(hour=10, minute=0) + (now.replace(hour=0) - now.replace(hour=0))
    from datetime import timedelta
    day = now + timedelta(days=(2 - now.weekday()) % 7 or 7)  # a Wednesday, Karim works
    return tuple(
        FreeSlot(day.replace(hour=h, minute=m, second=0, microsecond=0), doctor)
        for h, m in [(10, 0), (10, 30), (11, 0), (12, 0)]
    )


@pytest.mark.parametrize("reference,expected_hour,expected_minute", [
    ("the first one", 10, 0),
    ("option 2", 10, 30),
    ("the 11am one", 11, 0),
    ("the last one", 12, 0),
])
def test_references_to_a_listed_option_resolve(reference, expected_hour, expected_minute, now, db):
    """
    We showed four times and then could not accept "the first one" -- the patient was
    asked to start again while looking at our own list.
    """
    options = _four_slots(now)
    ctx = Context(now=now, db=db, offered_options=options)
    decision = decide(
        E(intent=Intent.BOOK_APPOINTMENT, option_reference=reference), ctx
    )
    assert decision.offer is not None, f"{reference!r} did not resolve"
    assert decision.offer.start.hour == expected_hour
    assert decision.offer.start.minute == expected_minute


def test_picking_an_option_still_needs_confirmation(now, db):
    """Choosing from a list is not consent to book. G2 applies here too."""
    ctx = Context(now=now, db=db, offered_options=_four_slots(now))
    decision = decide(E(intent=Intent.BOOK_APPOINTMENT, option_reference="the first one"), ctx)
    assert decision.action not in WRITE_ACTIONS


@pytest.mark.parametrize("reference", ["the 7pm one", "option 99", "banana"])
def test_an_unresolvable_reference_asks_rather_than_guesses(reference, now, db):
    """Guessing which appointment they meant is how you book the wrong one."""
    ctx = Context(now=now, db=db, offered_options=_four_slots(now))
    decision = decide(E(intent=Intent.BOOK_APPOINTMENT, option_reference=reference), ctx)
    assert decision.action == Action.ASK_FOR_MORE_INFORMATION
    assert decision.offer is None


def test_a_specific_choice_beats_a_bare_yes(now, db):
    """
    "The second one" is often read as a confirmation. The confirm rule used to run first,
    find nothing pending, and reply "I don't have anything waiting for a yes" -- while
    the patient was looking at the numbered list we had just sent.
    """
    ctx = Context(now=now, db=db, offered_options=_four_slots(now), pending_offer=None)
    decision = decide(
        E(intent=Intent.CONFIRM, option_reference="the second one"), ctx
    )
    assert decision.offer is not None
    assert decision.offer.start.minute == 30


# ══════════════════════════════════════════════════════════════════════════════
# Replies have parts, and the facts are not one of the rewritable ones
# ══════════════════════════════════════════════════════════════════════════════

def test_a_reply_assembles_its_parts_in_order():
    reply = Reply(
        acknowledgement="Sorry to hear that.",
        opening="Dr. Karim Nassar is our cardiologist.",
        body=("1. Wednesday at 10:00", "2. Wednesday at 10:30"),
        closing="Would either suit?",
    )
    text = reply.as_text()
    assert text.index("Sorry to hear that.") < text.index("1. Wednesday")
    assert text.index("1. Wednesday") < text.index("Would either suit?")


def test_options_are_numbered_not_bulleted(now, db):
    """
    Numbering is not decoration. "The first one" is only answerable because the list is
    numbered -- with bullets there is nothing for the patient to refer to.
    """
    ctx = Context(now=now, db=db)
    decision = decide(E(intent=Intent.BOOK_APPOINTMENT, doctor="Dr. Khoury"), ctx)
    reply = respond(decision, ToolResult(True, decision.action, ""), ctx)
    assert reply.body, "expected a list of doctors to choose from"
    assert reply.body[0].startswith("1.")
    assert not any(line.strip().startswith("•") for line in reply.body)


def test_nothing_that_is_asked_about_is_written(now, db):
    """
    Every branch that asks a question must leave the appointment book alone. Checked
    across the whole reply surface rather than one case at a time.
    """
    before = [(a.id, a.start, a.status) for a in db.appointments]
    for intent in list(Intent):
        for described in (None, "my heart is racing", "a rash"):
            ctx = Context(now=now, db=db)
            decide(E(intent=intent, reason_for_visit=described,
                     request_summary="something"), ctx)
    assert [(a.id, a.start, a.status) for a in db.appointments] == before


# ══════════════════════════════════════════════════════════════════════════════
# "Yes" to a question we asked, not just to an offer we made
# ══════════════════════════════════════════════════════════════════════════════

def test_yes_answers_the_question_we_just_asked(now, db):
    """
    We said "Would you like me to check when Dr. Nassar is free?" and the patient said
    "yes please" -- and got back "I'm not sure which request you're confirming", while
    answering the question we had just put to them.

    Only Offers were remembered (a specific appointment awaiting consent). A plain yes/no
    question had nothing behind it.
    """
    ctx = Context(now=now, db=db)
    asked = decide(
        E(intent=Intent.ASK_CLINIC_INFO, request_summary="who treats heart problems"), ctx
    )
    assert asked.question is not None, "asking about a department should leave a question open"

    answered = decide(
        E(intent=Intent.CONFIRM),
        Context(now=now, db=db, pending_question=asked.question),
    )
    assert answered.action == Action.CHECK_AVAILABILITY
    assert answered.action not in WRITE_ACTIONS


def test_yes_to_a_question_can_never_book(now, db):
    """
    Saying yes to "shall I look something up" must never book anything. A question is not
    an offer, and only an offer can lead to a write.
    """
    from clinikit.agent.policy import PendingQuestion
    looking_up = PendingQuestion(kind="check_availability", summary="checking availability",
                                 doctor_ids=("d_karim",))
    decision = decide(E(intent=Intent.CONFIRM),
                      Context(now=now, db=db, pending_question=looking_up))
    assert decision.action not in WRITE_ACTIONS


def test_yes_with_neither_a_question_nor_an_offer_still_asks(now, db):
    """The original G2 behaviour must survive: a yes out of nowhere books nothing."""
    decision = decide(E(intent=Intent.CONFIRM), Context(now=now, db=db))
    assert decision.action == Action.ASK_FOR_MORE_INFORMATION
    assert decision.guarantee == "G2"


# ══════════════════════════════════════════════════════════════════════════════
# Replies that made no sense for the situation
# ══════════════════════════════════════════════════════════════════════════════

def test_declining_a_list_does_not_talk_about_booking(now, db):
    """
    We showed four free times, the patient said "hmm, not really", and the reply was
    "No problem, I won't book that" -- nothing was being booked. The template was wrong
    for the situation, and the model dutifully reworded a wrong sentence.
    """
    from clinikit.agent.policy import FreeSlot
    options = _four_slots(now)
    ctx = Context(now=now, db=db, offered_options=options, pending_offer=None)
    decision = decide(E(intent=Intent.DENY), ctx)

    assert decision.problem == "declined_options"
    reply = respond(decision, ToolResult(True, decision.action, ""), ctx).as_text().lower()
    assert "book" not in reply, f"must not mention booking; said: {reply!r}"


def test_declining_an_actual_offer_does_mention_it(now, db):
    """The other side: if there WAS an offer, saying nothing was booked is the point."""
    from clinikit.agent.policy import Offer
    offer = Offer(kind="create", summary="Dr. George on Wednesday at 11:00",
                  doctor_id="d_george", start=now)
    decision = decide(E(intent=Intent.DENY), Context(now=now, db=db, pending_offer=offer))
    assert decision.problem == "declined_offer"


@pytest.mark.parametrize("message,is_move", [
    ("can i do this thursday at 12 pm?", False),
    ("can i come in friday", False),
    ("book me thursday", False),
    ("move my appointment to friday", True),
    ("can i change it to monday", True),
    ("push it back a week", True),
])
def test_only_a_real_reschedule_looks_for_an_appointment(message, is_move, now, db):
    """
    "can i do this thursday at 12 pm?" was read as a reschedule -- the word "this" was
    enough -- so a patient asking for a NEW appointment was shown their existing ones and
    asked which they meant.
    """
    from clinikit.agent.policy import _sounds_like_moving_something
    extraction = E(intent=Intent.RESCHEDULE_APPOINTMENT, raw_message=message)
    assert _sounds_like_moving_something(extraction) is is_move


def test_model_date_phrase_does_not_force_a_reschedule(now, db):
    """
    The live bug: Gemini labelled "can i do this thursday at 12 pm?" as a reschedule and
    put "this thursday" in existing_appointment_phrase. Policy trusted that field and
    listed the patient's existing appointments instead of offering the new slot.
    """
    from clinikit.agent.policy import _sounds_like_moving_something

    extraction = E(
        intent=Intent.RESCHEDULE_APPOINTMENT,
        raw_message="can i do this thursday at 12 pm?",
        preferred_date="this thursday",
        preferred_time="12 pm",
        existing_appointment_phrase="this thursday",
        doctor="Dr. George",
    )
    assert _sounds_like_moving_something(extraction) is False

    decision = decide(extraction, Context(now=now, db=db, known_slots={"doctor": "Dr. George"}))
    assert "which_appointment" not in decision.missing
    assert decision.action in (
        Action.ASK_FOR_MORE_INFORMATION,
        Action.CHECK_AVAILABILITY,
    )


def test_decline_then_new_time_does_not_list_existing_appointments(now, db):
    """
    Full thread from the screenshot: ask for George, decline the list, ask for Thursday
    noon. Must never answer with "You have these coming up."
    """
    from clinikit.agent.session import Session
    from clinikit.agent.schema import Extraction as Ex

    session = Session(backend="rules", clock=lambda: now)

    # Drive the reader with hand-built extractions so we replay the exact LLM mistake.
    class Scripted:
        name = "scripted"
        def __init__(self, answers):
            self._answers = list(answers)
            self.last_used = None
        def extract(self, message, history=()):
            return self._answers.pop(0)

    session.extractor = Scripted([
        Ex(intent=Intent.ASK_DOCTOR_AVAILABILITY, confidence=0.9,
           doctor="Dr. George", preferred_date="tomorrow", preferred_time="afternoon",
           raw_message="Can I see Dr. George tomorrow afternoon?", backend="scripted"),
        Ex(intent=Intent.DENY, confidence=0.9,
           raw_message="hmm, not really", backend="scripted"),
        Ex(intent=Intent.RESCHEDULE_APPOINTMENT, confidence=0.85,
           doctor=None, preferred_date="this thursday", preferred_time="12 pm",
           existing_appointment_phrase="this thursday",
           raw_message="can i do this thursday at 12 pm?", backend="scripted"),
    ])

    session.handle("Can I see Dr. George tomorrow afternoon?")
    session.handle("hmm, not really")
    assert session.offered_options == ()

    turn = session.handle("can i do this thursday at 12 pm?")
    reply = turn.reply.lower()
    assert "which one did you mean" not in reply
    assert "you have these coming up" not in reply
    assert "which_appointment" not in turn.decision.missing


@pytest.mark.parametrize("message", ["thanks", "thank you", "bye", "ok", "alright"])
def test_a_courtesy_is_not_a_request(message, now, db):
    """
    "thanks" was handed to a colleague -- "I'm passing on your question about thanking
    us" -- which is a strange way to end a conversation that went perfectly well.
    """
    decision = decide(E(intent=Intent.OTHER, raw_message=message,
                        request_summary="thanking the clinic"), Context(now=now, db=db))
    assert decision.topic == "courtesy"
    assert decision.action != Action.HANDOFF_TO_HUMAN


def test_a_courtesy_after_an_offer_still_confirms_it(now, db):
    """
    The distinction that matters: "ok" on its own is a sign-off, but "ok" right after we
    offer an appointment is consent. Getting this backwards either books nothing or
    books something nobody agreed to.
    """
    from clinikit.agent.policy import Offer
    offer = Offer(kind="create", summary="Dr. George on Wednesday at 11:00",
                  doctor_id="d_george",
                  start=now.replace(hour=11, minute=0) + __import__("datetime").timedelta(days=1))
    decision = decide(E(intent=Intent.CONFIRM, raw_message="ok"),
                      Context(now=now, db=db, pending_offer=offer))
    assert decision.action == Action.CREATE_APPOINTMENT


@pytest.mark.parametrize("phrase,weekday", [
    ("thursday", 3),
    ("this thursday", 3),
    ("coming friday", 4),
    ("on monday", 0),
    ("this coming saturday", 5),
    ("next thursday", 3),
])
def test_weekday_prefixes_resolve(phrase, weekday, now):
    """
    "can i do this thursday at 12?" came back as "no usable date was given". The reader
    had correctly returned "this thursday"; the arithmetic only understood "thursday" and
    "next thursday", so it threw a perfectly good answer away.

    Checked by weekday rather than by a fixed offset, so the test says what it means:
    the phrase must land on the right DAY OF THE WEEK, on or after today.
    """
    from clinikit.agent.temporal import resolve_date

    days, problems = resolve_date(phrase, now)
    assert not problems, f"{phrase!r} produced {problems}"
    assert days, f"{phrase!r} resolved to nothing"
    assert days[0].weekday() == weekday
    assert days[0] >= now.date()


def test_next_weekday_is_further_out_than_the_bare_one(now):
    """"next thursday" must mean a different Thursday from "thursday"."""
    from clinikit.agent.temporal import resolve_date

    plain, _ = resolve_date("thursday", now)
    later, _ = resolve_date("next thursday", now)
    assert later[0] > plain[0]
