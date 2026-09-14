"""
Check we actually match the assessment brief.

The README could claim "our output matches the example in the brief". These tests make it
something the test suite proves instead of something a human asserts.
"""

from clinikit.agent.policy import Action
from clinikit.agent.schema import Extraction, Intent


def test_output_contains_the_briefs_exact_example_fields():
    """
    The brief shows this example output:

        {
          "intent": "reschedule_appointment",
          "doctor": null,
          "preferred_date": "Wednesday",
          "preferred_time": "after 4 PM"
        }

    Those four field names must exist, spelled exactly that way.
    """
    produced = Extraction(
        intent=Intent.RESCHEDULE_APPOINTMENT, confidence=0.93,
        doctor=None, preferred_date="Wednesday", preferred_time="after 4 PM",
    ).model_dump(mode="json")

    for field in ("intent", "doctor", "preferred_date", "preferred_time"):
        assert field in produced, f"the brief's field {field!r} is missing"

    assert produced["intent"] == "reschedule_appointment"
    assert produced["doctor"] is None
    assert produced["preferred_date"] == "Wednesday"
    assert produced["preferred_time"] == "after 4 PM"


def test_all_seven_intents_from_the_brief_exist():
    for name in ("book_appointment", "reschedule_appointment", "cancel_appointment",
                 "ask_opening_hours", "ask_doctor_availability", "talk_to_human", "other"):
        assert name in {i.value for i in Intent}, f"the brief lists intent {name!r}"


def test_all_six_actions_from_the_brief_exist():
    for name in ("check_availability", "create_appointment", "reschedule_appointment",
                 "cancel_appointment", "handoff_to_human", "ask_for_more_information"):
        assert name in vars(Action).values(), f"the brief lists action {name!r}"
