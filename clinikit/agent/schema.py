"""
The extraction contract.

This module defines the ONLY shape of data that may pass from the language-understanding
layer into the rest of the system. Every backend — rule-based or LLM — must produce an
`Extraction`, and nothing downstream accepts anything else.

Why this matters: the backends are interchangeable precisely because they all speak this
one language. Swapping Gemini for regex changes nothing downstream, because downstream
never sees a backend, only an `Extraction`.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Intent(str, Enum):
    """
    What the patient is trying to do.

    The first seven are taken verbatim from the CliniKit brief. The last two are
    conversational rather than clinical: they exist because a multi-turn agent must be
    able to recognise "yes" and "no" as answers to a question it just asked. They are
    deliberately marked as additions rather than quietly mixed in.
    """

    # --- the seven from the brief ---
    BOOK_APPOINTMENT = "book_appointment"
    RESCHEDULE_APPOINTMENT = "reschedule_appointment"
    CANCEL_APPOINTMENT = "cancel_appointment"
    ASK_OPENING_HOURS = "ask_opening_hours"
    ASK_DOCTOR_AVAILABILITY = "ask_doctor_availability"
    TALK_TO_HUMAN = "talk_to_human"
    OTHER = "other"

    # --- additions, all deliberate and all explained ---------------------
    # The seven above are the brief's list, untouched. These four exist because a real
    # conversation needs them, and each would otherwise be forced into "other", where the
    # only possible reply is "could you tell me more" -- a dead end.
    CONFIRM = "confirm"            # "yes", answering a question we just asked
    DENY = "deny"                  # "no thanks", same
    GREETING = "greeting"          # "hi" -- a greeting is not a failure to understand
    ASK_CLINIC_INFO = "ask_clinic_info"  # where are you, parking, what does Dr X treat


class Extraction(BaseModel):
    """
    Structured information pulled out of a single patient message.

    Design notes:

    1. Almost every field is Optional. A patient message is usually incomplete, and a
       schema that cannot represent "I don't know" forces the extractor to invent
       things. Missing information must be expressible, not guessable.

    2. Dates and times are captured as *phrases*, not as dates. "tomorrow afternoon"
       stays as the literal string "tomorrow afternoon". Converting it into a real
       datetime is the job of `temporal.py`, using the clinic's clock and calendar.
       Language models are unreliable at date arithmetic and have no idea what today
       is; Python is perfect at it. So the model reports what it *read*, and
       deterministic code decides what it *means*.

    3. `is_hedged` is the safety-critical field. See the class docstring below.
    """

    intent: Intent = Field(
        description="The single best label for what the patient wants."
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "How sure the extractor is about `intent`, from 0 to 1. Used by the policy "
            "layer to decide whether to act, ask a clarifying question, or hand off to "
            "a human."
        ),
    )

    # ---------- who ----------
    # NOTE: field names below marked (brief) match the example output in the CliniKit
    # brief verbatim, so their example JSON is a literal subset of ours.
    doctor: Optional[str] = Field(
        default=None,
        description=(
            "The doctor mentioned, exactly as the patient wrote it, e.g. 'Dr. George' "
            "or 'george'. Do not correct spelling or guess a doctor who was not "
            "mentioned. Matching this to a real doctor happens later."
        ),
    )

    # ---------- when (as written, not as resolved) ----------
    preferred_date: Optional[str] = Field(
        default=None,
        description=(
            "The date the patient wants, copied verbatim: 'tomorrow', 'next week', "
            "'Friday'. For a reschedule this is the NEW date they are moving to."
        ),
    )
    preferred_time: Optional[str] = Field(
        default=None,
        description=(
            "The time the patient wants, copied verbatim: 'after 5', '4', "
            "'in the afternoon'. For a reschedule this is the NEW time."
        ),
    )

    # ---------- which existing appointment ----------
    existing_appointment_phrase: Optional[str] = Field(
        default=None,
        description=(
            "How the patient referred to an appointment they already have, e.g. "
            "'my Monday appointment' or 'my appointment with Dr. Karim'. Only for "
            "reschedule and cancel. In 'move my appointment from Monday to "
            "Wednesday', this is 'Monday' and `preferred_date` is 'Wednesday'."
        ),
    )

    # ---------- flags that change what we are allowed to do ----------
    is_hedged: bool = Field(
        default=False,
        description=(
            "True if the patient explicitly signalled they do NOT want action taken "
            "yet: 'don't book anything yet', 'just checking', 'I might', 'not sure "
            "yet'. This is about explicit deferral, not politeness or uncertainty in "
            "tone. A plain question like 'is Dr. George free tomorrow?' is NOT hedged."
        ),
    )

    refers_to_previous_visit: bool = Field(
        default=False,
        description=(
            "True if the patient referenced an earlier visit without naming details, "
            "e.g. 'my doctor', 'the same problem as last time'. Signals that we need "
            "patient history we may not have."
        ),
    )

    reason_for_visit: Optional[str] = Field(
        default=None,
        description="Any stated medical reason, e.g. 'follow-up', 'sore throat'.",
    )

    option_reference: Optional[str] = Field(
        default=None,
        description=(
            "If the patient is picking from a list we just showed them, copy how they "
            "referred to it: 'the first one', 'the 9am one', 'option 2', 'the second'. "
            "Leave empty if they are not choosing from a list."
        ),
    )

    # ---------- what they actually wanted, in your own words ----------
    request_summary: str = Field(
        default="",
        description=(
            "One short line saying what the patient wants, in your own words, with no "
            "restrictions. Fill this in EVERY time, including when intent is 'other'. "
            "Examples: 'asking whether the clinic accepts their insurance', 'wants a "
            "prescription refill', 'asking where the clinic is'. This is how information "
            "that does not fit the intent list still reaches a human."
        ),
    )

    # ---------- transparency ----------
    reasoning: str = Field(
        default="",
        description=(
            "One short sentence on why this intent was chosen. Not shown to the "
            "patient — it exists so a human reviewing a transcript can see why the "
            "system did what it did."
        ),
    )

    # ---------- provenance (filled in by us, not by the model) ----------
    raw_message: str = Field(
        default="",
        description="The original patient message, kept for logging and evaluation.",
    )
    backend: str = Field(
        default="",
        description="Which extractor produced this, e.g. 'gemini' or 'rules'.",
    )
