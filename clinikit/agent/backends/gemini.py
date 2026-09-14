"""
Reading a patient message with Google's Gemini.

HOW IT WORKS, IN PLAIN TERMS
    We send Gemini the patient's message along with a description of the exact form we
    want back. Gemini fills in that form. Because we hand it the form rather than asking
    for "some JSON", it cannot reply with a shape we did not ask for.

THE FORM IS THE SCHEMA
    We pass the `Extraction` class straight to Google as the required output shape.
    Google turns each field's `description=` text into instructions the model reads. So
    the comments explaining a field to a human ARE the instructions given to the model.
    They are the same words, so they can never drift apart and quietly disagree.

THREE THINGS THAT MAKE THIS MORE THAN AN API CALL
    1. Retry. Google's free tier returns "503 busy" fairly often. We saw it on our very
       first test, so this is a response to evidence rather than caution.
    2. Model fallback. If one model is unavailable we try the next. We also saw a model
       return 404 because it had been retired.
    3. Caching. Identical message plus identical prompt gives an identical answer, so we
       save it to disk. Re-running the evaluation over 70 messages then costs nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Sequence
from pathlib import Path

from ..clinic import DOCTORS
from ..schema import Extraction
from .base import Extractor, ExtractorUnavailable

# Bump this whenever the prompt below changes. It is part of the cache key, so old
# cached answers are ignored rather than silently reused against a new prompt.
PROMPT_VERSION = "v1"

# Tried in order. The first that responds wins.
#
# Order matters more than it looks. We originally listed the newest preview model first;
# it was returning 503 every time, so every single message wasted about 12 seconds failing
# through it before reaching a model that worked. The fallback chain did its job, but a
# working system that takes 15 seconds per reply is not a working system.
#
# Stable first, newest last.
DEFAULT_MODELS = ("gemini-3.5-flash", "gemini-3-flash-preview", "gemini-flash-latest")

# Hard ceiling on one extraction, across every retry and every model. Without this, a bad
# day at Google turns into a request that hangs for minutes. Better to fail quickly and
# let the caller fall back to the keyword reader.
REQUEST_BUDGET_SECONDS = 25.0

# The free tier allows roughly 15 requests a minute, so we leave 4 seconds between calls.
MIN_SECONDS_BETWEEN_CALLS = 4.0

MAX_RETRIES_PER_MODEL = 2

CACHE_DIR = Path(__file__).resolve().parents[3] / ".cache" / "gemini"


def _doctor_roster() -> str:
    return "\n".join(f"  - {d.full_name} ({d.specialty})" for d in DOCTORS)


SYSTEM_PROMPT = f"""\
You read messages that patients send to a medical clinic in Beirut, Lebanon, and turn
them into structured data. You do not reply to the patient and you never take any action.
Another part of the system decides what to do. Your only job is to report accurately what
the message says.

Doctors at this clinic:
{_doctor_roster()}

RULES

1. Copy dates and times EXACTLY as the patient wrote them. Put "tomorrow" in
   preferred_date, not a calendar date. Put "after 5" in preferred_time, not 17:00.
   You do not know today's date and must not try to work it out. Separate code does that.

2. For a reschedule, preferred_date is the NEW date they want. The appointment being
   moved away from goes in existing_appointment_phrase. In "move my appointment from
   Monday to Wednesday", preferred_date is "Wednesday" and
   existing_appointment_phrase is "Monday".

3. is_hedged means the patient explicitly told you NOT to act yet. "don't book anything
   yet", "just checking", "I'm not sure yet". A plain question such as "is Dr. George
   free tomorrow?" is NOT hedged. Politeness is not hedging. This field is used to block
   real bookings, so judge it on the words actually written.

4. Only record a doctor the patient actually named. "my doctor" names nobody, so leave
   doctor empty and set refers_to_previous_visit to true.

5. Messages may contain typos, or Lebanese Arabic written in Latin letters and numbers
   ("Arabizi"), for example "badde shouf" (I want to see), "bukra" (tomorrow),
   "3anjad" (really). Read them as normal language.

6. confidence is how sure you are of the intent, from 0 to 1. Be honest. A vague or
   confusing message should get a low score so the system knows to ask a question
   instead of guessing.

7. If the message is not about appointments or the clinic, use intent "other".
   Describing a symptom without asking for anything is usually book_appointment with
   modest confidence, and the symptom goes in reason_for_visit.
"""

_EXAMPLES = [
    (
        "I might want to see Dr. George tomorrow at 4, but don't book anything yet.",
        {"intent": "book_appointment", "confidence": 0.9, "doctor": "Dr. George",
         "preferred_date": "tomorrow", "preferred_time": "4",
         "existing_appointment_phrase": None, "is_hedged": True,
         "refers_to_previous_visit": False, "reason_for_visit": None,
         "reasoning": "names a doctor and time but explicitly defers booking"},
    ),
    (
        "cn u mve my apt frm mon to wed pls",
        {"intent": "reschedule_appointment", "confidence": 0.88, "doctor": None,
         "preferred_date": "wed", "preferred_time": None,
         "existing_appointment_phrase": "mon", "is_hedged": False,
         "refers_to_previous_visit": False, "reason_for_visit": None,
         "reasoning": "typo-heavy but clearly moving an appointment from Monday to Wednesday"},
    ),
    (
        "I want to see my doctor again for the same problem.",
        {"intent": "book_appointment", "confidence": 0.7, "doctor": None,
         "preferred_date": None, "preferred_time": None,
         "existing_appointment_phrase": None, "is_hedged": False,
         "refers_to_previous_visit": True, "reason_for_visit": "same problem as before",
         "reasoning": "wants an appointment but names no doctor, date or time"},
    ),
]


class GeminiExtractor(Extractor):
    """Language-model reader. Needs GEMINI_API_KEY."""

    name = "gemini"

    def __init__(
        self,
        models: Sequence[str] = DEFAULT_MODELS,
        api_key: str | None = None,
        use_cache: bool = True,
        min_interval: float = MIN_SECONDS_BETWEEN_CALLS,
    ) -> None:
        key = api_key or _load_key()
        if not key:
            raise ExtractorUnavailable(
                "GEMINI_API_KEY is not set. Put it in .env, or use the 'rules' backend."
            )
        try:
            from google import genai
        except ImportError as exc:
            raise ExtractorUnavailable("google-genai is not installed.") from exc

        self._client = genai.Client(api_key=key)
        self._models = tuple(models)
        self._use_cache = use_cache
        self._min_interval = min_interval
        self._last_call_at = 0.0
        self.last_model_used: str | None = None
        self.last_latency: float | None = None
        self.last_was_cached: bool = False

    # ---- availability ----

    @classmethod
    def is_configured(cls) -> bool:
        """True if this backend could run. Used to decide what to offer the user."""
        try:
            from google import genai  # noqa: F401
        except ImportError:
            return False
        return bool(_load_key())

    # ---- the main entry point ----

    def extract(self, message: str, history: Sequence[str] = ()) -> Extraction:
        cached = self._cache_get(message, history)
        if cached is not None:
            self.last_was_cached = True
            self.last_latency = 0.0
            return self._finalise(cached, message)

        self.last_was_cached = False
        contents = self._build_contents(message, history)

        deadline = time.monotonic() + REQUEST_BUDGET_SECONDS
        last_error: Exception | None = None
        for model in self._models:
            for attempt in range(MAX_RETRIES_PER_MODEL):
                if time.monotonic() > deadline:
                    raise ExtractorUnavailable(
                        f"Gave up after {REQUEST_BUDGET_SECONDS}s. "
                        f"Last error: {type(last_error).__name__}: {str(last_error)[:150]}"
                    ) from last_error
                try:
                    self._throttle()
                    started = time.monotonic()
                    response = self._client.models.generate_content(
                        model=model,
                        contents=contents,
                        config={
                            "response_mime_type": "application/json",
                            "response_schema": Extraction,
                            "system_instruction": SYSTEM_PROMPT,
                            "temperature": 0.0,
                        },
                    )
                    self.last_latency = time.monotonic() - started
                    self.last_model_used = model

                    result = Extraction.model_validate_json(response.text)
                    self._cache_put(message, history, result)
                    return self._finalise(result, message)

                except Exception as exc:  # noqa: BLE001 - inspected below
                    last_error = exc
                    if _is_transient(exc) and attempt + 1 < MAX_RETRIES_PER_MODEL:
                        time.sleep(2.0 * (attempt + 1))
                        continue
                    break  # move on to the next model

        raise ExtractorUnavailable(
            f"Every Gemini model failed. Last error: {type(last_error).__name__}: "
            f"{str(last_error)[:200]}"
        ) from last_error

    # ---- helpers ----

    def _build_contents(self, message: str, history: Sequence[str]) -> str:
        """
        Assemble what the model sees: worked examples, then any conversation so far,
        then the message to read.

        The examples are included in the prompt rather than fine-tuned in. For three
        examples that is by far the cheaper and more maintainable option, and changing
        one is a text edit rather than a retraining run.
        """
        parts = ["Examples of correct output:"]
        for example_message, example_json in _EXAMPLES:
            parts.append(f"\nMessage: {example_message}\nOutput: {json.dumps(example_json)}")

        if history:
            recent = "\n".join(f"  {line}" for line in list(history)[-6:])
            parts.append(
                "\nEarlier in this conversation (for context only — do not extract "
                f"from these):\n{recent}"
            )

        parts.append(f"\nNow read this message:\n{message}")
        return "\n".join(parts)

    def _throttle(self) -> None:
        """Wait if needed so we stay inside the free tier's requests-per-minute limit."""
        elapsed = time.monotonic() - self._last_call_at
        if self._last_call_at and elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call_at = time.monotonic()

    # ---- cache ----

    def _cache_key(self, message: str, history: Sequence[str]) -> str:
        blob = json.dumps(
            {"m": message, "h": list(history), "p": PROMPT_VERSION, "models": self._models},
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:32]

    def _cache_get(self, message: str, history: Sequence[str]) -> Extraction | None:
        if not self._use_cache:
            return None
        path = CACHE_DIR / f"{self._cache_key(message, history)}.json"
        if not path.exists():
            return None
        try:
            return Extraction.model_validate_json(path.read_text())
        except Exception:
            return None  # a corrupt cache entry should never break a run

    def _cache_put(self, message: str, history: Sequence[str], result: Extraction) -> None:
        if not self._use_cache:
            return
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = CACHE_DIR / f"{self._cache_key(message, history)}.json"
        path.write_text(result.model_dump_json(indent=1))


def _load_key() -> str | None:
    """Read GEMINI_API_KEY from the environment, loading .env first if present."""
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"]
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[3] / ".env")
    except ImportError:
        pass
    return os.environ.get("GEMINI_API_KEY")


def _is_transient(exc: Exception) -> bool:
    """
    Is this failure worth retrying?

    503 means the servers are busy and 429 means we sent too many requests — both pass
    on a second attempt. A 404 (model retired) or 400 (bad request) never will, so we
    move straight to the next model instead of wasting time.
    """
    text = str(exc)
    return any(code in text for code in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED"))
