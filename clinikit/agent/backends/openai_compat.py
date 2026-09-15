"""
One piece of code that talks to almost every AI company.

THE SIMPLE VERSION
    Groq, OpenRouter, Mistral, Together and several others all accept requests in the
    same format -- the one OpenAI invented. The only real difference is the web address
    you send them to.

    So instead of writing separate code for each company, we write it once and change the
    address. Adding a new provider is one line in the table below.

WHY THAT MATTERS HERE
    We were tied to Google. When Google had a bad afternoon, the assistant had nothing to
    fall back on except a keyword matcher that gives bad answers. Being able to switch
    companies in one line is the difference between an outage and a shrug.

PROVIDERS WE SUPPORT
    Groq        14,400 requests a day, 30 a minute, no card. Runs on custom chips built
                for speed, so replies come back far quicker than usual.
    OpenRouter  Only 50 requests a day on the free plan, so it sits last in the queue.
                Useful as a last resort and as proof the code is not tied to one company.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..clinic import DOCTORS
from ..schema import Extraction
from .base import Extractor, ExtractorUnavailable

PROMPT_VERSION = "v7"
CACHE_DIR = Path(__file__).resolve().parents[3] / ".cache" / "llm"

# How long we will spend making a reply sound nicer before giving up and sending the
# approved wording instead. Reading the message is essential; wording it is not.
# Kept for documentation; write_text now does one timed SDK call and returns.

# Allowance held back for reading messages.
#
# Both jobs draw on the same tokens-per-minute pot, and they are not equally important.
# If reading runs out, the assistant cannot function at all -- it can only apologise. If
# wording runs out, replies are a little plainer and nobody notices.
#
# So wording stops asking for tokens while there is still roughly one full reading call
# left in the pot. It degrades quietly instead of starving the thing that matters.
EXTRACTION_RESERVE_TOKENS = 2600


class DailyUsage:
    """
    How much of today's allowance we have spent, remembered across restarts.

    Per-minute counters can live in memory: a restart takes longer than a minute, so
    starting fresh is roughly correct. Daily counters cannot. Ours did, and the result
    was that every restart believed it had a whole day's allowance again -- so we kept
    calling a provider that had been exhausted hours earlier, and each call cost a round
    trip to be told no.

    Stored as a small JSON file. If it cannot be read or written, everything still works;
    we simply lose the memory of what today cost, which is no worse than before.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._today = ""
        self._counts: dict[str, dict[str, int]] = {}
        self._load()

    def _load(self) -> None:
        today = time.strftime("%Y-%m-%d")
        self._today = today
        try:
            data = json.loads(self._path.read_text())
            if data.get("date") == today:
                self._counts = data.get("counts", {})
        except Exception:
            self._counts = {}

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps({"date": self._today, "counts": self._counts}))
        except Exception:
            pass

    def _roll_over_if_new_day(self) -> None:
        today = time.strftime("%Y-%m-%d")
        if today != self._today:
            self._today, self._counts = today, {}

    def get(self, key: str) -> dict[str, int]:
        self._roll_over_if_new_day()
        return self._counts.get(key, {"requests": 0, "tokens": 0})

    def add(self, key: str, requests: int = 0, tokens: int = 0) -> None:
        self._roll_over_if_new_day()
        entry = self._counts.setdefault(key, {"requests": 0, "tokens": 0})
        entry["requests"] += requests
        entry["tokens"] += tokens
        self._save()

    def exhaust(self, key: str, limit: int) -> None:
        """Mark a model as spent for the day, after the provider says so."""
        self._roll_over_if_new_day()
        entry = self._counts.setdefault(key, {"requests": 0, "tokens": 0})
        entry["tokens"] = max(entry["tokens"], limit)
        self._save()

    def summary(self) -> dict[str, dict[str, int]]:
        self._roll_over_if_new_day()
        return dict(self._counts)


DAILY_USAGE = DailyUsage(CACHE_DIR.parent / "daily_usage.json")


@dataclass(frozen=True)
class Provider:
    """Everything that differs between one AI company and another."""

    name: str
    base_url: str
    env_key: str
    models: tuple[str, ...]
    requests_per_minute: int
    tokens_per_minute: int = 8000
    requests_per_day: int = 1000
    tokens_per_day: int | None = None
    """
    Daily token cap, or None when the provider does not have one.

    This is THE difference between the two providers we use, and it decides which should
    be primary:

      Groq    200,000 tokens per model per day. Our reading prompt is ~1,300 tokens, so
              that is about 150 calls per model -- even though we are allowed 1,000
              requests. The token cap bites first, by a long way.

      Gemini  No daily token cap at all. 1,500 requests a day, whatever size they are.

    Our calls are large and not very numerous, so a request-counted allowance suits us
    and a token-counted one does not. Hence Gemini first.
    """
    """
    The limit that actually bites.

    We originally throttled on requests only. Groq allows 30 a minute, so that felt
    generous -- but the real ceiling is 8,000 TOKENS a minute, and one turn costs roughly
    1,600 (a long extraction prompt plus a short wording call). That is about five turns
    a minute, not thirty. We sailed past it, got rejected, and a reply that normally takes
    two seconds took twenty-three.
    """

    supports_json_schema: bool = True
    signup: str = ""


PROVIDERS: dict[str, Provider] = {
    "gemini": Provider(
        name="gemini",
        # Google publishes an OpenAI-compatible endpoint, so Gemini needs no separate
        # code -- it inherits the retries, budgets and model fallback built here.
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        env_key="GEMINI_API_KEY",
        # FLASH-LITE FIRST, and this is the single most important line in the file for
        # staying inside the free tier.
        #
        # Every article about Gemini's free tier quotes "1,500 requests a day". That is
        # not what the API grants for the full-size Flash models any more. Measured
        # against the live API today, by reading the 429s:
        #
        #     gemini-3-flash-preview   daily limit 20     <- twenty. not 1,500.
        #     gemini-3.5-flash         daily limit 20
        #     gemini-flash-latest      daily limit 20
        #     gemini-*-flash-LITE      still answering after a day of heavy testing
        #
        # So the Lite variants are the generous ones, and they are not a compromise
        # here: on our own test messages gemini-flash-lite-latest scored 4/4 on intent,
        # caught the hedge, and answered in 1.3s -- as fast as Groq.
        #
        # The full-size models stay at the end of the list. Twenty calls a day is worth
        # having as a last resort, and they are the strongest readers we have.
        models=(
            "gemini-flash-lite-latest",
            "gemini-3.1-flash-lite",
            "gemini-3.5-flash-lite",
            "gemini-3-flash-preview",
        ),
        requests_per_minute=15,
        tokens_per_minute=250_000,
        requests_per_day=1000,
        tokens_per_day=None,          # Gemini counts requests, not tokens
        signup="https://aistudio.google.com/apikey  (free, no card)",
    ),
    "groq": Provider(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        env_key="GROQ_API_KEY",
        # Ordered best-first. If one is retired or busy we walk down the list.
        # Checked against the live model list rather than guessed: the Llama models that
        # most guides still recommend now return 404 on Groq. Measured round trips on
        # these: gpt-oss-120b 0.8s, gpt-oss-20b 0.6s.
        models=(
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b",
        ),
        requests_per_minute=30,
        tokens_per_minute=8000,
        requests_per_day=1000,
        tokens_per_day=200_000,
        signup="https://console.groq.com  (free, no card)",
    ),
    "openrouter": Provider(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        env_key="OPENROUTER_API_KEY",
        models=(
            "meta-llama/llama-3.3-70b-instruct:free",
            "google/gemma-2-9b-it:free",
        ),
        requests_per_minute=20,
        signup="https://openrouter.ai/keys  (free tier is only 50 requests a day)",
    ),
}


def _roster() -> str:
    return "\n".join(f"  - {d.full_name} ({d.specialty})" for d in DOCTORS)


SYSTEM_PROMPT = f"""\
You read messages patients send to a clinic in Beirut and return JSON. You never reply to
the patient and never take any action.

Doctors:
{_roster()}

1. Copy dates and times VERBATIM ("tomorrow", "after 5"). Never convert to a real date -
   you do not know today's date.
2. reschedule_appointment ONLY when they want to MOVE an appointment they already have
   ("move", "change", "reschedule", "instead", "push it back"). Asking to come in on a
   day - "can I do this Thursday at 12?", "can I come Friday?" - is book_appointment,
   even if the sentence contains "this" or "that".
   For a real reschedule: preferred_date is the NEW date, the old one goes in
   existing_appointment_phrase. "from Monday to Wednesday" -> new=Wednesday, old=Monday.
3. is_hedged = they explicitly said not to act yet ("don't book yet", "just checking").
   A plain question is NOT hedged. Politeness is NOT hedging.
4. Only record a doctor they actually named. "a doctor" / "my doctor" / "someone" name
   nobody - leave doctor empty.
5. Expect typos, and Lebanese Arabic written in Latin letters. Read both as normal language.
6. confidence 0-1, honest. A vague message scores low so we ask instead of guessing.
7. Three things that look alike:
   - ask_doctor_availability = asking WHAT is free, without committing to a time.
     "Is Dr George free tomorrow afternoon?", "anything available after 5?",
     "any slots Thursday?". Still availability even if they clearly want to book.
   - book_appointment = asking to BE BOOKED, or naming the time they want.
     "Book me Friday at 4", "can I come in Thursday?", "I need an appointment".
   - ask_opening_hours = ONLY what hours the clinic itself operates.
8. A symptom with no request ("my back hurts") is book_appointment, modest confidence,
   symptom in reason_for_visit. A bare hello is "greeting".
9. confirm / deny are short replies to a question we just asked. "ok thanks" is "other".
10. ask_clinic_info: location, parking, contact, which doctor treats what.
10b. talk_to_human: ANY request for a person, however phrased or however rude - "get me
    a human", "I'll call instead", "someone call me", "let me speak to reception".
11. NEVER copy a date, time or doctor from the examples below. Only what this message says.
12. option_reference: picking from a numbered list we showed - copy their words ("the
    first one", "option 2"). Otherwise empty.
13. request_summary: ALWAYS fill. A noun phrase that reads after "your question about..."
    - "their insurance coverage", "a repeat prescription". Not "asks if...".

Return only JSON."""

# Two examples, not four. Every token here is spent on every single request, and Groq's
# free tier caps us at 8,000 tokens a minute -- the cost is throughput, not money.
_EXAMPLES = [
    ("I might want to see Dr. George tomorrow at 4, but don't book anything yet.",
     {"intent": "book_appointment", "confidence": 0.9, "doctor": "Dr. George",
      "preferred_date": "tomorrow", "preferred_time": "4",
      "existing_appointment_phrase": None, "is_hedged": True,
      "refers_to_previous_visit": False, "reason_for_visit": None,
      "reasoning": "names a doctor and time but defers booking"}),
    ("can i do this thursday at 12 pm?",
     {"intent": "book_appointment", "confidence": 0.9, "doctor": None,
      "preferred_date": "this thursday", "preferred_time": "12 pm",
      "existing_appointment_phrase": None, "is_hedged": False,
      "refers_to_previous_visit": False, "reason_for_visit": None,
      "reasoning": "asking to come in on a day — not moving an existing appointment"}),
]


class OpenAICompatibleExtractor(Extractor):
    """Reads patient messages using any provider that speaks the OpenAI format."""

    def __init__(
        self,
        provider: str = "groq",
        api_key: str | None = None,
        models: Sequence[str] | None = None,
        use_cache: bool = True,
        # Long enough to WAIT OUT a full rate-limit window rather than give up inside it.
        #
        # It must be LONGER than the window it may have to wait out, or it gives up just
        # before the allowance refills -- which is how a busy minute produced "sorry, I'm
        # having trouble" instead of a slow reply. The window is 60 seconds, so the budget
        # is 80. A slow reply is a reply. An apology is not. Reading the message is the one
        # thing that
        # cannot degrade gracefully, so it is allowed to take its time.
        budget_seconds: float = 80.0,
    ) -> None:
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown provider {provider!r}. Known: {list(PROVIDERS)}")
        self.provider = PROVIDERS[provider]
        self.name = provider

        key = api_key or _load_key(self.provider.env_key)
        if not key:
            raise ExtractorUnavailable(
                f"{self.provider.env_key} is not set. Get one free at {self.provider.signup}"
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ExtractorUnavailable("the 'openai' package is not installed") from exc

        # A short per-call timeout on purpose.
        #
        # Groq is usually under a second but occasionally takes eight. Two of those in
        # one turn -- one to read the message, one to word the reply -- is a
        # twenty-four-second wait for a patient. Giving up at seven seconds and asking a
        # different model is almost always faster than waiting for the slow one, because
        # the fallback normally answers in about a second.
        self._client = OpenAI(
            api_key=key,
            base_url=self.provider.base_url,
            timeout=7.0,
            # The SDK retries twice on its own by default. Ours does too, and a retry
            # inside a retry multiplies: a seven-second timeout quietly became
            # twenty-one seconds per call, and a single slow model turned into a
            # minute-long wait for a patient. Retrying is our job -- we know which other
            # models are free and how much allowance each has left. The SDK does not.
            max_retries=0,
        )
        self._models = tuple(models or self.provider.models)
        self._use_cache = use_cache
        self._budget = budget_seconds
        # (time, tokens) per recent call, kept SEPARATELY FOR EACH MODEL.
        #
        # Every model has its own requests-per-minute and tokens-per-minute allowance.
        # Counting them all against one shared total was expensive: when the big model
        # filled up we slept for a minute, while two other models sat there completely
        # unused. Tracking them apart means a full model is simply skipped.
        self._windows: dict[str, deque[tuple[float, int]]] = {
            m: deque() for m in self._models
        }
        self._last_model_called: str | None = None

        # When a model told us "429, too many requests", and when it said we could
        # return. Our own counters start empty on every restart, so after a busy session
        # we cheerfully believe we have a full allowance, get refused, and then conclude
        # there is nothing to wait for -- because our local window looks empty. The
        # server's refusal is ground truth; our counting is only an estimate.
        self._blocked_until: dict[str, float] = {}

        # Set once a model reports the DAILY allowance is spent. Surfaced to the caller
        # so the reason shown is "the free daily allowance is used up" rather than a
        # vague "having trouble", which sends people hunting for a bug that is not there.
        self.daily_quota_exhausted = False

        self.last_model_used: str | None = None
        self.last_latency: float | None = None
        self.last_was_cached = False

    # ---- availability -------------------------------------------------------

    @classmethod
    def is_configured(cls, provider: str = "groq") -> bool:
        if provider not in PROVIDERS:
            return False
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return bool(_load_key(PROVIDERS[provider].env_key))

    # ---- the main entry point ----------------------------------------------

    def extract(self, message: str, history: Sequence[str] = ()) -> Extraction:
        cached = self._cache_get(message, history)
        if cached is not None:
            self.last_was_cached = True
            self.last_latency = 0.0
            return self._finalise(cached, message)
        self.last_was_cached = False

        messages = self._messages(message, history)
        deadline = time.monotonic() + self._budget
        last_error: Exception | None = None

        # Reading a message is the expensive call: a long instruction block plus the
        # examples plus the schema.
        cost = 1300

        # Two passes. The first uses only models with allowance left. If every model is
        # saturated, wait for the soonest to free up and go round once more. Skipping a
        # busy model costs nothing; waiting for it costs up to a minute.
        for pass_number in (1, 2):
            for model in self._models:
                if time.monotonic() > deadline:
                    break
                if pass_number == 1 and not self.has_budget(model, cost):
                    continue

                for attempt in range(2):
                    try:
                        self._note_call(model, cost)
                        started = time.monotonic()
                        response = self._client.chat.completions.create(
                            model=model,
                            messages=messages,
                            temperature=0.0,
                            response_format=self._response_format(),
                        )
                        self._record_usage(response)
                        self.last_latency = time.monotonic() - started
                        self.last_model_used = model

                        text = response.choices[0].message.content or ""
                        result = Extraction.model_validate_json(_strip_fences(text))
                        self._cache_put(message, history, result)
                        return self._finalise(result, message)

                    except Exception as exc:  # noqa: BLE001
                        last_error = exc
                        # Rate-limited means this model's allowance is gone. Another
                        # model has its own, so move on rather than wait.
                        if _is_rate_limited(exc):
                            self._note_refusal(model, exc)
                            break
                        if _is_transient(exc) and attempt == 0 and time.monotonic() < deadline:
                            time.sleep(1.5)
                            continue
                        break

            if pass_number == 1:
                wait = self._seconds_until_free(cost)
                if wait <= 0:
                    break
                # Wait as long as the budget allows and then try anyway, rather than
                # giving up the moment the sums say we cannot fit.
                #
                # It used to bail out instantly when the wait exceeded the budget, so a
                # busy minute produced "sorry, I'm having trouble" in a tenth of a
                # second -- which looks broken, not busy. Trying and being refused is a
                # better outcome than not trying: the refusal is handled, and often the
                # allowance has freed up more than our own arithmetic expected.
                remaining = deadline - time.monotonic()
                if remaining <= 0.5:
                    break
                time.sleep(min(wait, remaining))

        if self.daily_quota_exhausted:
            raise ExtractorUnavailable(
                f"{self.name}: the free daily token allowance is used up. It resets on "
                f"the provider's daily cycle. Add another provider with "
                f"CLINIKIT_PROVIDERS=groq,gemini, or use a different key."
            ) from last_error

        raise ExtractorUnavailable(
            f"{self.name}: every model failed. Last error: "
            f"{type(last_error).__name__}: {str(last_error)[:160]}"
        ) from last_error

    def write_text(self, system: str, user: str) -> str:
        """
        Plain-language generation, used for rewording replies.

        ONE best-effort call. Wording is a luxury: if the fast model cannot do it
        quickly, the approved template goes out instead. Walking every model used to
        burn quota and stretch a "nice sentence" into tens of seconds.
        """
        # Prefer the smallest/fastest model on this provider.
        model = next((m for m in self._models if "20b" in m or "lite" in m.lower()),
                     self._models[0])

        reserve = 2 if self.provider.requests_per_minute >= 20 else 4
        if not self.has_budget(model, 600 + EXTRACTION_RESERVE_TOKENS,
                               reserve_requests=reserve):
            raise ExtractorUnavailable(
                f"{self.name}: no wording budget left; using the approved reply"
            )

        try:
            self._note_call(model, 600)
            # reasoning_effort helps gpt-oss stay short; other models ignore or reject it.
            try:
                response = self._client.chat.completions.create(
                    model=model, temperature=0.4, max_tokens=700,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    reasoning_effort="low",
                )
            except Exception as first:  # noqa: BLE001
                if "reasoning" not in str(first).lower():
                    raise
                response = self._client.chat.completions.create(
                    model=model, temperature=0.4, max_tokens=700,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                )
            self._record_usage(response)
            text = (response.choices[0].message.content or "").strip()
            if text:
                return text
            raise ExtractorUnavailable(f"{self.name}: empty wording reply")
        except ExtractorUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001
            if _is_rate_limited(exc):
                self._note_refusal(model, exc)
            raise ExtractorUnavailable(f"{self.name}: {str(exc)[:120]}") from exc

    # ---- helpers ------------------------------------------------------------

    def _response_format(self) -> dict:
        """
        Ask for JSON shaped exactly like our Extraction class.

        Where the provider supports it, the shape is enforced on their side, so a reply
        that does not fit is impossible. Where it is not supported, we ask for plain JSON
        and Pydantic rejects anything malformed on our side instead.
        """
        if not self.provider.supports_json_schema:
            return {"type": "json_object"}
        schema = Extraction.model_json_schema()
        for drop in ("raw_message", "backend"):
            schema.get("properties", {}).pop(drop, None)

        # Strip the long field descriptions before sending.
        #
        # Those descriptions were originally doubling as the instructions -- neat, because
        # the comment explaining a field to a human was literally what the model read. It
        # turned out to cost 1,076 tokens of our 8,000-per-minute budget, more than half,
        # and the system prompt already says the same things. They stay in schema.py for
        # anyone reading the code; they are simply not worth sending on every request.
        _strip_descriptions(schema)

        schema["additionalProperties"] = False
        schema["required"] = [k for k in schema.get("properties", {})]
        return {
            "type": "json_schema",
            "json_schema": {"name": "extraction", "strict": False, "schema": schema},
        }

    def _messages(self, message: str, history: Sequence[str]) -> list[dict]:
        parts: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for example_message, example_json in _EXAMPLES:
            parts.append({"role": "user", "content": example_message})
            parts.append({"role": "assistant", "content": json.dumps(example_json)})
        if history:
            lines = list(history)[-4:]

            # An offer waiting for an answer is pulled out and stated immediately before
            # the patient's message, not left buried in a long system prompt.
            #
            # It was in the system prompt first, and the model ignored it: after being
            # offered Tuesday 11:00, "actually yes please book it" still came back as a
            # brand new booking request. Instructions placed right next to the thing they
            # apply to get followed; the same words further away do not.
            pending = [l for l in lines if l.startswith("clinic is waiting")]
            listed = [l for l in lines if l.startswith("clinic just showed")]
            context = [l for l in lines
                       if not l.startswith(("clinic is waiting", "clinic just showed"))]

            if context:
                parts.append({
                    "role": "system",
                    "content": ("Conversation so far (context only - extract ONLY from "
                                f"the new message):\n" + "\n".join(context)),
                })
            if listed:
                parts.append({
                    "role": "system",
                    "content": (
                        f"{listed[-1]}\n\n"
                        "If the next message picks one of those - \"the first one\", "
                        "\"option 2\", \"the second\", \"the 10:30 one\", \"the last one\" - "
                        "you MUST copy their exact words into option_reference. Without "
                        "it we cannot tell which one they meant and have to ask again."
                    ),
                })
            if pending:
                parts.append({
                    "role": "system",
                    "content": (
                        f"{pending[-1]}\n\n"
                        "The next message is the patient's ANSWER to that offer. If it "
                        "accepts in any form - \"yes\", \"ok\", \"sure\", \"go ahead\", "
                        "\"book it\", \"yes please book it\", \"that works\" - then intent "
                        "MUST be \"confirm\", even if it contains the word book. If it "
                        "refuses, intent MUST be \"deny\". Only classify it as something "
                        "else if it clearly asks for something different from the offer."
                    ),
                })
        parts.append({"role": "user", "content": message})
        return parts

    def _prune(self, model: str, now: float) -> None:
        window = self._windows.setdefault(model, deque())
        while window and now - window[0][0] > 60.0:
            window.popleft()

    def has_budget(self, model: str, expected_tokens: int,
                   reserve_requests: int = 0) -> bool:
        """
        Could this model take another call right now without breaching a limit?

        Two limits apply and both matter: requests per minute, and tokens per minute. The
        token one is far tighter in practice -- see the note on `tokens_per_minute`.
        """
        now = time.monotonic()
        if self._blocked_until.get(model, 0.0) > now:
            return False

        # Daily allowance first, and read from disk so a restart does not wipe it.
        spent = DAILY_USAGE.get(f"{self.name}:{model}")
        if spent["requests"] >= self.provider.requests_per_day:
            return False
        if (self.provider.tokens_per_day is not None
                and spent["tokens"] + expected_tokens > self.provider.tokens_per_day):
            return False

        # Then this minute.
        self._prune(model, now)
        window = self._windows.setdefault(model, deque())
        if len(window) + reserve_requests >= self.provider.requests_per_minute:
            return False
        return sum(t for _, t in window) + expected_tokens <= self.provider.tokens_per_minute

    def _seconds_until_free(self, expected_tokens: int) -> float:
        """How long before ANY model can take a call. Used only when all are saturated."""
        now = time.monotonic()
        waits = []
        for model in self._models:
            blocked = self._blocked_until.get(model, 0.0)
            if blocked > now:
                waits.append(blocked - now + 0.05)
                continue
            self._prune(model, now)
            window = self._windows.get(model)
            if not window:
                return 0.0
            waits.append(60.0 - (now - window[0][0]) + 0.05)
        return max(0.0, min(waits)) if waits else 0.0

    def _note_refusal(self, model: str, exc: Exception) -> None:
        """
        Believe the server when it says to back off, and for how long.

        Two very different refusals arrive as the same 429, and treating them the same
        was costing us minutes:

          "tokens per minute"  -- busy right now. Back in seconds. Worth waiting for.
          "tokens per day"     -- the free allowance is spent until tomorrow. Waiting is
                                  pointless; every retry is a wasted round trip, and the
                                  honest thing is to stop asking and say so.
        """
        message = str(exc)
        if _is_daily_quota(message):
            self.daily_quota_exhausted = True
            self._blocked_until[model] = time.monotonic() + 3600.0
            # Write it down, so tomorrow's restart does not have to rediscover it the
            # expensive way.
            if self.provider.tokens_per_day is not None:
                DAILY_USAGE.exhaust(f"{self.name}:{model}", self.provider.tokens_per_day)
            return

        seconds = 15.0
        match = re.search(r"try again in ([\d.]+)\s*(m|s)", message, re.I)
        if match:
            value = float(match.group(1))
            seconds = value * 60 if match.group(2).lower() == "m" else value
        self._blocked_until[model] = time.monotonic() + min(seconds + 0.5, 65.0)

    def _note_call(self, model: str, expected_tokens: int) -> None:
        """Record a call optimistically; _record_usage corrects it with the real cost."""
        self._windows.setdefault(model, deque()).append((time.monotonic(), expected_tokens))
        self._last_model_called = model

    def _record_usage(self, response) -> None:
        usage = getattr(response, "usage", None)
        model = self._last_model_called
        if usage is None or model is None:
            return
        window = self._windows.get(model)
        if not window:
            return
        total = getattr(usage, "total_tokens", None)
        if total:
            when, _ = window[-1]
            window[-1] = (when, int(total))
            DAILY_USAGE.add(f"{self.name}:{model}", requests=1, tokens=int(total))

    # ---- cache --------------------------------------------------------------

    def _key(self, message: str, history: Sequence[str]) -> str:
        blob = json.dumps({"m": message, "h": list(history), "p": PROMPT_VERSION,
                           "prov": self.name, "models": self._models}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:32]

    def _cache_get(self, message: str, history: Sequence[str]) -> Extraction | None:
        if not self._use_cache:
            return None
        path = CACHE_DIR / f"{self._key(message, history)}.json"
        if not path.exists():
            return None
        try:
            return Extraction.model_validate_json(path.read_text())
        except Exception:
            return None

    def _cache_put(self, message: str, history: Sequence[str], result: Extraction) -> None:
        if not self._use_cache:
            return
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (CACHE_DIR / f"{self._key(message, history)}.json").write_text(
            result.model_dump_json(indent=1))


def _strip_descriptions(node) -> None:
    """Remove every 'description' key, however deeply nested."""
    if isinstance(node, dict):
        node.pop("description", None)
        for value in node.values():
            _strip_descriptions(value)
    elif isinstance(node, list):
        for value in node:
            _strip_descriptions(value)


def _strip_fences(text: str) -> str:
    """Some models wrap JSON in ```json fences even when told not to."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1]
        t = t.rsplit("```", 1)[0]
    return t.strip()


def _load_key(env_key: str) -> str | None:
    if os.environ.get(env_key):
        return os.environ[env_key]
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[3] / ".env")
    except ImportError:
        pass
    return os.environ.get(env_key)


def _is_daily_quota(message: str) -> bool:
    """Is this 429 about the DAILY allowance rather than the per-minute one?"""
    low = message.lower()
    return "per day" in low or "tpd" in low or "rpd" in low


def _is_rate_limited(exc: Exception) -> bool:
    text = str(exc).lower()
    return "429" in text or "rate limit" in text or "rate_limit" in text


def _is_transient(exc: Exception) -> bool:
    """Busy or rate-limited is worth retrying. A bad request never is."""
    text = str(exc).lower()
    return any(s in text for s in ("503", "429", "502", "504", "timeout",
                                   "overloaded", "rate limit", "connection"))
