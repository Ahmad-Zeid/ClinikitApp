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
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..clinic import DOCTORS
from ..schema import Extraction
from .base import Extractor, ExtractorUnavailable

PROMPT_VERSION = "v2"
CACHE_DIR = Path(__file__).resolve().parents[3] / ".cache" / "llm"


@dataclass(frozen=True)
class Provider:
    """Everything that differs between one AI company and another."""

    name: str
    base_url: str
    env_key: str
    models: tuple[str, ...]
    requests_per_minute: int
    supports_json_schema: bool = True
    signup: str = ""


PROVIDERS: dict[str, Provider] = {
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
You read messages patients send to a clinic in Beirut and turn them into JSON. You never
reply to the patient and never take any action.

Doctors:
{_roster()}

RULES
1. Copy dates/times EXACTLY as written ("tomorrow", "after 5"). Never convert to a real
   date - you do not know today's date.
2. Reschedule: preferred_date is the NEW date; the old one goes in
   existing_appointment_phrase. "from Monday to Wednesday" -> new=Wednesday, old=Monday.
3. is_hedged = the patient explicitly said not to act yet ("don't book yet", "just
   checking"). A plain question is NOT hedged. Politeness is NOT hedging.
4. Only record a doctor they actually named. "a doctor"/"my doctor"/"someone" = nobody;
   leave doctor empty (set refers_to_previous_visit if they meant a past doctor).
5. Expect typos and Arabizi (Lebanese Arabic in Latin letters). Glossary:
   badde=I want, shouf/shoufo=see, bukra=tomorrow, lyom=today, 3anjad=really, shu=what,
   eymta=when, fi=is there, fadi/fadye=free, dawam=working hours, tabaakon=yours,
   3iyede=clinic, mawad=appointment, baddi=I want, ba3dein=later, mnee7=good,
   la2=no, eh/na3am=yes, 7akim/doktor=doctor, sa3a=hour/time, jomaa=Friday.
6. confidence 0-1, honest. Vague message = low, so we ask instead of guessing.
7. Asking to COME IN on a day ("can I come in Thursday?", "are you free Friday?") is
   book_appointment. ask_opening_hours is ONLY for what hours the clinic operates.
8. A symptom with no request ("my back hurts") is book_appointment, modest confidence,
   symptom in reason_for_visit. A bare greeting is "other", low confidence.
9. confirm/deny are ONLY short replies to a question we just asked ("yes", "go ahead",
   "no thanks"). "ok thanks" is "other". A yes carrying a new request is NOT a confirm.

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
    ("cn u mve my apt frm mon to wed pls",
     {"intent": "reschedule_appointment", "confidence": 0.88, "doctor": None,
      "preferred_date": "wed", "preferred_time": None,
      "existing_appointment_phrase": "mon", "is_hedged": False,
      "refers_to_previous_visit": False, "reason_for_visit": None,
      "reasoning": "typos, but clearly moving Monday to Wednesday"}),
]


class OpenAICompatibleExtractor(Extractor):
    """Reads patient messages using any provider that speaks the OpenAI format."""

    def __init__(
        self,
        provider: str = "groq",
        api_key: str | None = None,
        models: Sequence[str] | None = None,
        use_cache: bool = True,
        budget_seconds: float = 25.0,
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

        self._client = OpenAI(api_key=key, base_url=self.provider.base_url, timeout=20.0)
        self._models = tuple(models or self.provider.models)
        self._use_cache = use_cache
        self._budget = budget_seconds
        self._min_interval = 60.0 / self.provider.requests_per_minute
        self._last_call_at = 0.0

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

        for model in self._models:
            if time.monotonic() > deadline:
                break
            for attempt in range(2):
                try:
                    self._throttle()
                    started = time.monotonic()
                    response = self._client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.0,
                        response_format=self._response_format(),
                    )
                    self.last_latency = time.monotonic() - started
                    self.last_model_used = model

                    text = response.choices[0].message.content or ""
                    result = Extraction.model_validate_json(_strip_fences(text))
                    self._cache_put(message, history, result)
                    return self._finalise(result, message)

                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    # Each model has its OWN tokens-per-minute budget. So when one says
                    # "too many requests", the fastest fix is a different model, not
                    # waiting a minute for the same one to recover.
                    if _is_rate_limited(exc):
                        break
                    if _is_transient(exc) and attempt == 0 and time.monotonic() < deadline:
                        time.sleep(1.5)
                        continue
                    break

        raise ExtractorUnavailable(
            f"{self.name}: every model failed. Last error: "
            f"{type(last_error).__name__}: {str(last_error)[:160]}"
        ) from last_error

    def write_text(self, system: str, user: str) -> str:
        """
        Plain-language generation, used for rewording replies.

        max_tokens is generous and reasoning effort is set low on purpose. The gpt-oss
        models think before they answer, and that thinking is charged against the same
        budget. With a tight limit they used the whole allowance thinking and returned an
        empty answer -- which looked like the model being broken when it was us starving it.
        """
        last_error: Exception | None = None
        for model in self._models:
            for extra in ({"reasoning_effort": "low"}, {}):
                try:
                    self._throttle()
                    response = self._client.chat.completions.create(
                        model=model, temperature=0.4, max_tokens=700,
                        messages=[{"role": "system", "content": system},
                                  {"role": "user", "content": user}],
                        **extra,
                    )
                    text = (response.choices[0].message.content or "").strip()
                    if text:
                        return text
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    continue
        raise ExtractorUnavailable(f"{self.name}: {str(last_error)[:120]}") from last_error

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
            recent = " | ".join(list(history)[-4:])
            parts.append({"role": "system",
                          "content": f"Earlier messages, for context only: {recent}"})
        parts.append({"role": "user", "content": message})
        return parts

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call_at
        if self._last_call_at and elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call_at = time.monotonic()

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


def _is_rate_limited(exc: Exception) -> bool:
    text = str(exc).lower()
    return "429" in text or "rate limit" in text or "rate_limit" in text


def _is_transient(exc: Exception) -> bool:
    """Busy or rate-limited is worth retrying. A bad request never is."""
    text = str(exc).lower()
    return any(s in text for s in ("503", "429", "502", "504", "timeout",
                                   "overloaded", "rate limit", "connection"))
