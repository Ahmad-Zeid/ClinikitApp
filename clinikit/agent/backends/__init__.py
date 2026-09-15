"""
Interchangeable ways of reading a patient message.

Two kinds live here, and the difference matters:

  READERS THAT TALK TO PATIENTS
      groq, gemini, openrouter, and "auto" which tries them in that order.
      These use a language model and can handle typos, Arabizi and vague messages.

  A READER THAT NEVER TALKS TO PATIENTS
      "rules" -- keyword matching. It exists only so the evaluation can answer
      "did the language model actually earn its place?" by comparing against it.
      It gives bad answers to ordinary messages, so it is never used as a stand-in
      when a provider is down. See chain.py for what happens instead.
"""

from .base import Extractor, ExtractorUnavailable

__all__ = [
    "Extractor", "ExtractorUnavailable", "get_extractor",
    "available_backends", "patient_facing_backends",
]

import os

# Which providers "auto" tries, in order.
#
# GEMINI FIRST, GROQ AS THE BACKUP.
#
# SPEED IS NO LONGER THE DECIDER. It used to be: Groq answered in 1-2 seconds while
# Gemini took 5-16. That gap closed when we moved to Gemini's flash-lite model, which
# our own evaluation measures at a 1.2s median. Both are fast enough that a patient
# cannot tell them apart, so the tie is broken on how long each one lasts.
#
# THEIR FREE ALLOWANCES RUN OUT IN COMPLETELY DIFFERENT WAYS:
#     Gemini  1,500 REQUESTS a day, and NO daily token cap. Our calls are small, so the
#             size of each one is irrelevant. That is ~1,500 turns.
#     Groq    200,000 TOKENS per model per day. Our reading prompt is ~1,300 tokens, so
#             the token cap bites at roughly 460 calls -- long before the request
#             allowance is touched.
#
# Gemini is therefore the bigger tank, and it goes first. Groq covers the overflow, and
# the two together carry roughly two thousand turns a day on free accounts and no card.
#
# ONE PRACTICAL NOTE, FOUND THE HARD WAY
#     Groq's API is unreachable from some networks -- ours answers Cloudflare's 403 from
#     a home connection in Lebanon while working perfectly from the deployed site. That
#     is a second, quieter reason to lead with Gemini: it is the provider that works from
#     everywhere we run.
#
# The handover is automatic and survives restarts: daily usage is written to
# .cache/daily_usage.json. It used to live only in memory, so every restart believed it
# had a fresh day's allowance and kept calling an exhausted provider.
#
# Override with: CLINIKIT_PROVIDERS=groq,gemini
CHAIN_ORDER = tuple(
    p.strip() for p in os.environ.get("CLINIKIT_PROVIDERS", "gemini,groq").split(",")
    if p.strip()
)


def get_extractor(name: str = "auto") -> Extractor:
    name = (name or "auto").lower().strip()

    if name == "auto":
        from .chain import ChainExtractor
        members = []
        for provider in CHAIN_ORDER:
            try:
                members.append(get_extractor(provider))
            except ExtractorUnavailable:
                continue
        if not members:
            raise ExtractorUnavailable(
                "No AI provider is configured. Add GROQ_API_KEY (free, no card, from "
                "https://console.groq.com) or GEMINI_API_KEY to your .env file."
            )
        return ChainExtractor(members)

    if name == "rules":
        from .rules import RuleBasedExtractor
        return RuleBasedExtractor()

    if name in ("gemini", "groq", "openrouter"):
        # All three speak the OpenAI format, Gemini included via Google's compatibility
        # endpoint. One class, three providers, and every one of them gets the same
        # retries, budgets and model fallback for free.
        from .openai_compat import OpenAICompatibleExtractor
        return OpenAICompatibleExtractor(provider=name)

    raise ValueError(f"Unknown backend {name!r}. Try: auto, groq, gemini, openrouter, rules")


def patient_facing_backends() -> list[str]:
    """Readers good enough to answer a real patient. Never includes 'rules'."""
    found = []
    try:
        from .openai_compat import OpenAICompatibleExtractor
        for provider in ("gemini", "groq", "openrouter"):
            if OpenAICompatibleExtractor.is_configured(provider):
                found.append(provider)
    except ImportError:
        pass
    # Put them back in preference order, then offer the combined chain.
    ordered = [p for p in CHAIN_ORDER if p in found]
    return (["auto"] + ordered) if ordered else []


def available_backends() -> list[str]:
    """Everything that can run, including the evaluation-only keyword reader."""
    return patient_facing_backends() + ["rules"]
