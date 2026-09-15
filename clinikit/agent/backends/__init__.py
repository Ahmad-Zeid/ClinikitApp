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
# Groq only by default. Gemini's free tier is 1,500 requests a day and a single
# evaluation run can use most of it -- after which every request fails with 429 and the
# assistant has nothing to say. Keeping a provider in the queue that is usually out of
# quota costs a wasted call on every single message.
#
# The others are one environment variable away, which is the point of the whole
# multi-provider design:
#     CLINIKIT_PROVIDERS=groq,gemini,openrouter
CHAIN_ORDER = tuple(
    p.strip() for p in os.environ.get("CLINIKIT_PROVIDERS", "groq").split(",") if p.strip()
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

    if name == "gemini":
        from .gemini import GeminiExtractor
        return GeminiExtractor()

    if name in ("groq", "openrouter"):
        from .openai_compat import OpenAICompatibleExtractor
        return OpenAICompatibleExtractor(provider=name)

    raise ValueError(f"Unknown backend {name!r}. Try: auto, groq, gemini, openrouter, rules")


def patient_facing_backends() -> list[str]:
    """Readers good enough to answer a real patient. Never includes 'rules'."""
    found = []
    try:
        from .openai_compat import OpenAICompatibleExtractor
        for provider in ("groq", "openrouter"):
            if OpenAICompatibleExtractor.is_configured(provider):
                found.append(provider)
    except ImportError:
        pass
    try:
        from .gemini import GeminiExtractor
        if GeminiExtractor.is_configured():
            found.append("gemini")
    except ImportError:
        pass
    # Put them back in preference order, then offer the combined chain.
    ordered = [p for p in CHAIN_ORDER if p in found]
    return (["auto"] + ordered) if ordered else []


def available_backends() -> list[str]:
    """Everything that can run, including the evaluation-only keyword reader."""
    return patient_facing_backends() + ["rules"]
