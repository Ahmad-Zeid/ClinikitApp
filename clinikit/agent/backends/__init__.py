"""Interchangeable ways of reading a patient message."""

from .base import Extractor, ExtractorUnavailable

__all__ = ["Extractor", "ExtractorUnavailable", "get_extractor", "available_backends"]


def get_extractor(name: str = "rules") -> Extractor:
    """
    Build a backend by name.

    Kept here rather than in base.py so that importing the interface never drags in the
    Gemini SDK. Someone running the rule-based backend on a clean machine should not need
    google-genai installed at all.
    """
    name = name.lower().strip()

    if name == "rules":
        from .rules import RuleBasedExtractor
        return RuleBasedExtractor()

    if name == "gemini":
        from .gemini import GeminiExtractor
        return GeminiExtractor()

    raise ValueError(f"Unknown backend {name!r}. Available: rules, gemini")


def available_backends() -> list[str]:
    """
    Which backends can actually run right now.

    'rules' is always present — that is the point of it. 'gemini' appears only if the SDK
    is installed and an API key is configured, so the CLI and the API can degrade to
    something that works instead of showing an error.
    """
    found = ["rules"]
    try:
        from .gemini import GeminiExtractor
        if GeminiExtractor.is_configured():
            found.append("gemini")
    except ImportError:
        pass
    return found
