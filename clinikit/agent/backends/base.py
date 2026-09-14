"""
The shape every message-reader must have.

Think of this as a plug socket. We will have more than one way of reading a patient
message: a simple one built from keyword matching, and a smarter one powered by Gemini.
This file describes the socket. Anything that fits it can be plugged in, and the rest of
the program never has to know which one is plugged in right now.

The reason that matters: it lets us swap the smart reader for the simple one and measure
the difference. Without a shared socket, "is the LLM actually helping?" is a question we
could only answer by rewriting everything twice.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from ..schema import Extraction


class ExtractorUnavailable(RuntimeError):
    """
    Raised when a backend cannot run — no API key, SDK missing, service down.

    Deliberately its own error type rather than a generic one, so callers can catch
    exactly this and fall back to a different backend, instead of catching everything
    and accidentally swallowing real bugs.
    """


class Extractor(ABC):
    """
    Base class for anything that turns a patient message into an `Extraction`.

    `ABC` means "abstract base class" — a class that exists to be inherited from, never
    used directly. Combined with `@abstractmethod` below, Python will refuse to create a
    backend that forgot to write an `extract` method. The mistake is caught immediately
    rather than at 2am in production.
    """

    name: str = "base"
    """Short identifier recorded on every Extraction, so results are traceable."""

    @abstractmethod
    def extract(self, message: str, history: Sequence[str] = ()) -> Extraction:
        """
        Read one patient message and return structured information.

        `history` holds earlier messages in the conversation, oldest first. The
        rule-based backend ignores it; the Gemini backend uses it to make sense of
        replies like "the dermatologist" that only mean something in context.

        Implementations must NOT perform any clinic action. They return data and nothing
        else. Deciding what may be done with that data belongs to policy.py — which is
        the single most important rule in this codebase.
        """

    def write_text(self, system: str, user: str) -> str:
        """
        Ask the model for a plain sentence rather than structured data.

        Used by phrasing.py to reword a reply in natural language. Kept here so any
        provider can offer it and the phrasing code does not care which one answered.

        Readers with no language model behind them (the keyword matcher) raise, and the
        caller falls back to the hand-written reply.
        """
        raise ExtractorUnavailable(f"{self.name} cannot write free text")

    def _finalise(self, extraction: Extraction, message: str) -> Extraction:
        """
        Stamp provenance onto a finished extraction.

        Every result records the message it came from and the backend that produced it,
        so an evaluation run or an audit log can always answer "where did this come
        from?" without guesswork.
        """
        return extraction.model_copy(update={"raw_message": message, "backend": self.name})

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"
