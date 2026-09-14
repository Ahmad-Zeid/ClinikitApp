"""
Trying several AI companies in order.

THE SIMPLE VERSION
    If the first company is busy, ask the second. If that one is busy too, ask the third.
    All of them would have to be broken at the same moment before we are stuck.

WHAT IT DOES NOT DO
    It does not fall back to the keyword matcher. That was the old behaviour and it was
    wrong: when Google had a bad hour, real patients got answers from a reader that cannot
    tell "I need to see a doctor" from somebody named Doctor.

    If every provider is down, this raises an error. The layer above then tells the patient
    the truth -- that we are having trouble and a person will help them. An honest "I can't
    do this right now" is better than a confident wrong answer, especially in a clinic.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..schema import Extraction
from .base import Extractor, ExtractorUnavailable


class ChainExtractor(Extractor):
    """Asks each reader in turn until one answers."""

    def __init__(self, extractors: Sequence[Extractor], name: str = "auto") -> None:
        if not extractors:
            raise ExtractorUnavailable("no readers are configured")
        self._extractors = list(extractors)
        self.name = name
        self.last_used: str | None = None
        self.skipped: list[str] = []

    @property
    def members(self) -> list[str]:
        return [e.name for e in self._extractors]

    def extract(self, message: str, history: Sequence[str] = ()) -> Extraction:
        self.skipped = []
        problems: list[str] = []

        for extractor in self._extractors:
            try:
                result = extractor.extract(message, history)
                self.last_used = extractor.name
                return result
            except ExtractorUnavailable as exc:
                self.skipped.append(extractor.name)
                problems.append(f"{extractor.name}: {str(exc)[:90]}")
                continue

        raise ExtractorUnavailable(
            "every provider is unavailable — " + " | ".join(problems)
        )

    def write_text(self, system: str, user: str) -> str:
        problems = []
        for extractor in self._extractors:
            try:
                return extractor.write_text(system, user)
            except ExtractorUnavailable as exc:
                problems.append(str(exc)[:60])
        raise ExtractorUnavailable("no provider could write text: " + " | ".join(problems))

    def __repr__(self) -> str:
        return f"<ChainExtractor {' -> '.join(self.members)}>"
