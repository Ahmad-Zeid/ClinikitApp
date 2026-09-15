"""
Turning what the patient *wrote* into what they *meant*.

The extraction schema deliberately keeps dates and times as raw phrases — "tomorrow",
"after 5" — because language models are unreliable at date arithmetic and have no idea
what today is. This module does the arithmetic instead, in ordinary Python, against the
clinic's real calendar.

Two principles run through the whole file:

1. **`now` is always a parameter, never `datetime.now()`.**
   A function that reads the clock itself cannot be tested: the same test passes at 9am
   and fails at midnight. Passing the clock in makes every result reproducible forever.

2. **Ambiguity is reported, never guessed.**
   "Friday at 4" could mean 04:00 or 16:00. Where the clinic's opening hours settle it,
   we settle it. Where they don't, we say so and let the policy layer ask. A confident
   wrong answer is worse here than an admitted uncertainty.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Optional

from .clinic import CLINIC_HOURS, TIMEZONE, clinic_hours_on

# ──────────────────────────────────────────────────────────────────────────────
# Problem codes
# ──────────────────────────────────────────────────────────────────────────────
# Returned instead of a resolution when we cannot honestly produce one. These are
# codes rather than sentences so the policy layer can branch on them and the
# responder can phrase them; the two concerns stay separate.

PAST_DATE = "past_date"                  # the day they named has already gone
CLOSED_DAY = "closed_day"                # clinic is shut that day (Sunday)
AMBIGUOUS_HOUR = "ambiguous_hour"        # "at 9" — both 09:00 and 21:00 plausible
OUTSIDE_HOURS = "outside_hours"          # a real time, but the clinic is shut then
UNPARSEABLE_DATE = "unparseable_date"    # we could not make sense of the date phrase
UNPARSEABLE_TIME = "unparseable_time"    # we could not make sense of the time phrase


# ──────────────────────────────────────────────────────────────────────────────
# The result
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Resolved:
    """
    What a date phrase and a time phrase together add up to.

    Deliberately expressed as a *search space* rather than a single moment, because most
    patient messages describe one. "Next week" is six days. "Afternoon" is five hours.
    "Friday at 4" is the rare case that pins down exactly one slot — and only that case
    sets `exact`.
    """

    days: tuple[date, ...] = ()
    """Candidate days, earliest first. Empty means the date was absent or unusable."""

    earliest: Optional[time] = None
    """Lower bound on time-of-day. None means no constraint."""

    latest: Optional[time] = None
    """Upper bound on time-of-day. None means no constraint."""

    exact: Optional[datetime] = None
    """Set only when exactly one day and one unambiguous time were given."""

    problems: tuple[str, ...] = ()
    """Problem codes from the list above. Non-empty means something needs saying."""

    @property
    def is_exact(self) -> bool:
        """True when the patient named one specific bookable slot."""
        return self.exact is not None

    @property
    def has_date(self) -> bool:
        return bool(self.days)

    @property
    def has_time(self) -> bool:
        return self.earliest is not None or self.latest is not None

    @property
    def is_empty(self) -> bool:
        return not self.has_date and not self.has_time


# ──────────────────────────────────────────────────────────────────────────────
# Opening-hours envelope
# ──────────────────────────────────────────────────────────────────────────────

def _opening_envelope() -> tuple[time, time]:
    """
    The widest span the clinic is ever open in a week — currently 08:00 to 18:00.

    Used to disambiguate bare hours when we don't yet know which day is meant. Computed
    from CLINIC_HOURS rather than hardcoded, so changing the clinic's hours automatically
    changes how "at 4" is interpreted.
    """
    spans = [h for h in CLINIC_HOURS.values() if h is not None]
    return min(s[0] for s in spans), max(s[1] for s in spans)


def _plausible_hours(hour_12: int, day: Optional[date] = None) -> list[int]:
    """
    Given a bare hour like "4", which 24-hour readings could the clinic actually mean?

    This is the domain knowledge that lets us resolve something a language model can't.
    The clinic opens 08:00-18:00, so:

        "4"  -> 04:00 is shut, 16:00 is open  -> unambiguous, 16:00
        "9"  -> 09:00 is open, 21:00 is shut  -> unambiguous, 09:00
        "11" -> 11:00 is open, 23:00 is shut  -> unambiguous, 11:00
        "7"  -> 07:00 is shut, 19:00 is shut  -> neither works; report it

    Returns every plausible reading. One means we can proceed; two means genuinely
    ambiguous; zero means the clinic is never open at that hour.
    """
    if day is not None:
        hours = clinic_hours_on(day)
        if hours is None:
            return []
        opens, closes = hours
    else:
        opens, closes = _opening_envelope()

    candidates = {hour_12 % 12, (hour_12 % 12) + 12}
    if hour_12 == 12:
        candidates = {12, 0}
    return sorted(h for h in candidates if opens.hour <= h < closes.hour)


# ──────────────────────────────────────────────────────────────────────────────
# Date phrases
# ──────────────────────────────────────────────────────────────────────────────

_WEEKDAYS = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}


def _next_occurrence(weekday: int, today: date, allow_today: bool) -> date:
    """The next date falling on `weekday`. Includes today only if `allow_today`."""
    delta = (weekday - today.weekday()) % 7
    if delta == 0 and not allow_today:
        delta = 7
    return today + timedelta(days=delta)


def _week_span(start: date, days: int = 6) -> tuple[date, ...]:
    """`days` consecutive dates from `start`, dropping any the clinic is closed."""
    out = []
    for i in range(days):
        d = start + timedelta(days=i)
        if clinic_hours_on(d) is not None:
            out.append(d)
    return tuple(out)


def resolve_date(phrase: Optional[str], now: datetime) -> tuple[tuple[date, ...], tuple[str, ...]]:
    """
    Turn a date phrase into candidate days.

    Returns (days, problems). An empty `days` with an empty `problems` means the patient
    simply didn't mention a date — which is normal and not an error.
    """
    if not phrase or not phrase.strip():
        return (), ()

    p = phrase.lower().strip()
    today = now.date()

    # --- relative single days ---
    if p in ("today", "todays", "2day"):
        if clinic_hours_on(today) is None:
            return (), (CLOSED_DAY,)
        return (today,), ()

    if p in ("tomorrow", "tmrw", "tmr", "2morrow", "bukra", "boukra"):
        d = today + timedelta(days=1)
        if clinic_hours_on(d) is None:
            return (), (CLOSED_DAY,)
        return (d,), ()

    if p in ("day after tomorrow", "day after"):
        return (today + timedelta(days=2),), ()

    # --- "in N days" ---
    m = re.fullmatch(r"in (\d+) days?", p)
    if m:
        return (today + timedelta(days=int(m.group(1))),), ()

    # --- week-scale ranges ---
    # "next week" -> Monday of the following week, six days forward.
    if "next week" in p:
        monday = _next_occurrence(0, today, allow_today=False)
        return _week_span(monday), ()

    if "this week" in p:
        return _week_span(today + timedelta(days=1), days=6 - today.weekday()), ()

    # --- weekday names, with an optional prefix ---
    #
    # "this thursday", "coming Friday" and "on Monday" all name the same kind of thing as
    # a bare weekday. Only "next" changes the meaning. Missing "this" meant "can I do
    # this thursday at 12?" came back as "no usable date was given" -- the reader had
    # done its job and the arithmetic threw the answer away.
    wants_following_week = p.startswith("next ")
    bare = p[5:].strip() if wants_following_week else p
    for filler in ("this coming ", "this ", "coming ", "on ", "the "):
        if bare.startswith(filler):
            bare = bare[len(filler):].strip()
    bare = bare.replace("on ", "").strip()

    if bare in _WEEKDAYS:
        wd = _WEEKDAYS[bare]
        # "Monday" said on a Monday morning usually means today, if we're still open.
        allow_today = (
            not wants_following_week
            and wd == today.weekday()
            and (clinic_hours_on(today) or (time(0), time(0)))[1] > now.time()
        )
        d = _next_occurrence(wd, today, allow_today=allow_today)
        if wants_following_week and d <= today + timedelta(days=(6 - today.weekday())):
            d += timedelta(days=7)
        if clinic_hours_on(d) is None:
            return (), (CLOSED_DAY,)
        return (d,), ()

    # --- explicit calendar dates: 15/9, 15-09-2026, 2026-09-15 ---
    for pattern, order in (
        (r"(\d{4})-(\d{1,2})-(\d{1,2})", "ymd"),
        (r"(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?", "dmy"),
    ):
        m = re.fullmatch(pattern, p)
        if not m:
            continue
        try:
            if order == "ymd":
                d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            else:
                day_n, month_n = int(m.group(1)), int(m.group(2))
                year_n = int(m.group(3)) if m.group(3) else today.year
                if year_n < 100:
                    year_n += 2000
                d = date(year_n, month_n, day_n)
        except ValueError:
            return (), (UNPARSEABLE_DATE,)
        if d < today:
            return (), (PAST_DATE,)
        if clinic_hours_on(d) is None:
            return (), (CLOSED_DAY,)
        return (d,), ()

    return (), (UNPARSEABLE_DATE,)


# ──────────────────────────────────────────────────────────────────────────────
# Time phrases
# ──────────────────────────────────────────────────────────────────────────────

# Phrases that mention time without constraining it. These must resolve to "no
# constraint", NOT to an error: a patient saying "anytime next week" has given us
# perfectly usable information, and reporting UNPARSEABLE_TIME here would make the
# policy layer ask a clarifying question that the patient already answered.
_VAGUE_TIME = {
    "any", "anytime", "any time", "sometime", "some time", "whenever",
    "flexible", "no preference", "doesn't matter", "dont matter",
    "does not matter", "up to you", "asap", "as soon as possible", "soon",
}

_PARTS_OF_DAY = {
    "morning": (None, time(12, 0)),        # opening -> noon
    "the morning": (None, time(12, 0)),
    "afternoon": (time(12, 0), time(17, 0)),
    "the afternoon": (time(12, 0), time(17, 0)),
    "arvo": (time(12, 0), time(17, 0)),
    "evening": (time(17, 0), None),        # 5pm -> closing
    "the evening": (time(17, 0), None),
    "noon": (time(12, 0), time(12, 30)),
    "midday": (time(12, 0), time(12, 30)),
}


def _parse_clock(text: str, day: Optional[date]) -> tuple[Optional[time], tuple[str, ...]]:
    """
    Parse a clock reading such as "4", "4pm", "4:30", "16:00".

    Returns (time, problems). A bare hour with two plausible readings yields
    (None, (AMBIGUOUS_HOUR,)) rather than a coin-flip.
    """
    t = text.strip().lower().replace(".", "")

    m = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", t)
    if not m:
        return None, (UNPARSEABLE_TIME,)

    hour = int(m.group(1))
    minute = int(m.group(2) or 0)
    meridiem = m.group(3)

    if minute > 59 or hour > 24:
        return None, (UNPARSEABLE_TIME,)

    # An explicit am/pm settles it outright.
    if meridiem:
        if hour > 12:
            return None, (UNPARSEABLE_TIME,)
        if meridiem == "pm" and hour != 12:
            hour += 12
        if meridiem == "am" and hour == 12:
            hour = 0
        return time(hour, minute), ()

    # 24-hour readings are already unambiguous.
    if hour > 12 or hour == 0:
        return time(hour, minute), ()

    # Bare 1-12: let the clinic's opening hours decide.
    options = _plausible_hours(hour, day)
    if len(options) == 1:
        return time(options[0], minute), ()
    if len(options) == 0:
        return None, (OUTSIDE_HOURS,)
    return None, (AMBIGUOUS_HOUR,)


def resolve_time(
    phrase: Optional[str], day: Optional[date] = None
) -> tuple[Optional[time], Optional[time], tuple[str, ...]]:
    """
    Turn a time phrase into an (earliest, latest) window.

    Returns (earliest, latest, problems). Both None with no problems means the patient
    didn't mention a time.
    """
    if not phrase or not phrase.strip():
        return None, None, ()

    p = phrase.lower().strip()

    if p in _VAGUE_TIME:
        return None, None, ()

    p = re.sub(r"^(at|around|about|sometime)\s+", "", p).strip()

    if p in _VAGUE_TIME:
        return None, None, ()

    if p in _PARTS_OF_DAY:
        lo, hi = _PARTS_OF_DAY[p]
        return lo, hi, ()

    # "after 5", "later than 5"
    m = re.fullmatch(r"(?:after|past|later than|from)\s+(.+)", p)
    if m:
        t, probs = _parse_clock(m.group(1), day)
        return (t, None, probs) if t else (None, None, probs)

    # "before 11", "until 11"
    m = re.fullmatch(r"(?:before|earlier than|until|by)\s+(.+)", p)
    if m:
        t, probs = _parse_clock(m.group(1), day)
        return (None, t, probs) if t else (None, None, probs)

    # "between 2 and 4"
    m = re.fullmatch(r"between\s+(.+?)\s+(?:and|to|-)\s+(.+)", p)
    if m:
        lo, p1 = _parse_clock(m.group(1), day)
        hi, p2 = _parse_clock(m.group(2), day)
        return lo, hi, tuple(dict.fromkeys(p1 + p2))

    # A plain clock reading: an exact half-hour slot.
    t, probs = _parse_clock(p, day)
    if t is not None:
        end_minute = t.minute + 30
        end = time(t.hour + end_minute // 60, end_minute % 60) if t.hour < 23 else time(23, 59)
        return t, end, ()
    return None, None, probs


# ──────────────────────────────────────────────────────────────────────────────
# The combined entry point
# ──────────────────────────────────────────────────────────────────────────────

def resolve(
    preferred_date: Optional[str],
    preferred_time: Optional[str],
    now: Optional[datetime] = None,
) -> Resolved:
    """
    Resolve the two phrases from an `Extraction` into a searchable window.

    `now` defaults to the real clock only as a convenience for the CLI. Every test and
    every call from the policy layer passes it explicitly.
    """
    if now is None:
        now = datetime.now(TIMEZONE)

    days, date_problems = resolve_date(preferred_date, now)

    # Resolving the time against a known day gives better disambiguation — that day's
    # actual opening hours are narrower than the week's envelope.
    day_hint = days[0] if len(days) == 1 else None
    earliest, latest, time_problems = resolve_time(preferred_time, day_hint)

    problems = tuple(dict.fromkeys(date_problems + time_problems))

    # An exact slot needs exactly one day and one specific start time. A window like
    # "afternoon" has an `earliest`, but it is a range, not a booking.
    exact = None
    if len(days) == 1 and earliest is not None and preferred_time:
        looks_like_a_point = bool(
            re.fullmatch(r"(?:at|around|about)?\s*\d{1,2}(?::\d{2})?\s*(?:am|pm)?",
                         preferred_time.strip().lower())
        )
        if looks_like_a_point:
            candidate = datetime.combine(days[0], earliest, tzinfo=TIMEZONE)
            if candidate < now:
                problems = tuple(dict.fromkeys(problems + (PAST_DATE,)))
            else:
                exact = candidate

    return Resolved(
        days=days,
        earliest=earliest,
        latest=latest,
        exact=exact,
        problems=problems,
    )
