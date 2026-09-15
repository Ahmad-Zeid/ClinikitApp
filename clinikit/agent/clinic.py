"""
The mock clinic: doctors, opening hours, and the appointment book.

This is the *world* the agent acts in. No AI here — just data and lookups. Keeping it
separate matters, because the policy layer's questions ("is that a real doctor?", "are
we even open then?") have to be answered by something other than a language model.

The data is fictional but shaped like a real Beirut clinic: a six-day week with a short
Saturday, closed Sunday, and doctors who each work a subset of the clinic's hours.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

# All clinic reasoning happens in local time. Storing a timezone explicitly — rather
# than relying on the server's clock — is the difference between "works on my laptop"
# and "works when deployed on a server in Frankfurt".
TIMEZONE = ZoneInfo("Asia/Beirut")

# Appointments start on the hour or the half hour.
SLOT_MINUTES = 30


# ──────────────────────────────────────────────────────────────────────────────
# Opening hours
# ──────────────────────────────────────────────────────────────────────────────

# Monday = 0 ... Sunday = 6, matching Python's datetime.weekday().
# A value of None means the clinic is closed that day.
CLINIC_HOURS: dict[int, Optional[tuple[time, time]]] = {
    0: (time(8, 0), time(18, 0)),   # Monday
    1: (time(8, 0), time(18, 0)),   # Tuesday
    2: (time(8, 0), time(18, 0)),   # Wednesday
    3: (time(8, 0), time(18, 0)),   # Thursday
    4: (time(8, 0), time(18, 0)),   # Friday
    5: (time(9, 0), time(13, 0)),   # Saturday — half day
    6: None,                        # Sunday — closed
}

WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]

# Plain facts about the clinic, so questions like "where are you?" have an answer instead
# of falling through to "could you tell me more about what you need".
CLINIC_NAME = "CliniKit Medical Centre"
CLINIC_ADDRESS = "Rue Verdun, Hamra, Beirut"
CLINIC_PHONE = "+961 1 000 000"


def clinic_hours_on(d: date) -> Optional[tuple[time, time]]:
    """Opening and closing time on a given date, or None if closed."""
    return CLINIC_HOURS[d.weekday()]


def is_clinic_open_at(dt: datetime) -> bool:
    """True if the clinic is open at this exact moment."""
    hours = clinic_hours_on(dt.date())
    if hours is None:
        return False
    opens, closes = hours
    return opens <= dt.time() < closes


def describe_opening_hours() -> str:
    """A human-readable summary, used to answer the `ask_opening_hours` intent."""
    return (
        "Monday to Friday: 8:00 AM - 6:00 PM\n"
        "Saturday: 9:00 AM - 1:00 PM\n"
        "Sunday: closed"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Doctors
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Doctor:
    id: str
    first_name: str
    last_name: str
    specialty: str
    working_days: tuple[int, ...]          # weekday numbers this doctor works
    start: time
    end: time

    @property
    def full_name(self) -> str:
        return f"Dr. {self.first_name} {self.last_name}"

    @property
    def short_name(self) -> str:
        return f"Dr. {self.first_name}"

    @property
    def hours_summary(self) -> str:
        """
        When this doctor works, written the way a receptionist would say it.

        "Mondays, Wednesdays and Fridays, 10am to 6pm" -- not "(0, 2, 4) 10:00-18:00".
        Offered whenever we suggest a doctor, so the patient can tell straight away
        whether that person is any use to them.
        """
        days = [WEEKDAY_NAMES[d] + "s" for d in sorted(self.working_days)]
        if len(days) == 1:
            when = days[0]
        elif len(days) == 5 and set(self.working_days) == {0, 1, 2, 3, 4}:
            when = "weekdays"
        else:
            when = ", ".join(days[:-1]) + " and " + days[-1]

        def clock(t: time) -> str:
            hour = t.hour % 12 or 12
            suffix = "am" if t.hour < 12 else "pm"
            return f"{hour}{suffix}" if t.minute == 0 else f"{hour}:{t.minute:02d}{suffix}"

        return f"{when}, {clock(self.start)} to {clock(self.end)}"

    def works_on(self, d: date) -> bool:
        return d.weekday() in self.working_days


# Dr. George and Dr. Karim are named in the CliniKit brief's own example messages, so
# they are kept. Note the two Khourys: in Lebanon a shared surname is entirely ordinary,
# and it gives the agent a real disambiguation case to handle rather than a contrived one.
DOCTORS: list[Doctor] = [
    Doctor("d_george", "George", "Haddad", "General Practitioner",
           (0, 1, 2, 3, 4), time(8, 0), time(16, 0)),
    Doctor("d_karim", "Karim", "Nassar", "Cardiologist",
           (0, 2, 4), time(10, 0), time(18, 0)),
    Doctor("d_rima", "Rima", "Khoury", "Dermatologist",
           (1, 3, 5), time(9, 0), time(13, 0)),
    Doctor("d_nadim", "Nadim", "Khoury", "Paediatrician",
           (0, 1, 2, 3), time(8, 0), time(14, 0)),
    Doctor("d_layla", "Layla", "Aoun", "General Practitioner",
           (1, 2, 3, 4, 5), time(11, 0), time(18, 0)),
]


def _normalise(name: str) -> str:
    """
    Strip titles and punctuation so 'Dr. George', 'dr george' and 'GEORGE' all match.

    Also folds the common Lebanese spelling variants of a few first names, because
    patients write the name they use, not the one in our database.
    """
    n = name.lower().strip()
    for prefix in ("dr.", "dr ", "doctor ", "d."):
        if n.startswith(prefix):
            n = n[len(prefix):]
    n = n.strip(" .,")
    # e.g. a patient writing "Georges" (French spelling) means our "George"
    variants = {"georges": "george", "kareem": "karim", "nadeem": "nadim",
                "leila": "layla", "laila": "layla", "reema": "rima"}
    return variants.get(n, n)


def find_doctors(query: Optional[str]) -> list[Doctor]:
    """
    Look up doctors by whatever the patient wrote.

    Returns a LIST rather than a single doctor, deliberately. The caller must handle
    three distinct outcomes, and collapsing them would hide the interesting one:

        []            -> no such doctor; tell the patient
        [one]         -> unambiguous; proceed
        [two or more] -> ambiguous ("Dr. Khoury"); we must ask which one

    Silently picking the first match in the ambiguous case is exactly the sort of quiet
    wrong guess this system is designed not to make.
    """
    if not query:
        return []
    q = _normalise(query)
    if not q:
        return []

    # Exact match on first name, last name, or "first last".
    exact = [d for d in DOCTORS
             if q in (d.first_name.lower(), d.last_name.lower(),
                      f"{d.first_name} {d.last_name}".lower())]
    if exact:
        return exact

    # Fall back to a prefix match, which catches typos like "Geor" or "Nass".
    return [d for d in DOCTORS
            if d.first_name.lower().startswith(q) or d.last_name.lower().startswith(q)]


# ──────────────────────────────────────────────────────────────────────────────
# Pointing a patient at the right doctor
# ──────────────────────────────────────────────────────────────────────────────
#
# A patient who says "something is wrong with my heart" should be told about the
# cardiologist, not shown a general practitioner's diary. That is what this table does.
#
# THIS IS A CLINICAL DECISION AND IS TREATED AS ONE.
#
#   - We name a department. We never say what the symptom means, never suggest a
#     diagnosis, never tell anyone how urgent their problem is.
#   - The reply is always "Dr. Karim Nassar is our cardiologist", never "you should see
#     a cardiologist" -- the difference matters. One is directory information. The other
#     is medical advice, and this system is not qualified to give it.
#   - In a real clinic this table would be written or approved by a doctor, and the
#     README says so. It is here to show the shape of the idea, not to be relied on.

SYMPTOM_ROUTES: dict[str, tuple[str, ...]] = {
    "Cardiologist": (
        "heart", "chest", "cardiac", "palpitation", "blood pressure", "bp",
        "cholesterol", "pulse", "ecg", "angina",
    ),
    "Dermatologist": (
        "skin", "rash", "acne", "mole", "eczema", "psoriasis", "itch", "hives",
        "dermatolog", "spot on my",
    ),
    "Paediatrician": (
        "child", "children", "kid", "baby", "infant", "toddler", "newborn",
        "son", "daughter", "paediatric", "pediatric", "my boy", "my girl",
    ),
}

# Matched on WHOLE WORDS, not substrings.
#
# It was substring matching at first, which forced awkward entries like "my son" instead
# of "son" -- because a bare "son" also matches "person", "reason" and "season". The
# workaround then failed on "his son has a fever", which is exactly the case it existed
# for. Word boundaries let the entries be the words people actually use.
_WORD_CACHE: dict[str, re.Pattern] = {}


def _mentions(text: str, word: str) -> bool:
    pattern = _WORD_CACHE.get(word)
    if pattern is None:
        pattern = re.compile(rf"(?<!\w){re.escape(word)}(?!\w)", re.I)
        _WORD_CACHE[word] = pattern
    return bool(pattern.search(text))

# Anything the table does not recognise goes to a general practitioner, which is what a
# real clinic does too: when in doubt, the GP sees you and refers you on.
DEFAULT_SPECIALTY = "General Practitioner"


def specialty_for(reason: Optional[str], fall_back_to_gp: bool = True) -> Optional[str]:
    """
    Which department a described problem belongs to.

    `fall_back_to_gp` decides what happens when nothing in the table matches, and the two
    callers genuinely want different things:

      True  -- the patient IS booking and has described something. "My back hurts" is not
               in the table, and a general practitioner is the right answer.
      False -- the patient asked a question that might not be about a symptom at all.
               "Do you take my insurance?" matches nothing, and falling back to a GP sent
               a patient asking about insurance to a dermatologist's diary.

    Returns the SPECIALTY, not a doctor -- several doctors may share one.
    """
    if not reason or not reason.strip():
        return None
    text = reason.lower()
    for specialty, words in SYMPTOM_ROUTES.items():
        if any(_mentions(text, word) for word in words):
            return specialty
    return DEFAULT_SPECIALTY if fall_back_to_gp else None


def doctors_in(specialty: str) -> list[Doctor]:
    return [d for d in DOCTORS if d.specialty == specialty]


def suggest_doctors_for(
    reason: Optional[str], fall_back_to_gp: bool = True
) -> tuple[Optional[str], list[Doctor]]:
    """
    Given what the patient said is wrong, return (specialty, doctors who cover it).

    ("Cardiologist", [Dr. Karim Nassar]) for "my heart has been racing".
    (None, []) when nothing was described, or when nothing matched and the caller asked
    not to fall back to a general practitioner.
    """
    specialty = specialty_for(reason, fall_back_to_gp=fall_back_to_gp)
    if specialty is None:
        return None, []
    return specialty, doctors_in(specialty)


# ──────────────────────────────────────────────────────────────────────────────
# Appointments
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Appointment:
    id: str
    patient_id: str
    doctor_id: str
    start: datetime
    reason: Optional[str] = None
    status: str = "booked"          # "booked" | "cancelled"

    @property
    def end(self) -> datetime:
        return self.start + timedelta(minutes=SLOT_MINUTES)


@dataclass
class ClinicDB:
    """
    The appointment book. In-memory only — this is a mock, as the brief allows.

    In production this would be a real database behind the same method names, which is
    why the actions in tools.py go through this object rather than touching a list.
    """
    appointments: list[Appointment] = field(default_factory=list)
    _counter: int = 0

    # ---- reads ----

    def doctor(self, doctor_id: str) -> Optional[Doctor]:
        return next((d for d in DOCTORS if d.id == doctor_id), None)

    def active_for_patient(self, patient_id: str, now: datetime) -> list[Appointment]:
        """
        Upcoming, non-cancelled appointments, soonest first.

        `now` is a parameter rather than read from the clock inside this method. A hidden
        clock makes a function untestable: the same test would pass in the morning and
        fail in the evening. Everything time-dependent in this codebase takes the time in.
        """
        return sorted(
            (a for a in self.appointments
             if a.patient_id == patient_id and a.status == "booked" and a.start >= now),
            key=lambda a: a.start,
        )

    def is_slot_taken(self, doctor_id: str, start: datetime) -> bool:
        return any(a.doctor_id == doctor_id and a.start == start and a.status == "booked"
                   for a in self.appointments)

    def free_slots(self, doctor: Doctor, d: date, now: datetime) -> list[datetime]:
        """
        Every open slot for one doctor on one day, respecting clinic AND doctor hours.

        `now` is passed in so past slots can be excluded deterministically — see the note
        on active_for_patient above.
        """
        hours = clinic_hours_on(d)
        if hours is None or not doctor.works_on(d):
            return []
        clinic_open, clinic_close = hours

        # The doctor is available only where their hours overlap the clinic's.
        start_t = max(clinic_open, doctor.start)
        end_t = min(clinic_close, doctor.end)
        if start_t >= end_t:
            return []

        slots, cursor = [], datetime.combine(d, start_t, tzinfo=TIMEZONE)
        limit = datetime.combine(d, end_t, tzinfo=TIMEZONE)
        while cursor < limit:
            if cursor > now and not self.is_slot_taken(doctor.id, cursor):
                slots.append(cursor)
            cursor += timedelta(minutes=SLOT_MINUTES)
        return slots

    # ---- writes ----

    def add(self, patient_id: str, doctor_id: str, start: datetime,
            reason: Optional[str] = None) -> Appointment:
        self._counter += 1
        appt = Appointment(f"appt_{self._counter:03d}", patient_id, doctor_id, start, reason)
        self.appointments.append(appt)
        return appt


def seeded_db(patient_id: str = "p_001") -> ClinicDB:
    """
    A clinic with a few appointments already in the book.

    Without these, 'cancel my appointment' and 'move my Monday appointment' would have
    nothing to act on, and those are two of the seven intents we must support.
    Appointments are seeded relative to today so they are always in the future.
    """
    db = ClinicDB()
    today = datetime.now(TIMEZONE).date()

    def next_weekday(weekday: int, min_days: int = 1) -> date:
        d = today + timedelta(days=min_days)
        while d.weekday() != weekday:
            d += timedelta(days=1)
        return d

    # Monday 09:00 with Dr. George — the appointment the brief's reschedule example moves.
    db.add(patient_id, "d_george",
           datetime.combine(next_weekday(0), time(9, 0), tzinfo=TIMEZONE),
           reason="follow-up")

    # Wednesday 11:30 with Dr. Karim — the one the brief's cancel example targets.
    db.add(patient_id, "d_karim",
           datetime.combine(next_weekday(2), time(11, 30), tzinfo=TIMEZONE),
           reason="cardiology review")

    return db
