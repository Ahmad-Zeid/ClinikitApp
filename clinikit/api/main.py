"""
A small web service wrapping the agent.

    .venv/bin/uvicorn clinikit.api.main:app --reload
    open http://127.0.0.1:8000

WHY THIS EXISTS
    The assessment asks how the system would fit into a real product. This is the answer
    in working code rather than a paragraph: the agent behind an HTTP endpoint, with the
    web page served from the same process.

WHAT EACH REPLY CARRIES
    Every response includes not just the reply, but what was understood, what was decided,
    why, which guarantee applied, and whether the appointment book changed. The web page
    needs the reasoning, not just the sentence -- and so would a real clinic's audit log.

SESSIONS
    Kept in a dictionary in memory. Fine for a demo; a real deployment would use Redis or
    a database so that sessions survive a restart and work across several servers. Because
    all the memory lives in one class, that swap touches one file.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..agent.backends import available_backends, patient_facing_backends
from ..agent.cli import SCENARIOS
from ..agent.clinic import DOCTORS, describe_opening_hours
from ..agent.session import Session

WEB_DIR = Path(__file__).resolve().parents[2] / "web"

app = FastAPI(
    title="CliniKit assistant",
    description="Conversational agent for a medical clinic. Part 1 of the CliniKit AI assessment.",
    version="1.0.0",
)

# Open CORS because this is a demo that may be opened from a file:// page or a different
# host. A real clinic would list its own domains here and nothing else.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_SESSIONS: dict[str, Session] = {}

def _default_backend() -> str:
    """
    Which reader the web page uses unless told otherwise.

    This used to name a provider directly, which broke badly: it still said "gemini" long
    after Groq became the main one, so when Gemini ran out of quota every visitor was
    handed to a human. "auto" asks the chain, so adding or removing a provider needs no
    change here.
    """
    chosen = os.environ.get("CLINIKIT_BACKEND")
    if chosen:
        return chosen
    usable = patient_facing_backends()
    return usable[0] if usable else "rules"


DEFAULT_BACKEND = _default_backend()
NATURAL_REPLIES = os.environ.get("CLINIKIT_NATURAL_REPLIES", "1") != "0"


# ──────────────────────────────────────────────────────────────────────────────
# Request and response shapes
# ──────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: Optional[str] = None
    backend: Optional[str] = Field(default=None, description="'rules' or 'gemini'")


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    backend: str
    seconds: float
    extraction: dict
    action: str
    reason: str
    guarantee: Optional[str]
    was_write: bool
    changed_the_book: bool
    detail: str
    slots: list[dict]
    """Free times to offer. Each is {when, label, doctor} -- the doctor is included
    because a bare list of times tells the patient nothing about who they would see."""
    reply_source: str
    reply_rejected_because: Optional[str]
    degraded_reason: Optional[str]
    appointments: list[dict]


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────

def _get_session(session_id: Optional[str], backend: Optional[str]) -> tuple[str, Session]:
    if session_id and session_id in _SESSIONS:
        return session_id, _SESSIONS[session_id]

    new_id = session_id or uuid.uuid4().hex[:12]
    wanted = backend or DEFAULT_BACKEND
    if wanted not in available_backends():
        wanted = DEFAULT_BACKEND
    _SESSIONS[new_id] = Session(backend=wanted, natural_replies=NATURAL_REPLIES)
    return new_id, _SESSIONS[new_id]


def _appointments(session: Session) -> list[dict]:
    now = session._clock()
    return [
        {
            "id": a.id,
            "doctor": session.db.doctor(a.doctor_id).full_name,
            "specialty": session.db.doctor(a.doctor_id).specialty,
            "when": a.start.isoformat(),
            "label": f"{a.start:%A %d %B, %H:%M}",
        }
        for a in session.db.active_for_patient(session.patient_id, now)
    ]


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Send one patient message and get back the reply plus the full reasoning."""
    session_id, session = _get_session(request.session_id, request.backend)
    turn = session.handle(request.message)
    payload = turn.as_dict()
    return ChatResponse(
        session_id=session_id, appointments=_appointments(session), **payload
    )


@app.post("/api/start")
def start(backend: Optional[str] = None) -> dict:
    """
    Open a conversation without saying anything.

    The web page used to send a fake "hello" just to get a session id, which meant every
    visitor waited two seconds and burned a request before the page was even usable. This
    creates the session and returns the appointment book, with no model call at all.
    """
    session_id, session = _get_session(None, backend)
    return {
        "session_id": session_id,
        "appointments": _appointments(session),
        "greeting": "Hello — how can the clinic help today?",
    }


@app.post("/api/reset")
def reset(session_id: Optional[str] = None) -> dict:
    """Clear a conversation and restore the appointment book to its starting state."""
    if session_id and session_id in _SESSIONS:
        del _SESSIONS[session_id]
    return {"ok": True}


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "backends": available_backends(),
        "default_backend": DEFAULT_BACKEND,
        "natural_replies": NATURAL_REPLIES,
    }


@app.get("/api/quota")
def quota() -> dict:
    """
    How much of today's free allowance is left, per provider.

    Worth having visible rather than discovered. The two free tiers run out in completely
    different ways -- Groq by tokens, Gemini by request count -- and knowing which one is
    close to the edge explains a slow reply far better than guessing does.
    """
    from ..agent.backends.openai_compat import DAILY_USAGE, PROVIDERS

    spent = DAILY_USAGE.summary()
    out = []
    for provider_name, provider in PROVIDERS.items():
        for model in provider.models:
            used = spent.get(f"{provider_name}:{model}")
            if not used:
                continue
            row = {
                "provider": provider_name,
                "model": model,
                "requests_used": used["requests"],
                "requests_limit": provider.requests_per_day,
                "tokens_used": used["tokens"],
                "tokens_limit": provider.tokens_per_day,
            }
            if provider.tokens_per_day:
                row["percent_used"] = round(100 * used["tokens"] / provider.tokens_per_day)
            else:
                row["percent_used"] = round(
                    100 * used["requests"] / provider.requests_per_day)
            out.append(row)
    return {"date": DAILY_USAGE._today, "models": out}


@app.get("/api/scenarios")
def scenarios() -> dict:
    """Canned conversations, so a reviewer can click rather than think up messages."""
    return {
        "scenarios": [
            {"name": name, "messages": messages} for name, messages in SCENARIOS.items()
        ]
    }


@app.get("/api/clinic")
def clinic() -> dict:
    return {
        "doctors": [
            {"name": d.full_name, "specialty": d.specialty,
             "hours": f"{d.start:%H:%M}-{d.end:%H:%M}"}
            for d in DOCTORS
        ],
        "opening_hours": describe_opening_hours(),
    }


@app.get("/api/audit")
def audit(session_id: str) -> dict:
    """Everything that happened in one conversation. A real clinic would need this."""
    session = _SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(404, "No such session")
    return {"turns": session.audit_log()}


# ──────────────────────────────────────────────────────────────────────────────
# The web page, served from the same process
# ──────────────────────────────────────────────────────────────────────────────

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")
