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
    This service keeps NO memory of its own between messages.

    Most hosts today do not keep one computer running for your site. They start a small
    worker when a message arrives and throw it away afterwards, so the worker answering
    "yes" may never have seen the offer the patient is saying yes to.

    So the conversation's memory travels with the patient instead: it goes out with every
    reply as a `state` string, and the browser sends it back with the next message. The
    string is stamped so it cannot be edited -- see `clinikit/agent/state.py`.

    The upside beyond hosting: any number of servers can answer any message, and a
    restart loses nothing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..agent.backends import available_backends, patient_facing_backends
from ..agent.examples import SCENARIOS
from ..agent.clinic import DOCTORS, describe_opening_hours
from ..agent.session import Session
from ..agent.state import TamperedState, open_sealed, restore, seal

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
    state: Optional[str] = Field(
        default=None,
        description="The conversation so far, as handed back by the previous reply. "
                    "Omit it to start a new conversation.",
    )
    backend: Optional[str] = Field(default=None, description="'auto', 'gemini' or 'rules'")


class ChatResponse(BaseModel):
    state: str
    """The conversation so far. Send it back with the next message, unchanged. It is
    stamped, so an edited one is refused."""

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

def _session_for(state: Optional[str], backend: Optional[str]) -> Session:
    """
    Build the session this message belongs to.

    Always a brand-new object. If the browser sent the conversation back, its memory is
    poured into the new object first. Nothing is kept between requests.
    """
    wanted = backend or DEFAULT_BACKEND
    if wanted not in available_backends():
        wanted = DEFAULT_BACKEND

    session = Session(backend=wanted, natural_replies=NATURAL_REPLIES)

    if state:
        try:
            restore(session, open_sealed(state))
        except TamperedState as exc:
            # Refuse rather than quietly starting over. A ticket that fails its stamp is
            # either a bug or someone editing their own memory to invent an offer, and
            # both deserve to be visible instead of silently swallowed.
            raise HTTPException(400, f"Conversation state rejected: {exc}") from exc

    return session


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
    session = _session_for(request.state, request.backend)
    turn = session.handle(request.message)
    return ChatResponse(
        state=seal(session),
        appointments=_appointments(session),
        **turn.as_dict(),
    )


@app.post("/api/start")
def start(backend: Optional[str] = None) -> dict:
    """
    Open a conversation without saying anything.

    The web page used to send a fake "hello" just to get a session id, which meant every
    visitor waited two seconds and burned a request before the page was even usable. This
    creates the session and returns the appointment book, with no model call at all.
    """
    session = _session_for(None, backend)
    return {
        "state": seal(session),
        "appointments": _appointments(session),
        "greeting": "Hello — how can the clinic help today?",
    }


@app.post("/api/reset")
def reset() -> dict:
    """
    Start again.

    There is nothing on the server to clear, so this just hands back a fresh, empty
    conversation. The browser throws away the old one by replacing it with this.
    """
    session = _session_for(None, None)
    return {"ok": True, "state": seal(session),
            "appointments": _appointments(session)}


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


class AuditRequest(BaseModel):
    state: str


@app.post("/api/audit")
def audit(request: AuditRequest) -> dict:
    """
    What this conversation looks like from the clinic's side. A real clinic needs this.

    The per-turn reasoning -- what was understood, what was decided, which guarantee
    applied -- comes back with every single reply, so the caller already has it. What
    this adds is the signed record: the transcript and the appointment book as the
    server sees them, from a ticket that cannot have been edited.
    """
    session = _session_for(request.state, None)
    return {
        "history": session.history,
        "appointments": _appointments(session),
        "pending_offer": session.pending_offer.summary if session.pending_offer else None,
    }


# ──────────────────────────────────────────────────────────────────────────────
# The web page, served from the same process
# ──────────────────────────────────────────────────────────────────────────────

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")
