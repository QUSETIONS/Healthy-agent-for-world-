from __future__ import annotations

import os
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .models import AgentTurn
from .orchestrator import MedicalAgentSystem

app = FastAPI(title="Medical World-Model Agent", version="0.1.0")
system = MedicalAgentSystem()
_static_dir = Path(__file__).resolve().parents[1] / "static"
app.mount("/static", StaticFiles(directory=_static_dir), name="static")


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    required = os.getenv("HEALTHY_AGENT_API_KEY", "")
    if required and x_api_key != required:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _turn_to_dict(t: AgentTurn) -> dict[str, object]:
    return {
        "message": t.message,
        "tool": t.tool_action.kind.value,
        "diagnosis": t.diagnosis,
        "diagnosis_confidence": t.diagnosis_confidence,
        "recommendation": t.recommendation,
        "urgency": t.urgency,
        "safety_notice": t.safety_notice,
        "emergency": t.emergency,
        "red_flags": t.red_flags,
        "dangerous_miss": t.dangerous_miss,
        "guideline_refs": t.guideline_refs,
        "evidence_chain": t.evidence_chain,
        "escalate_to_human": t.escalate_to_human,
        "refusal": t.refusal,
        "refusal_reason": t.refusal_reason,
    }


@app.get("/")
def dashboard() -> FileResponse:
    return FileResponse(_static_dir / "dashboard.html")


class StartSessionRequest(BaseModel):
    case_id: str
    random_seed: int | None = None
    observation_noise: float = 0.15
    noise_profile: dict[str, object] | None = None
    knowledge_corpus_path: str | None = None
    knowledge_corpus_url: str | None = None
    evidence_top_k: int = 3


class StartSessionResponse(BaseModel):
    session_id: str


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    message: str
    tool: str
    urgency: str
    diagnosis: str
    recommendation: str
    emergency: bool
    red_flags: list[str]
    dangerous_miss: bool
    guideline_refs: list[str]
    evidence_chain: list[str]
    diagnosis_confidence: float
    escalate_to_human: bool
    refusal: bool
    refusal_reason: str


@app.post("/sessions/start", response_model=StartSessionResponse)
def start_session(req: StartSessionRequest, _: None = Depends(require_api_key)) -> StartSessionResponse:
    try:
        session_id = system.start_session(
            req.case_id,
            random_seed=req.random_seed,
            observation_noise=req.observation_noise,
            noise_profile=req.noise_profile,
            knowledge_corpus_path=req.knowledge_corpus_path,
            knowledge_corpus_url=req.knowledge_corpus_url,
            evidence_top_k=req.evidence_top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return StartSessionResponse(session_id=session_id)


@app.post("/sessions/{session_id}/chat", response_model=ChatResponse)
def chat(session_id: str, req: ChatRequest, _: None = Depends(require_api_key)) -> ChatResponse:
    try:
        turn = system.chat(session_id, req.message)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ChatResponse(
        message=turn.message,
        tool=turn.tool_action.kind.value,
        urgency=turn.urgency,
        diagnosis=turn.diagnosis,
        recommendation=turn.recommendation,
        emergency=turn.emergency,
        red_flags=turn.red_flags,
        dangerous_miss=turn.dangerous_miss,
        guideline_refs=turn.guideline_refs,
        evidence_chain=turn.evidence_chain,
        diagnosis_confidence=turn.diagnosis_confidence,
        escalate_to_human=turn.escalate_to_human,
        refusal=turn.refusal,
        refusal_reason=turn.refusal_reason,
    )


@app.get("/sessions/{session_id}/state")
def state(session_id: str, _: None = Depends(require_api_key)) -> dict[str, object]:
    try:
        state_obj = system.state(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "case_id": state_obj.case_id,
        "demographics": state_obj.demographics,
        "symptoms": state_obj.symptoms,
        "known_facts": state_obj.known_facts,
        "completed_tests": state_obj.completed_tests,
        "history": state_obj.history,
    }


@app.get("/sessions")
def list_sessions(_: None = Depends(require_api_key)) -> dict[str, object]:
    return {"items": system.list_sessions()}


@app.get("/sessions/{session_id}/turns")
def list_turns(session_id: str, _: None = Depends(require_api_key)) -> dict[str, object]:
    try:
        turns = system.turns(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"items": [_turn_to_dict(t) for t in turns]}


@app.get("/sessions/{session_id}/pathway")
def get_pathway(session_id: str, _: None = Depends(require_api_key)) -> dict[str, object]:
    try:
        return system.pathway(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
