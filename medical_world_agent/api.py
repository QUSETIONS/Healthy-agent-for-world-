from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .orchestrator import MedicalAgentSystem

app = FastAPI(title="Medical World-Model Agent", version="0.1.0")
system = MedicalAgentSystem()


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
def start_session(req: StartSessionRequest) -> StartSessionResponse:
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
def chat(session_id: str, req: ChatRequest) -> ChatResponse:
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
def state(session_id: str) -> dict[str, object]:
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
