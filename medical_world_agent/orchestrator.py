from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from .audit import AuditLogger
from .knowledge import GuidelineRetriever
from .models import AgentTurn, PatientState
from .subagents import DiagnosticAgent, SafetyAgent, TriageAgent
from .tools import ToolRegistry, build_default_registry
from .world_model import MedicalWorldModel


@dataclass
class SessionRuntime:
    session_id: str
    world_model: MedicalWorldModel
    tools: ToolRegistry
    triage: TriageAgent
    diagnostic: DiagnosticAgent
    safety: SafetyAgent
    retriever: GuidelineRetriever
    evidence_top_k: int
    audit: AuditLogger
    turns: list[AgentTurn] = field(default_factory=list)


class MedicalAgentSystem:
    def __init__(self, audit_log_path: str = "reports/audit_log.jsonl") -> None:
        self._sessions: dict[str, SessionRuntime] = {}
        self._audit = AuditLogger(audit_log_path)

    def start_session(
        self,
        case_id: str,
        random_seed: int | None = None,
        observation_noise: float = 0.15,
        noise_profile: dict[str, Any] | None = None,
        knowledge_corpus_path: str | None = None,
        knowledge_corpus_url: str | None = None,
        evidence_top_k: int = 3,
    ) -> str:
        world_model = MedicalWorldModel(
            random_seed=random_seed,
            observation_noise=observation_noise,
            noise_profile=noise_profile,
        )
        world_model.reset(case_id, random_seed=random_seed)
        runtime = SessionRuntime(
            session_id=str(uuid4()),
            world_model=world_model,
            tools=build_default_registry(world_model),
            triage=TriageAgent(),
            diagnostic=DiagnosticAgent(),
            safety=SafetyAgent(),
            retriever=GuidelineRetriever(corpus_path=knowledge_corpus_path, corpus_url=knowledge_corpus_url),
            evidence_top_k=max(1, evidence_top_k),
            audit=self._audit,
        )
        self._sessions[runtime.session_id] = runtime
        runtime.audit.write(
            "session_start",
            {
                "session_id": runtime.session_id,
                "case_id": case_id,
                "observation_noise": observation_noise,
                "evidence_top_k": runtime.evidence_top_k,
            },
        )
        return runtime.session_id

    def chat(self, session_id: str, user_message: str) -> AgentTurn:
        runtime = self._sessions.get(session_id)
        if runtime is None:
            raise ValueError(f"Unknown session_id: {session_id}")

        state = runtime.world_model.get_state()
        urgency = runtime.triage.assess(state)
        action = runtime.diagnostic.choose_action(state, user_message)
        result = runtime.tools.invoke(action)

        latest_state = runtime.world_model.get_state()
        diagnosis = runtime.diagnostic.infer_diagnosis(latest_state)
        recommendation = runtime.diagnostic.treatment_plan(diagnosis)
        safety_notice, emergency, red_flags, dangerous_miss = runtime.safety.evaluate(latest_state, diagnosis, urgency)
        guideline_refs, guideline_confidence = self._guideline_refs(runtime, latest_state, diagnosis)
        evidence_chain = runtime.diagnostic.build_evidence_chain(latest_state, diagnosis)
        diagnosis_confidence = runtime.diagnostic.estimate_confidence(latest_state, diagnosis, guideline_confidence)
        escalate_to_human, refusal, refusal_reason = runtime.safety.handoff_decision(
            diagnosis=diagnosis,
            emergency=emergency,
            dangerous_miss=dangerous_miss,
            evidence_confidence=diagnosis_confidence,
        )

        message = self._compose_message(
            action.kind.value,
            result.observation,
            diagnosis,
            recommendation,
            safety_notice,
            guideline_refs,
            evidence_chain,
            diagnosis_confidence,
            escalate_to_human,
            refusal,
            refusal_reason,
        )
        turn = AgentTurn(
            message=message,
            tool_action=action,
            tool_result=result,
            diagnosis=diagnosis,
            recommendation=recommendation,
            urgency=urgency,
            safety_notice=safety_notice,
            emergency=emergency,
            red_flags=red_flags,
            dangerous_miss=dangerous_miss,
            guideline_refs=guideline_refs,
            evidence_chain=evidence_chain,
            diagnosis_confidence=diagnosis_confidence,
            escalate_to_human=escalate_to_human,
            refusal=refusal,
            refusal_reason=refusal_reason,
        )
        runtime.turns.append(turn)
        runtime.audit.write(
            "chat_turn",
            {
                "session_id": session_id,
                "user_message": user_message,
                "tool": action.kind.value,
                "diagnosis": diagnosis,
                "diagnosis_confidence": diagnosis_confidence,
                "emergency": emergency,
                "refusal": refusal,
                "escalate_to_human": escalate_to_human,
            },
        )
        return turn

    def state(self, session_id: str) -> PatientState:
        runtime = self._sessions.get(session_id)
        if runtime is None:
            raise ValueError(f"Unknown session_id: {session_id}")
        return runtime.world_model.get_state()

    def true_diagnosis(self, session_id: str) -> str:
        runtime = self._sessions.get(session_id)
        if runtime is None:
            raise ValueError(f"Unknown session_id: {session_id}")
        return runtime.world_model.true_diagnosis()

    def available_cases(self) -> list[str]:
        probe = MedicalWorldModel()
        return probe.list_case_ids()

    @staticmethod
    def _guideline_refs(runtime: SessionRuntime, state: PatientState, diagnosis: str) -> tuple[list[str], float]:
        query = f"{diagnosis} {' '.join(state.symptoms)} {' '.join(state.completed_tests.keys())}"
        hits = runtime.retriever.retrieve(query, top_k=runtime.evidence_top_k)
        refs = [
            f"{h.snippet.guideline_id}: {h.snippet.title} | {h.snippet.source} | confidence={h.confidence:.3f}"
            for h in hits
        ]
        confidence = max((h.confidence for h in hits), default=0.0)
        return (refs, confidence)

    @staticmethod
    def _compose_message(
        tool_name: str,
        observation: str,
        diagnosis: str,
        recommendation: str,
        safety_notice: str,
        guideline_refs: list[str],
        evidence_chain: list[str],
        diagnosis_confidence: float,
        escalate_to_human: bool,
        refusal: bool,
        refusal_reason: str,
    ) -> str:
        refs = "\n".join(f"- {x}" for x in guideline_refs) if guideline_refs else "- 暂无匹配指南"
        chain = "\n".join(f"- {x}" for x in evidence_chain) if evidence_chain else "- 无"
        handoff = "是" if escalate_to_human else "否"
        refused = "是" if refusal else "否"
        refusal_line = refusal_reason if refusal_reason else "-"
        return (
            f"[工具调用] {tool_name}\n"
            f"[观察结果] {observation}\n"
            f"[诊断建议] {diagnosis}\n"
            f"[诊断置信度] {diagnosis_confidence:.3f}\n"
            f"[证据链]\n{chain}\n"
            f"[处置建议] {recommendation}\n"
            f"[安全提示] {safety_notice}\n"
            f"[参考指南]\n{refs}\n"
            f"[转人工] {handoff}\n"
            f"[拒答] {refused}\n"
            f"[拒答原因] {refusal_line}"
        )
