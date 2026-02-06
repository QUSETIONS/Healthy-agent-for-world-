from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

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
    turns: list[AgentTurn] = field(default_factory=list)


class MedicalAgentSystem:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionRuntime] = {}

    def start_session(
        self,
        case_id: str,
        random_seed: int | None = None,
        observation_noise: float = 0.15,
        noise_profile: dict[str, Any] | None = None,
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
        )
        self._sessions[runtime.session_id] = runtime
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

        message = self._compose_message(action.kind.value, result.observation, diagnosis, recommendation, safety_notice)
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
        )
        runtime.turns.append(turn)
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
    def _compose_message(
        tool_name: str,
        observation: str,
        diagnosis: str,
        recommendation: str,
        safety_notice: str,
    ) -> str:
        return (
            f"[工具调用] {tool_name}\n"
            f"[观察结果] {observation}\n"
            f"[诊断建议] {diagnosis}\n"
            f"[处置建议] {recommendation}\n"
            f"[安全提示] {safety_notice}"
        )
