from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ToolKind(str, Enum):
    ASK_QUESTION = "ask_question"
    ORDER_TEST = "order_test"
    RECOMMEND_PLAN = "recommend_plan"


@dataclass
class ToolAction:
    kind: ToolKind
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    kind: ToolKind
    observation: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PatientCase:
    case_id: str
    demographics: dict[str, Any]
    symptoms: list[str]
    qa: dict[str, str]
    tests: dict[str, str]
    final_diagnosis: str


@dataclass
class PatientState:
    case_id: str
    demographics: dict[str, Any]
    symptoms: list[str]
    known_facts: dict[str, str] = field(default_factory=dict)
    completed_tests: dict[str, str] = field(default_factory=dict)
    history: list[str] = field(default_factory=list)


@dataclass
class AgentTurn:
    message: str
    tool_action: ToolAction
    tool_result: ToolResult
    diagnosis: str
    recommendation: str
    urgency: str
    safety_notice: str
    emergency: bool
    red_flags: list[str] = field(default_factory=list)
    dangerous_miss: bool = False
    guideline_refs: list[str] = field(default_factory=list)
    evidence_chain: list[str] = field(default_factory=list)
    diagnosis_confidence: float = 0.0
    escalate_to_human: bool = False
    refusal: bool = False
    refusal_reason: str = ""
