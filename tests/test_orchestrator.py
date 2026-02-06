from medical_world_agent.orchestrator import MedicalAgentSystem


def test_chest_pain_flow_tool_calls_and_recommendation() -> None:
    system = MedicalAgentSystem()
    session_id = system.start_session("chest_pain_001")

    turn1 = system.chat(session_id, "患者胸痛，请先处理")
    assert turn1.tool_action.kind.value == "order_test"
    assert "心电图" in turn1.tool_result.observation

    turn2 = system.chat(session_id, "继续")
    assert turn2.tool_action.kind.value == "order_test"
    assert "肌钙蛋白" in turn2.tool_result.observation

    turn3 = system.chat(session_id, "请给出建议")
    assert turn3.tool_action.kind.value == "recommend_plan"
    assert "心肌梗死" in turn3.recommendation
    assert turn3.emergency is True
    assert turn3.red_flags
    assert turn3.dangerous_miss is False
    assert turn3.guideline_refs
    assert any("acs" in ref.lower() or "胸痛" in ref for ref in turn3.guideline_refs)
    assert turn3.escalate_to_human is True
    assert turn3.refusal is True
    assert "转人工" in turn3.refusal_reason
    assert turn3.evidence_chain
    assert 0.0 <= turn3.diagnosis_confidence <= 1.0


def test_stroke_flow_triggers_emergency_red_flag() -> None:
    system = MedicalAgentSystem()
    session_id = system.start_session("stroke_001")
    system.chat(session_id, "继续")
    system.chat(session_id, "继续")
    turn = system.chat(session_id, "请给出建议")
    assert "脑卒中" in turn.diagnosis
    assert turn.emergency is True
    assert any("卒中" in flag for flag in turn.red_flags)
    assert turn.dangerous_miss is False
    assert turn.escalate_to_human is True
    assert turn.refusal is True
    assert turn.evidence_chain
    assert 0.0 <= turn.diagnosis_confidence <= 1.0


def test_resp_flow_no_forced_handoff_after_evidence_complete() -> None:
    system = MedicalAgentSystem()
    session_id = system.start_session("resp_001", observation_noise=0.0)
    system.chat(session_id, "继续")
    system.chat(session_id, "继续")
    turn = system.chat(session_id, "请给出建议")
    assert "肺炎" in turn.diagnosis
    assert turn.emergency is False
    assert turn.escalate_to_human is False
    assert turn.refusal is False
    assert turn.evidence_chain
    assert turn.diagnosis_confidence >= 0.5


def test_insufficient_evidence_forces_refusal() -> None:
    system = MedicalAgentSystem()
    session_id = system.start_session("resp_001", observation_noise=0.0)
    turn = system.chat(session_id, "请给出建议")
    assert "证据不足" in turn.diagnosis
    assert turn.refusal is True
    assert turn.escalate_to_human is True
