from pathlib import Path

from medical_world_agent.orchestrator import MedicalAgentSystem


def test_audit_log_written_for_session_and_chat(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    system = MedicalAgentSystem(audit_log_path=str(audit_path))

    session_id = system.start_session("resp_001", observation_noise=0.0)
    system.chat(session_id, "请继续")

    assert audit_path.exists()
    lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 2
    assert "session_start" in lines[0]
    assert "chat_turn" in lines[1]
