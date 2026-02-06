from fastapi.testclient import TestClient

from medical_world_agent.api import app


def test_api_start_and_chat() -> None:
    client = TestClient(app)

    start = client.post("/sessions/start", json={"case_id": "resp_001"})
    assert start.status_code == 200
    session_id = start.json()["session_id"]

    chat = client.post(f"/sessions/{session_id}/chat", json={"message": "请继续"})
    assert chat.status_code == 200
    payload = chat.json()
    assert payload["tool"] in {"order_test", "ask_question", "recommend_plan"}
    assert payload["recommendation"]
    assert "diagnosis" in payload
    assert "emergency" in payload
    assert "red_flags" in payload
    assert "dangerous_miss" in payload
    assert "guideline_refs" in payload
    assert "evidence_chain" in payload
    assert "diagnosis_confidence" in payload
    assert "escalate_to_human" in payload
    assert "refusal" in payload
    assert "refusal_reason" in payload
