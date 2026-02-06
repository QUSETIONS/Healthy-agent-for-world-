from fastapi.testclient import TestClient
import os

from medical_world_agent.api import app


def test_api_start_and_chat() -> None:
    os.environ.pop("HEALTHY_AGENT_API_KEY", None)
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

    sessions = client.get("/sessions")
    assert sessions.status_code == 200
    assert sessions.json()["items"]

    turns = client.get(f"/sessions/{session_id}/turns")
    assert turns.status_code == 200
    assert turns.json()["items"]

    pathway = client.get(f"/sessions/{session_id}/pathway")
    assert pathway.status_code == 200
    p = pathway.json()
    assert p["case_id"] == "resp_001"
    assert p["total_steps"] >= 1
    assert 0.0 <= p["progress"] <= 1.0


def test_dashboard_route_served() -> None:
    os.environ.pop("HEALTHY_AGENT_API_KEY", None)
    client = TestClient(app)
    page = client.get("/")
    assert page.status_code == 200
    assert '<html lang="zh-CN">' in page.text


def test_api_key_guard() -> None:
    os.environ["HEALTHY_AGENT_API_KEY"] = "secret-key"
    client = TestClient(app)

    unauthorized = client.post("/sessions/start", json={"case_id": "resp_001"})
    assert unauthorized.status_code == 401

    authorized = client.post(
        "/sessions/start",
        json={"case_id": "resp_001"},
        headers={"X-API-Key": "secret-key"},
    )
    assert authorized.status_code == 200

    os.environ.pop("HEALTHY_AGENT_API_KEY", None)
