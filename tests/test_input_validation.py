import pytest

from medical_world_agent.validators import (
    ValidationError,
    validate_corpus_path,
    validate_evidence_top_k,
    validate_message_length,
    validate_noise_range,
    validate_url,
)


def test_validate_url_accepts_https() -> None:
    assert validate_url("https://example.com/corpus.json") == "https://example.com/corpus.json"


def test_validate_url_accepts_http() -> None:
    assert validate_url("http://example.com/data.json") == "http://example.com/data.json"


def test_validate_url_rejects_file_scheme() -> None:
    with pytest.raises(ValidationError, match="scheme"):
        validate_url("file:///etc/passwd")


def test_validate_url_rejects_private_ip() -> None:
    with pytest.raises(ValidationError, match="private"):
        validate_url("http://127.0.0.1/secret")


def test_validate_url_rejects_empty() -> None:
    with pytest.raises(ValidationError, match="empty"):
        validate_url("")


def test_validate_corpus_path_accepts_valid() -> None:
    assert validate_corpus_path("data/corpus.json") == "data/corpus.json"


def test_validate_corpus_path_rejects_traversal() -> None:
    with pytest.raises(ValidationError, match="traversal"):
        validate_corpus_path("../../etc/passwd.json")


def test_validate_corpus_path_rejects_non_json() -> None:
    with pytest.raises(ValidationError, match=".json"):
        validate_corpus_path("data/corpus.txt")


def test_validate_corpus_path_rejects_unknown_root() -> None:
    with pytest.raises(ValidationError, match="allowed roots"):
        validate_corpus_path("tmp/corpus.json")


def test_validate_corpus_path_rejects_absolute_path() -> None:
    with pytest.raises(ValidationError, match="Absolute corpus paths"):
        validate_corpus_path("C:/temp/corpus.json")


def test_validate_message_length_accepts_short() -> None:
    assert validate_message_length("hello") == "hello"


def test_validate_message_length_rejects_long() -> None:
    with pytest.raises(ValidationError, match="exceeds"):
        validate_message_length("x" * 2001)


def test_validate_noise_range_valid() -> None:
    assert validate_noise_range(0.0) == 0.0
    assert validate_noise_range(0.5) == 0.5
    assert validate_noise_range(1.0) == 1.0


def test_validate_noise_range_invalid() -> None:
    with pytest.raises(ValidationError):
        validate_noise_range(-0.1)
    with pytest.raises(ValidationError):
        validate_noise_range(1.1)


def test_validate_evidence_top_k_valid() -> None:
    assert validate_evidence_top_k(1) == 1
    assert validate_evidence_top_k(10) == 10
    assert validate_evidence_top_k(20) == 20


def test_validate_evidence_top_k_invalid() -> None:
    with pytest.raises(ValidationError):
        validate_evidence_top_k(0)
    with pytest.raises(ValidationError):
        validate_evidence_top_k(21)


import os
from fastapi.testclient import TestClient
from medical_world_agent.api import app


def _client() -> TestClient:
    os.environ.pop("HEALTHY_AGENT_API_KEY", None)
    return TestClient(app)


def test_api_rejects_too_long_message() -> None:
    client = _client()
    start = client.post("/sessions/start", json={"case_id": "resp_001"})
    sid = start.json()["session_id"]
    resp = client.post(
        f"/sessions/{sid}/chat",
        json={"message": "x" * 2001},
    )
    assert resp.status_code == 422


def test_api_rejects_invalid_noise_range() -> None:
    client = _client()
    resp = client.post(
        "/sessions/start",
        json={"case_id": "resp_001", "observation_noise": 2.0},
    )
    assert resp.status_code == 422


def test_api_rejects_invalid_evidence_top_k() -> None:
    client = _client()
    resp = client.post(
        "/sessions/start",
        json={"case_id": "resp_001", "evidence_top_k": 0},
    )
    assert resp.status_code == 422


def test_api_delete_session() -> None:
    client = _client()
    start = client.post("/sessions/start", json={"case_id": "resp_001"})
    sid = start.json()["session_id"]

    resp = client.delete(f"/sessions/{sid}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"


def test_api_state_returns_sanitized_demographics() -> None:
    client = _client()
    start = client.post("/sessions/start", json={"case_id": "chest_pain_001"})
    sid = start.json()["session_id"]

    resp = client.get(f"/sessions/{sid}/state")
    assert resp.status_code == 200
    demographics = resp.json()["demographics"]
    assert demographics["age"] == "60-69"
    assert demographics["sex"] == "male"
