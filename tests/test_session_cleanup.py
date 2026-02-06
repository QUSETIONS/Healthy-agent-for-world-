from medical_world_agent.orchestrator import MedicalAgentSystem

import pytest


def test_delete_session_removes_it() -> None:
    system = MedicalAgentSystem()
    sid = system.start_session("resp_001")
    assert len(system.list_sessions()) == 1

    system.delete_session(sid)
    assert len(system.list_sessions()) == 0


def test_delete_unknown_session_raises() -> None:
    system = MedicalAgentSystem()
    with pytest.raises(ValueError, match="Unknown session_id"):
        system.delete_session("nonexistent")


def test_max_sessions_limit_enforced() -> None:
    system = MedicalAgentSystem(max_sessions=3)
    system.start_session("resp_001")
    system.start_session("abd_001")
    system.start_session("uti_001")

    with pytest.raises(ValueError, match="Maximum session limit"):
        system.start_session("chest_pain_001")


def test_expired_sessions_cleaned_up() -> None:
    system = MedicalAgentSystem(session_ttl_seconds=9999)
    system.start_session("resp_001")
    system.start_session("abd_001")
    assert len(system.list_sessions()) == 2

    system._session_ttl_seconds = 0.0
    system._cleanup_expired()
    assert len(system.list_sessions()) == 0


def test_chat_updates_last_accessed() -> None:
    system = MedicalAgentSystem(session_ttl_seconds=9999)
    sid = system.start_session("resp_001")
    runtime = system._sessions[sid]
    old_ts = runtime.last_accessed_at

    system.chat(sid, "请继续")
    assert runtime.last_accessed_at >= old_ts


def test_state_triggers_expired_session_cleanup() -> None:
    system = MedicalAgentSystem(session_ttl_seconds=0.0)
    sid = system.start_session("resp_001")

    with pytest.raises(ValueError, match="Unknown session_id"):
        system.state(sid)


def test_concurrent_max_sessions_boundary() -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    system = MedicalAgentSystem(max_sessions=8)

    def start_one(i: int) -> str:
        case_ids = ["resp_001", "abd_001", "uti_001", "stroke_001", "chest_pain_001"]
        try:
            system.start_session(case_ids[i % len(case_ids)])
            return "ok"
        except ValueError as exc:
            if "Maximum session limit" in str(exc):
                return "limit"
            return "err"

    results = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        futures = [ex.submit(start_one, i) for i in range(20)]
        for fut in as_completed(futures):
            results.append(fut.result())

    assert results.count("ok") == 8
    assert results.count("limit") == 12
    assert results.count("err") == 0
