from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import time

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from medical_world_agent.orchestrator import MedicalAgentSystem


CASES = ["chest_pain_001", "resp_001", "abd_001", "uti_001", "stroke_001"]


def _run_max_sessions_boundary(concurrency: int = 80, max_sessions: int = 40) -> dict[str, object]:
    system = MedicalAgentSystem(max_sessions=max_sessions, session_ttl_seconds=1800.0)
    success = 0
    limit_errors = 0
    other_errors: list[str] = []

    def _task(i: int) -> str:
        case_id = CASES[i % len(CASES)]
        try:
            system.start_session(case_id)
            return "ok"
        except ValueError as exc:
            msg = str(exc)
            if "Maximum session limit" in msg:
                return "limit"
            return f"err:{msg}"

    with ThreadPoolExecutor(max_workers=min(concurrency, 32)) as ex:
        futures = [ex.submit(_task, i) for i in range(concurrency)]
        for fut in as_completed(futures):
            result = fut.result()
            if result == "ok":
                success += 1
            elif result == "limit":
                limit_errors += 1
            else:
                other_errors.append(result)

    passed = success == max_sessions and limit_errors == (concurrency - max_sessions) and not other_errors
    return {
        "passed": passed,
        "concurrency": concurrency,
        "max_sessions": max_sessions,
        "success_count": success,
        "limit_error_count": limit_errors,
        "other_errors": other_errors,
    }


def _run_ttl_boundary(ttl_seconds: float = 0.2, warm_sessions: int = 20) -> dict[str, object]:
    system = MedicalAgentSystem(max_sessions=100, session_ttl_seconds=ttl_seconds)
    for i in range(warm_sessions):
        system.start_session(CASES[i % len(CASES)])

    before = len(system.list_sessions())
    time.sleep(ttl_seconds + 0.15)
    after = len(system.list_sessions())

    passed = before == warm_sessions and after == 0
    return {
        "passed": passed,
        "ttl_seconds": ttl_seconds,
        "warm_sessions": warm_sessions,
        "sessions_before_expire": before,
        "sessions_after_expire": after,
    }


def main() -> None:
    max_result = _run_max_sessions_boundary()
    ttl_result = _run_ttl_boundary()

    payload = {
        "max_sessions_boundary": max_result,
        "ttl_boundary": ttl_result,
        "passed": bool(max_result["passed"] and ttl_result["passed"]),
    }
    out_path = ROOT / "reports" / "stress_sessions_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
