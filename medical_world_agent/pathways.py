from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PathwayStep:
    order: int
    code: str
    title: str
    required_tests: tuple[str, ...] = ()


def pathway_for_case(case_id: str) -> list[PathwayStep]:
    pathways: dict[str, list[PathwayStep]] = {
        "chest_pain_001": [
            PathwayStep(1, "triage", "Immediate triage and risk screening"),
            PathwayStep(2, "ecg", "12-lead ECG", ("ecg",)),
            PathwayStep(3, "troponin", "Cardiac biomarker", ("troponin",)),
            PathwayStep(4, "decision", "Reperfusion pathway decision"),
        ],
        "stroke_001": [
            PathwayStep(1, "triage", "Neurological red-flag screening"),
            PathwayStep(2, "head_ct", "Head CT to exclude bleeding", ("head_ct",)),
            PathwayStep(3, "nihss", "NIHSS neurological score", ("nihss",)),
            PathwayStep(4, "decision", "Reperfusion window decision"),
        ],
        "resp_001": [
            PathwayStep(1, "triage", "Respiratory triage"),
            PathwayStep(2, "cbc", "CBC inflammatory screening", ("cbc",)),
            PathwayStep(3, "chest_xray", "Chest X-ray", ("chest_xray",)),
            PathwayStep(4, "decision", "Empiric antibiotic decision"),
        ],
        "abd_001": [
            PathwayStep(1, "triage", "Abdominal pain triage"),
            PathwayStep(2, "abdominal_ultrasound", "Abdominal ultrasound", ("abdominal_ultrasound",)),
            PathwayStep(3, "decision", "Surgical consult decision"),
        ],
        "uti_001": [
            PathwayStep(1, "triage", "Urinary symptom triage"),
            PathwayStep(2, "urinalysis", "Urinalysis", ("urinalysis",)),
            PathwayStep(3, "urine_culture", "Urine culture", ("urine_culture",)),
            PathwayStep(4, "decision", "Antibiotic optimization decision"),
        ],
    }
    return pathways.get(case_id, [PathwayStep(1, "triage", "General triage")])


def pathway_status(case_id: str, completed_tests: dict[str, str]) -> dict[str, object]:
    steps = pathway_for_case(case_id)
    completed: list[dict[str, object]] = []
    pending: list[dict[str, object]] = []

    for step in steps:
        is_done = True if not step.required_tests else all(t in completed_tests for t in step.required_tests)
        node = {
            "order": step.order,
            "code": step.code,
            "title": step.title,
            "required_tests": list(step.required_tests),
        }
        if is_done:
            completed.append(node)
        else:
            pending.append(node)

    progress = len(completed) / max(1, len(steps))
    return {
        "case_id": case_id,
        "total_steps": len(steps),
        "progress": round(progress, 3),
        "completed": completed,
        "pending": pending,
    }
