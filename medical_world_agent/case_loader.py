from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import PatientCase


_DEFAULT_CASES_DIR = Path(__file__).resolve().parents[1] / "cases"


def _load_json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}") from exc


def _parse_case(raw: dict[str, Any]) -> PatientCase:
    required = ("case_id", "demographics", "symptoms", "qa", "tests", "final_diagnosis")
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(f"Case payload missing required fields: {', '.join(missing)}")

    demographics = raw["demographics"]
    symptoms = raw["symptoms"]
    qa = raw["qa"]
    tests = raw["tests"]
    if not isinstance(demographics, dict):
        raise ValueError("Case field 'demographics' must be an object")
    if not isinstance(symptoms, list):
        raise ValueError("Case field 'symptoms' must be an array")
    if not isinstance(qa, dict):
        raise ValueError("Case field 'qa' must be an object")
    if not isinstance(tests, dict):
        raise ValueError("Case field 'tests' must be an object")

    return PatientCase(
        case_id=str(raw["case_id"]),
        demographics={str(k): v for k, v in demographics.items()},
        symptoms=[str(x) for x in symptoms],
        qa={str(k): str(v) for k, v in qa.items()},
        tests={str(k): str(v) for k, v in tests.items()},
        final_diagnosis=str(raw["final_diagnosis"]),
    )


def load_cases(cases_dir: Path | None = None) -> dict[str, PatientCase]:
    directory = cases_dir or _DEFAULT_CASES_DIR
    if not directory.is_dir():
        raise FileNotFoundError(f"Cases directory not found: {directory}")

    cases: dict[str, PatientCase] = {}
    for path in sorted(directory.glob("*.json")):
        if path.name == "variants.json":
            continue
        raw = _load_json_file(path)
        if not isinstance(raw, dict):
            raise ValueError(f"Case file must contain a JSON object: {path}")
        case = _parse_case(raw)
        cases[case.case_id] = case

    if not cases:
        raise ValueError(f"No case JSON files found under: {directory}")
    return cases


def load_variants(
    cases_dir: Path | None = None,
) -> tuple[dict[str, dict[str, list[str]]], dict[str, dict[str, list[str]]]]:
    directory = cases_dir or _DEFAULT_CASES_DIR
    variants_path = directory / "variants.json"
    if not variants_path.exists():
        return ({}, {})

    raw = _load_json_file(variants_path)
    if not isinstance(raw, dict):
        raise ValueError(f"Variants file must contain a JSON object: {variants_path}")

    qa = raw.get("qa_variants", {})
    test = raw.get("test_variants", {})
    if not isinstance(qa, dict) or not isinstance(test, dict):
        raise ValueError("variants.json must contain object fields 'qa_variants' and 'test_variants'")

    return (qa, test)


def load_key_tests(cases_dir: Path | None = None) -> dict[str, set[str]]:
    directory = cases_dir or _DEFAULT_CASES_DIR
    mapping: dict[str, set[str]] = {}
    for path in sorted(directory.glob("*.json")):
        if path.name == "variants.json":
            continue
        raw = _load_json_file(path)
        if not isinstance(raw, dict):
            raise ValueError(f"Case file must contain a JSON object: {path}")
        case_id = str(raw.get("case_id", ""))
        kt = raw.get("key_tests", [])
        if not case_id:
            raise ValueError(f"Missing case_id in {path}")
        if not isinstance(kt, list):
            raise ValueError(f"Field 'key_tests' must be an array in {path}")
        mapping[case_id] = {str(t) for t in kt}

    if not mapping:
        raise ValueError(f"No key_tests mapping found under: {directory}")
    return mapping
