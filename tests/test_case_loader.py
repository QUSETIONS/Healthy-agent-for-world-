import json

import pytest

from medical_world_agent.case_loader import load_cases, load_key_tests, load_variants


def test_load_cases_and_key_tests_from_json(tmp_path) -> None:
    case_payload = {
        "case_id": "demo_001",
        "demographics": {"age": 30, "sex": "female"},
        "symptoms": ["发热"],
        "qa": {"onset": "2天"},
        "tests": {"cbc": "白细胞升高"},
        "final_diagnosis": "上呼吸道感染",
        "key_tests": ["cbc"],
    }
    (tmp_path / "demo_001.json").write_text(json.dumps(case_payload, ensure_ascii=False), encoding="utf-8")

    cases = load_cases(tmp_path)
    assert set(cases.keys()) == {"demo_001"}
    assert cases["demo_001"].final_diagnosis == "上呼吸道感染"

    key_tests_map = load_key_tests(tmp_path)
    assert key_tests_map["demo_001"] == {"cbc"}


def test_load_variants_from_variants_json(tmp_path) -> None:
    variants = {
        "qa_variants": {"demo_001": {"onset": ["3天"]}},
        "test_variants": {"demo_001": {"cbc": ["白细胞中度升高"]}},
    }
    (tmp_path / "variants.json").write_text(json.dumps(variants, ensure_ascii=False), encoding="utf-8")

    qa, test = load_variants(tmp_path)
    assert qa["demo_001"]["onset"] == ["3天"]
    assert test["demo_001"]["cbc"] == ["白细胞中度升高"]


def test_load_key_tests_raises_on_invalid_shape(tmp_path) -> None:
    payload = {
        "case_id": "bad_001",
        "demographics": {"age": 50},
        "symptoms": ["胸痛"],
        "qa": {"onset": "1小时"},
        "tests": {"ecg": "ST抬高"},
        "final_diagnosis": "急性冠脉综合征",
        "key_tests": "ecg",
    }
    (tmp_path / "bad_001.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="key_tests"):
        load_key_tests(tmp_path)


def test_load_cases_raises_on_invalid_json(tmp_path) -> None:
    (tmp_path / "bad.json").write_text("{invalid", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON"):
        load_cases(tmp_path)
