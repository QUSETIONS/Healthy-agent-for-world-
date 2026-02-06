from medical_world_agent.sanitizer import (
    FieldRule,
    SanitizeStrategy,
    SanitizerConfig,
    sanitize_audit_payload,
    sanitize_demographics,
    sanitize_state_dict,
)


def test_age_bucketing_standard_ages() -> None:
    demographics = {"age": 62, "sex": "male"}
    result = sanitize_demographics(demographics)
    assert result["age"] == "60-69"
    assert result["sex"] == "male"


def test_age_bucketing_boundary_values() -> None:
    assert sanitize_demographics({"age": 0})["age"] == "0-9"
    assert sanitize_demographics({"age": 9})["age"] == "0-9"
    assert sanitize_demographics({"age": 10})["age"] == "10-19"
    assert sanitize_demographics({"age": 89})["age"] == "80-89"
    assert sanitize_demographics({"age": 90})["age"] == "90+"
    assert sanitize_demographics({"age": 105})["age"] == "90+"


def test_age_bucketing_invalid_values() -> None:
    assert sanitize_demographics({"age": "not_a_number"})["age"] == "unknown"
    assert sanitize_demographics({"age": -5})["age"] == "unknown"
    assert sanitize_demographics({"age": None})["age"] == "unknown"


def test_remove_strategy_strips_field() -> None:
    demographics = {"age": 30, "name": "张三", "phone": "13800138000"}
    result = sanitize_demographics(demographics)
    assert "name" not in result
    assert "phone" not in result
    assert result["age"] == "30-39"


def test_disabled_config_returns_original() -> None:
    config = SanitizerConfig(enabled=False)
    demographics = {"age": 62, "name": "张三"}
    result = sanitize_demographics(demographics, config)
    assert result["age"] == 62
    assert result["name"] == "张三"


def test_mask_strategy() -> None:
    config = SanitizerConfig(
        rules=(FieldRule("name", SanitizeStrategy.MASK),),
    )
    result = sanitize_demographics({"name": "张三丰"}, config)
    assert result["name"] == "张**"


def test_unknown_fields_kept_by_default() -> None:
    result = sanitize_demographics({"age": 25, "blood_type": "A+"})
    assert result["blood_type"] == "A+"
    assert result["age"] == "20-29"


def test_sanitize_state_dict_targets_demographics_only() -> None:
    state = {
        "case_id": "chest_pain_001",
        "demographics": {"age": 62, "sex": "male"},
        "symptoms": ["胸痛"],
        "known_facts": {"onset": "突发2小时"},
        "completed_tests": {"ecg": "ST段抬高"},
    }
    result = sanitize_state_dict(state)
    assert result["demographics"]["age"] == "60-69"
    assert result["symptoms"] == ["胸痛"]
    assert result["known_facts"]["onset"] == "突发2小时"
    assert result["completed_tests"]["ecg"] == "ST段抬高"


def test_sanitize_audit_payload_truncates_long_message() -> None:
    payload = {"user_message": "x" * 600, "session_id": "abc"}
    result = sanitize_audit_payload(payload)
    assert len(result["user_message"]) < 600
    assert result["user_message"].endswith("...[truncated]")
    assert result["session_id"] == "abc"


def test_sanitize_audit_payload_short_message_unchanged() -> None:
    payload = {"user_message": "请继续", "session_id": "abc"}
    result = sanitize_audit_payload(payload)
    assert result["user_message"] == "请继续"


def test_input_not_mutated() -> None:
    original = {"age": 62, "sex": "male"}
    sanitize_demographics(original)
    assert original["age"] == 62
    assert original["sex"] == "male"
