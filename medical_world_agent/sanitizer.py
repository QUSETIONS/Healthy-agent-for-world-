from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SanitizeStrategy(str, Enum):
    KEEP = "keep"
    REMOVE = "remove"
    AGE_BUCKET = "age_bucket"
    MASK = "mask"


@dataclass(frozen=True)
class FieldRule:
    field_name: str
    strategy: SanitizeStrategy
    bucket_size: int = 10


@dataclass(frozen=True)
class SanitizerConfig:
    rules: tuple[FieldRule, ...] = (
        FieldRule("age", SanitizeStrategy.AGE_BUCKET, bucket_size=10),
        FieldRule("sex", SanitizeStrategy.KEEP),
        FieldRule("name", SanitizeStrategy.REMOVE),
        FieldRule("id_number", SanitizeStrategy.REMOVE),
        FieldRule("phone", SanitizeStrategy.REMOVE),
        FieldRule("address", SanitizeStrategy.REMOVE),
    )
    enabled: bool = True

    def rule_for(self, field_name: str) -> FieldRule | None:
        for rule in self.rules:
            if rule.field_name == field_name:
                return rule
        return None


def _bucket_age(age: Any, bucket_size: int) -> str:
    try:
        numeric = int(age)
    except (TypeError, ValueError):
        return "unknown"
    if numeric < 0:
        return "unknown"
    if numeric >= 90:
        return "90+"
    lower = (numeric // bucket_size) * bucket_size
    upper = lower + bucket_size - 1
    return f"{lower}-{upper}"


def _mask_value(value: Any) -> str:
    text = str(value)
    if len(text) <= 1:
        return "*"
    return text[0] + "*" * (len(text) - 1)


def _apply_strategy(value: Any, rule: FieldRule) -> Any:
    if rule.strategy == SanitizeStrategy.KEEP:
        return value
    if rule.strategy == SanitizeStrategy.REMOVE:
        return None
    if rule.strategy == SanitizeStrategy.AGE_BUCKET:
        return _bucket_age(value, rule.bucket_size)
    if rule.strategy == SanitizeStrategy.MASK:
        return _mask_value(value)
    return value


def sanitize_demographics(
    demographics: dict[str, Any],
    config: SanitizerConfig | None = None,
) -> dict[str, Any]:
    if config is None:
        config = SanitizerConfig()
    if not config.enabled:
        return deepcopy(demographics)

    result: dict[str, Any] = {}
    for key, value in demographics.items():
        rule = config.rule_for(key)
        if rule is None:
            result[key] = deepcopy(value)
            continue
        sanitized = _apply_strategy(value, rule)
        if sanitized is not None:
            result[key] = sanitized
    return result


def sanitize_state_dict(
    state_dict: dict[str, Any],
    config: SanitizerConfig | None = None,
) -> dict[str, Any]:
    if config is None:
        config = SanitizerConfig()

    out = deepcopy(state_dict)
    if "demographics" in out and isinstance(out["demographics"], dict):
        out["demographics"] = sanitize_demographics(out["demographics"], config)
    return out


def sanitize_audit_payload(
    payload: dict[str, Any],
    config: SanitizerConfig | None = None,
) -> dict[str, Any]:
    if config is None:
        config = SanitizerConfig()

    out = deepcopy(payload)
    if "demographics" in out and isinstance(out["demographics"], dict):
        out["demographics"] = sanitize_demographics(out["demographics"], config)

    # Redact user_message to bounded length in audit
    if "user_message" in out and isinstance(out["user_message"], str):
        msg = out["user_message"]
        if len(msg) > 500:
            out["user_message"] = msg[:500] + "...[truncated]"
    return out
