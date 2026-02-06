from __future__ import annotations

from copy import deepcopy
import random
from typing import Any

from .case_loader import load_cases, load_key_tests, load_variants
from .models import PatientCase, PatientState, ToolAction, ToolKind, ToolResult


class MedicalWorldModel:
    """Rule-based clinical world model with gym-like reset/step behavior."""

    def __init__(
        self,
        random_seed: int | None = None,
        observation_noise: float = 0.15,
        noise_profile: dict[str, Any] | None = None,
    ) -> None:
        self._cases = load_cases()
        self._key_tests_map = load_key_tests()
        self._qa_variant_data, self._test_variant_data = load_variants()
        self._state: PatientState | None = None
        self._case: PatientCase | None = None
        self._rng = random.Random(random_seed)
        self._observation_noise = max(0.0, min(1.0, observation_noise))
        self._noise_profile = noise_profile or {}

    def reset(self, case_id: str, random_seed: int | None = None) -> PatientState:
        if case_id not in self._cases:
            raise ValueError(f"Unknown case_id: {case_id}")
        if random_seed is not None:
            self._rng.seed(random_seed)
        case = deepcopy(self._cases[case_id])
        self._case = case
        self._state = PatientState(
            case_id=case.case_id,
            demographics=case.demographics,
            symptoms=case.symptoms,
        )
        self._state.history.append(f"session_started:{case_id}")
        return deepcopy(self._state)

    def step(self, action: ToolAction) -> ToolResult:
        if self._state is None or self._case is None:
            raise RuntimeError("World model is not initialized. Call reset(case_id) first.")

        if action.kind == ToolKind.ASK_QUESTION:
            question = str(action.payload.get("question", "")).strip().lower()
            answer, noisy = self._answer_question(question)
            self._state.known_facts[question or "unknown_question"] = answer
            self._state.history.append(f"ask:{question}")
            return ToolResult(kind=action.kind, observation=answer, metadata={"noisy": noisy})

        if action.kind == ToolKind.ORDER_TEST:
            test_name = str(action.payload.get("test", "")).strip().lower()
            result, noisy = self._sample_test_result(test_name)
            if test_name in self._case.tests:
                self._state.completed_tests[test_name] = result
            self._state.history.append(f"test:{test_name}")
            return ToolResult(
                kind=action.kind,
                observation=result,
                metadata={"available_tests": sorted(self._case.tests.keys()), "noisy": noisy},
            )

        if action.kind == ToolKind.RECOMMEND_PLAN:
            proposed = str(action.payload.get("diagnosis", "")).strip().lower()
            target = self._case.final_diagnosis.lower()
            if proposed and proposed in target:
                observation = "诊断方向与世界模型标签一致，可进入处置建议阶段。"
            else:
                observation = "诊断证据不足或方向不一致，建议继续问诊/检查。"
            self._state.history.append("recommendation_evaluated")
            return ToolResult(kind=action.kind, observation=observation)

        raise ValueError(f"Unsupported tool kind: {action.kind}")

    def get_state(self) -> PatientState:
        if self._state is None:
            raise RuntimeError("World model is not initialized. Call reset(case_id) first.")
        return deepcopy(self._state)

    def true_diagnosis(self) -> str:
        if self._case is None:
            raise RuntimeError("World model is not initialized. Call reset(case_id) first.")
        return self._case.final_diagnosis

    def list_case_ids(self) -> list[str]:
        return sorted(self._cases.keys())

    def key_tests(self, case_id: str) -> set[str]:
        if case_id not in self._key_tests_map:
            raise ValueError(f"Unknown case_id: {case_id}")
        return set(self._key_tests_map[case_id])

    def _answer_question(self, question: str) -> tuple[str, bool]:
        assert self._case is not None
        if question in self._case.qa:
            return self._sample_variant_with_scope(
                self._case.qa[question],
                self._qa_variant_data.get(self._case.case_id, {}).get(question, []),
                signal_type="qa",
                signal_name=question,
            )

        alias_map = {
            "起病时间": "onset",
            "onset": "onset",
            "过敏史": "allergy",
            "allergy": "allergy",
            "危险因素": "risk_factor",
            "risk": "risk_factor",
        }
        mapped = alias_map.get(question)
        if mapped and mapped in self._case.qa:
            return self._sample_variant_with_scope(
                self._case.qa[mapped],
                self._qa_variant_data.get(self._case.case_id, {}).get(mapped, []),
                signal_type="qa",
                signal_name=mapped,
            )
        return ("患者暂时无法提供该信息。", False)

    def _sample_test_result(self, test_name: str) -> tuple[str, bool]:
        assert self._case is not None
        canonical = self._case.tests.get(test_name)
        if canonical is None:
            return ("未找到该检查，请选择可用检查项。", False)
        variants = self._test_variant_data.get(self._case.case_id, {}).get(test_name, [])
        return self._sample_variant_with_scope(
            canonical,
            variants,
            signal_type="test",
            signal_name=test_name,
        )

    def _sample_variant_with_scope(
        self,
        canonical: str,
        alternatives: list[str],
        *,
        signal_type: str,
        signal_name: str,
    ) -> tuple[str, bool]:
        if alternatives and self._rng.random() < self._effective_noise(signal_type=signal_type, signal_name=signal_name):
            return (self._rng.choice(alternatives), True)
        return (canonical, False)

    def _effective_noise(self, *, signal_type: str | None = None, signal_name: str | None = None) -> float:
        noise = self._observation_noise
        profile = self._noise_profile
        if "default" in profile:
            noise = float(profile["default"])

        if self._case is not None:
            case_overrides = profile.get("case", {})
            if self._case.case_id in case_overrides:
                noise = float(case_overrides[self._case.case_id])

            if signal_type == "test" and signal_name:
                test_overrides = profile.get("test", {})
                if signal_name in test_overrides:
                    noise = float(test_overrides[signal_name])
                case_test_overrides = profile.get("case_test", {})
                if self._case.case_id in case_test_overrides and signal_name in case_test_overrides[self._case.case_id]:
                    noise = float(case_test_overrides[self._case.case_id][signal_name])

            if signal_type == "qa" and signal_name:
                qa_overrides = profile.get("qa", {})
                if signal_name in qa_overrides:
                    noise = float(qa_overrides[signal_name])
                case_qa_overrides = profile.get("case_qa", {})
                if self._case.case_id in case_qa_overrides and signal_name in case_qa_overrides[self._case.case_id]:
                    noise = float(case_qa_overrides[self._case.case_id][signal_name])

        return max(0.0, min(1.0, noise))
