from __future__ import annotations

from copy import deepcopy
import random
from typing import Any

from .models import PatientCase, PatientState, ToolAction, ToolKind, ToolResult


class MedicalWorldModel:
    """Rule-based clinical world model with gym-like reset/step behavior."""

    def __init__(
        self,
        random_seed: int | None = None,
        observation_noise: float = 0.15,
        noise_profile: dict[str, Any] | None = None,
    ) -> None:
        self._cases = self._build_cases()
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

    @staticmethod
    def key_tests(case_id: str) -> set[str]:
        mapping = {
            "chest_pain_001": {"ecg", "troponin"},
            "resp_001": {"cbc", "chest_xray"},
            "abd_001": {"abdominal_ultrasound"},
            "uti_001": {"urinalysis", "urine_culture"},
            "stroke_001": {"head_ct", "nihss"},
        }
        if case_id not in mapping:
            raise ValueError(f"Unknown case_id: {case_id}")
        return set(mapping[case_id])

    def _answer_question(self, question: str) -> tuple[str, bool]:
        assert self._case is not None
        if question in self._case.qa:
            return self._sample_variant_with_scope(
                self._case.qa[question],
                self._qa_variants().get(self._case.case_id, {}).get(question, []),
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
                self._qa_variants().get(self._case.case_id, {}).get(mapped, []),
                signal_type="qa",
                signal_name=mapped,
            )
        return ("患者暂时无法提供该信息。", False)

    def _sample_test_result(self, test_name: str) -> tuple[str, bool]:
        assert self._case is not None
        canonical = self._case.tests.get(test_name)
        if canonical is None:
            return ("未找到该检查，请选择可用检查项。", False)
        variants = self._test_variants().get(self._case.case_id, {}).get(test_name, [])
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

    @staticmethod
    def _qa_variants() -> dict[str, dict[str, list[str]]]:
        return {
            "chest_pain_001": {
                "onset": ["胸痛约2小时前突发，伴左臂放射痛。"],
                "risk_factor": ["有长期吸烟和高血压病史。"],
            },
            "resp_001": {
                "onset": ["咳嗽发热3天，夜间更明显。"],
            },
            "abd_001": {
                "onset": ["腹痛由脐周转移到右下腹，持续近一天。"],
            },
            "uti_001": {
                "onset": ["排尿疼痛伴尿频约2天，夜间明显。"],
            },
            "stroke_001": {
                "onset": ["约1小时前突发言语不清和右侧肢体无力。"],
            },
        }

    @staticmethod
    def _test_variants() -> dict[str, dict[str, list[str]]]:
        return {
            "chest_pain_001": {
                "ecg": ["心电图示II、III、aVF导联明显ST抬高，考虑急性缺血改变。"],
                "troponin": ["肌钙蛋白升高，提示心肌损伤证据。"],
            },
            "resp_001": {
                "cbc": ["血常规示白细胞增高并以中性粒细胞为主。"],
                "chest_xray": ["胸片见右下肺斑片状浸润，倾向感染。"],
            },
            "abd_001": {
                "abdominal_ultrasound": ["腹部超声示阑尾壁增厚并周围炎性渗出。"],
            },
            "uti_001": {
                "urinalysis": ["尿常规示白细胞和亚硝酸盐阳性。"],
                "urine_culture": ["尿培养检出大肠埃希菌，菌落计数显著。"],
            },
            "stroke_001": {
                "head_ct": ["头颅CT未见明显出血征象。"],
                "nihss": ["NIHSS评分8分，提示中度神经功能缺损。"],
            },
        }

    @staticmethod
    def _build_cases() -> dict[str, PatientCase]:
        return {
            "chest_pain_001": PatientCase(
                case_id="chest_pain_001",
                demographics={"age": 62, "sex": "male"},
                symptoms=["胸痛", "呼吸急促", "出汗"],
                qa={
                    "onset": "胸痛突发2小时，向左臂放射。",
                    "risk_factor": "吸烟30年，高血压病史。",
                    "allergy": "无明确药物过敏史。",
                },
                tests={
                    "ecg": "心电图提示II、III、aVF导联ST段抬高。",
                    "troponin": "肌钙蛋白显著升高。",
                    "chest_xray": "未见明显肺部感染灶。",
                },
                final_diagnosis="急性下壁心肌梗死",
            ),
            "resp_001": PatientCase(
                case_id="resp_001",
                demographics={"age": 41, "sex": "female"},
                symptoms=["发热", "咳嗽", "咳痰"],
                qa={
                    "onset": "发热咳嗽3天，夜间加重。",
                    "risk_factor": "近期受凉，无慢性肺病。",
                    "allergy": "青霉素过敏。",
                },
                tests={
                    "cbc": "白细胞12.8x10^9/L，中性粒细胞比例升高。",
                    "crp": "CRP升高。",
                    "chest_xray": "右下肺片状浸润影。",
                },
                final_diagnosis="社区获得性肺炎",
            ),
            "abd_001": PatientCase(
                case_id="abd_001",
                demographics={"age": 25, "sex": "male"},
                symptoms=["右下腹痛", "恶心", "低热"],
                qa={
                    "onset": "腹痛先脐周后转移至右下腹，约18小时。",
                    "risk_factor": "无特殊既往史。",
                    "allergy": "无。",
                },
                tests={
                    "cbc": "白细胞及中性粒细胞升高。",
                    "abdominal_ultrasound": "阑尾增粗，周围渗出。",
                    "urinalysis": "尿常规无感染证据。",
                },
                final_diagnosis="急性阑尾炎",
            ),
            "uti_001": PatientCase(
                case_id="uti_001",
                demographics={"age": 33, "sex": "female"},
                symptoms=["尿频", "尿痛", "低热"],
                qa={
                    "onset": "尿频尿痛2天，伴轻度发热。",
                    "risk_factor": "近期饮水较少，无已知肾病史。",
                    "allergy": "无明确药物过敏史。",
                },
                tests={
                    "urinalysis": "尿常规白细胞增多，亚硝酸盐阳性。",
                    "urine_culture": "尿培养提示大肠埃希菌生长。",
                    "cbc": "白细胞轻度升高。",
                },
                final_diagnosis="急性下尿路感染",
            ),
            "stroke_001": PatientCase(
                case_id="stroke_001",
                demographics={"age": 68, "sex": "female"},
                symptoms=["言语不清", "右侧肢体无力", "口角歪斜"],
                qa={
                    "onset": "症状突发约1小时，进行性加重。",
                    "risk_factor": "房颤和高血压病史。",
                    "allergy": "无。",
                },
                tests={
                    "head_ct": "头颅CT未见明显出血灶。",
                    "nihss": "NIHSS评分8分。",
                    "ecg": "心电图示房颤。",
                },
                final_diagnosis="急性缺血性脑卒中",
            ),
        }
