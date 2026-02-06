from __future__ import annotations

from .models import PatientState, ToolAction, ToolKind


class TriageAgent:
    def assess(self, state: PatientState) -> str:
        symptoms = " ".join(state.symptoms)
        if "胸痛" in symptoms or "呼吸急促" in symptoms:
            return "high"
        if "言语不清" in symptoms or "肢体无力" in symptoms:
            return "high"
        if "发热" in symptoms and "咳嗽" in symptoms:
            return "medium"
        return "medium"


class DiagnosticAgent:
    def choose_action(self, state: PatientState, user_message: str) -> ToolAction:
        msg = user_message.lower()
        completed = state.completed_tests
        symptoms = " ".join(state.symptoms)

        if "建议" in user_message or "总结" in user_message or "diagnosis" in msg:
            return ToolAction(ToolKind.RECOMMEND_PLAN, payload={"diagnosis": self.infer_diagnosis(state)})

        if "胸痛" in symptoms:
            if "ecg" not in completed:
                return ToolAction(ToolKind.ORDER_TEST, payload={"test": "ecg"})
            if "troponin" not in completed:
                return ToolAction(ToolKind.ORDER_TEST, payload={"test": "troponin"})
            if not self._acs_evidence(state) and "chest_xray" not in completed:
                return ToolAction(ToolKind.ORDER_TEST, payload={"test": "chest_xray"})
            return ToolAction(ToolKind.RECOMMEND_PLAN, payload={"diagnosis": self.infer_diagnosis(state)})

        if "发热" in symptoms and "咳嗽" in symptoms:
            if "cbc" not in completed:
                return ToolAction(ToolKind.ORDER_TEST, payload={"test": "cbc"})
            if "chest_xray" not in completed:
                return ToolAction(ToolKind.ORDER_TEST, payload={"test": "chest_xray"})
            if not self._pneumonia_evidence(state) and "crp" not in completed:
                return ToolAction(ToolKind.ORDER_TEST, payload={"test": "crp"})
            return ToolAction(ToolKind.RECOMMEND_PLAN, payload={"diagnosis": self.infer_diagnosis(state)})

        if "右下腹痛" in symptoms:
            if "abdominal_ultrasound" not in completed:
                return ToolAction(ToolKind.ORDER_TEST, payload={"test": "abdominal_ultrasound"})
            return ToolAction(ToolKind.RECOMMEND_PLAN, payload={"diagnosis": self.infer_diagnosis(state)})

        if "尿痛" in symptoms or "尿频" in symptoms:
            if "urinalysis" not in completed:
                return ToolAction(ToolKind.ORDER_TEST, payload={"test": "urinalysis"})
            if "urine_culture" not in completed:
                return ToolAction(ToolKind.ORDER_TEST, payload={"test": "urine_culture"})
            return ToolAction(ToolKind.RECOMMEND_PLAN, payload={"diagnosis": self.infer_diagnosis(state)})

        if "言语不清" in symptoms or "肢体无力" in symptoms or "口角歪斜" in symptoms:
            if "head_ct" not in completed:
                return ToolAction(ToolKind.ORDER_TEST, payload={"test": "head_ct"})
            if "nihss" not in completed:
                return ToolAction(ToolKind.ORDER_TEST, payload={"test": "nihss"})
            return ToolAction(ToolKind.RECOMMEND_PLAN, payload={"diagnosis": self.infer_diagnosis(state)})

        return ToolAction(ToolKind.ASK_QUESTION, payload={"question": "onset"})

    def infer_diagnosis(self, state: PatientState) -> str:
        tests = state.completed_tests
        ecg = tests.get("ecg", "")
        troponin = tests.get("troponin", "")
        chest_xray = tests.get("chest_xray", "")
        us = tests.get("abdominal_ultrasound", "")
        urinalysis = tests.get("urinalysis", "")
        urine_culture = tests.get("urine_culture", "")
        head_ct = tests.get("head_ct", "")
        nihss = tests.get("nihss", "")

        if self._acs_evidence(state):
            return "急性下壁心肌梗死"
        if self._pneumonia_evidence(state):
            return "社区获得性肺炎"
        if "阑尾" in us:
            return "急性阑尾炎"
        if "亚硝酸盐" in urinalysis and ("大肠埃希菌" in urine_culture or "菌" in urine_culture):
            return "急性下尿路感染"
        if "未见明显出血" in head_ct and "nihss" in nihss.lower():
            return "急性缺血性脑卒中"
        return "诊断证据不足，建议继续检查"

    def _acs_evidence(self, state: PatientState) -> bool:
        tests = state.completed_tests
        ecg = tests.get("ecg", "").lower()
        troponin = tests.get("troponin", "")
        has_st_elevation = "st" in ecg and "抬高" in ecg
        has_troponin = "肌钙蛋白" in troponin and "升高" in troponin
        return has_st_elevation and has_troponin

    def _pneumonia_evidence(self, state: PatientState) -> bool:
        chest_xray = state.completed_tests.get("chest_xray", "")
        return "浸润" in chest_xray

    def treatment_plan(self, diagnosis: str) -> str:
        if "心肌梗死" in diagnosis:
            return "诊断倾向急性下壁心肌梗死：建议立即启动急诊胸痛流程，进行心电监护、抗血小板与再灌注评估。"
        if "肺炎" in diagnosis:
            return "建议经验性抗感染治疗并评估氧饱和度，必要时住院观察。"
        if "阑尾炎" in diagnosis:
            return "建议外科会诊，评估抗感染及手术时机。"
        if "下尿路感染" in diagnosis:
            return "建议经验性抗感染并根据尿培养结果调整治疗，注意补液。"
        if "脑卒中" in diagnosis:
            return "建议立即启动卒中绿色通道，评估再灌注时窗与神经监护。"
        return "建议补充关键检查后再决策。"


class SafetyAgent:
    def evaluate(self, state: PatientState, diagnosis: str, urgency: str) -> tuple[str, bool, list[str], bool]:
        red_flags: list[str] = []
        symptoms = " ".join(state.symptoms)

        if "胸痛" in symptoms and ("呼吸急促" in symptoms or "出汗" in symptoms):
            red_flags.append("胸痛伴呼吸急促/出汗")

        if "言语不清" in symptoms and ("肢体无力" in symptoms or "口角歪斜" in symptoms):
            red_flags.append("疑似急性卒中症状组合")

        ecg = state.completed_tests.get("ecg", "").lower()
        if "st" in ecg and "抬高" in ecg:
            red_flags.append("心电图ST段抬高")

        troponin = state.completed_tests.get("troponin", "")
        if "肌钙蛋白" in troponin and "升高" in troponin:
            red_flags.append("肌钙蛋白升高")

        emergency = len(red_flags) > 0
        dangerous_miss = False
        if "胸痛伴呼吸急促/出汗" in red_flags and ("心肌梗死" not in diagnosis):
            dangerous_miss = True
        if "疑似急性卒中症状组合" in red_flags and ("脑卒中" not in diagnosis):
            dangerous_miss = True

        if emergency:
            notice = "红旗规则触发：疑似急危重症，请立即急诊处置（本系统仅辅助）。"
        elif urgency == "high":
            notice = "高风险病例：本系统仅供辅助决策，请立即联系急诊医生。"
        else:
            notice = "本系统为辅助决策工具，最终诊疗请由持证医生确认。"

        return (notice, emergency, red_flags, dangerous_miss)
