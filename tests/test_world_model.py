from medical_world_agent.models import ToolAction, ToolKind
from medical_world_agent.world_model import MedicalWorldModel


def test_world_model_reset_and_step() -> None:
    model = MedicalWorldModel(observation_noise=0.0)
    state = model.reset("chest_pain_001")
    assert state.case_id == "chest_pain_001"
    assert "胸痛" in state.symptoms

    qa_result = model.step(ToolAction(ToolKind.ASK_QUESTION, {"question": "onset"}))
    assert "胸痛突发" in qa_result.observation

    test_result = model.step(ToolAction(ToolKind.ORDER_TEST, {"test": "ecg"}))
    assert "ST段抬高" in test_result.observation


def test_world_model_noisy_observation_with_seed() -> None:
    model = MedicalWorldModel(random_seed=123, observation_noise=1.0)
    model.reset("resp_001", random_seed=123)
    test_result = model.step(ToolAction(ToolKind.ORDER_TEST, {"test": "cbc"}))
    assert test_result.metadata.get("noisy") is True
    assert "白细胞" in test_result.observation


def test_world_model_has_expanded_cases() -> None:
    model = MedicalWorldModel()
    case_ids = model.list_case_ids()
    assert "uti_001" in case_ids
    assert "stroke_001" in case_ids


def test_noise_profile_case_test_override() -> None:
    model = MedicalWorldModel(
        random_seed=99,
        observation_noise=0.0,
        noise_profile={
            "default": 0.0,
            "case_test": {"resp_001": {"cbc": 1.0}},
        },
    )
    model.reset("resp_001", random_seed=99)
    cbc = model.step(ToolAction(ToolKind.ORDER_TEST, {"test": "cbc"}))
    xray = model.step(ToolAction(ToolKind.ORDER_TEST, {"test": "chest_xray"}))
    assert cbc.metadata.get("noisy") is True
    assert xray.metadata.get("noisy") is False
