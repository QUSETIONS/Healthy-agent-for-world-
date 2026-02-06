from pathlib import Path

from medical_world_agent.eval import QualityThresholds, quality_gate, run_replay_evaluation, save_replay_report


def test_replay_metrics_shape_and_bounds() -> None:
    metrics, episodes = run_replay_evaluation(episodes=20, max_turns=4, random_seed=9, observation_noise=0.2)
    assert metrics.episodes == 20
    assert len(episodes) == 20
    assert 0.0 <= metrics.diagnosis_accuracy <= 1.0
    assert 0.0 <= metrics.key_test_hit_rate <= 1.0
    assert 0.0 <= metrics.over_testing_rate <= 1.0
    assert 0.0 <= metrics.dangerous_miss_rate <= 1.0


def test_replay_report_persistence(tmp_path: Path) -> None:
    metrics, episodes = run_replay_evaluation(episodes=20, max_turns=4, random_seed=11, observation_noise=0.2)
    json_path, csv_path = save_replay_report(metrics, episodes, tmp_path)
    assert json_path.exists()
    assert csv_path.exists()


def test_quality_gate_pass_and_fail() -> None:
    metrics, _ = run_replay_evaluation(episodes=20, max_turns=4, random_seed=11, observation_noise=0.2)

    ok, failures = quality_gate(metrics, QualityThresholds())
    assert ok is True
    assert failures == []

    strict = QualityThresholds(
        min_diagnosis_accuracy=1.1,
        min_key_test_hit_rate=1.1,
        max_over_testing_rate=-0.1,
        max_dangerous_miss_rate=-0.1,
    )
    ok2, failures2 = quality_gate(metrics, strict)
    assert ok2 is False
    assert failures2
