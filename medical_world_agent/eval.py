from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import random

from .case_loader import load_key_tests
from .orchestrator import MedicalAgentSystem
from .world_model import MedicalWorldModel


@dataclass
class ReplayMetrics:
    episodes: int
    diagnosis_accuracy: float
    key_test_hit_rate: float
    over_testing_rate: float
    dangerous_miss_rate: float


@dataclass
class ReplayEpisode:
    episode_id: int
    case_id: str
    predicted_diagnosis: str
    true_diagnosis: str
    diagnosis_correct: int
    ordered_tests: list[str]
    key_test_hit_rate: float
    over_testing_rate: float
    dangerous_miss: int


@dataclass(frozen=True)
class QualityThresholds:
    min_diagnosis_accuracy: float = 0.8
    min_key_test_hit_rate: float = 0.8
    max_over_testing_rate: float = 0.25
    max_dangerous_miss_rate: float = 0.0


def run_replay_evaluation(
    episodes: int = 30,
    max_turns: int = 4,
    random_seed: int = 7,
    observation_noise: float = 0.15,
) -> tuple[ReplayMetrics, list[ReplayEpisode]]:
    if episodes <= 0:
        raise ValueError("episodes must be > 0")
    if max_turns < 2:
        raise ValueError("max_turns must be >= 2")

    rng = random.Random(random_seed)
    case_ids = MedicalWorldModel().list_case_ids()
    key_tests_map = load_key_tests()

    diagnosis_correct = 0
    key_test_hit_sum = 0.0
    over_testing_sum = 0.0
    dangerous_miss_count = 0
    episodes_detail: list[ReplayEpisode] = []

    for idx in range(episodes):
        case_id = rng.choice(case_ids)
        system = MedicalAgentSystem()
        session_id = system.start_session(
            case_id=case_id,
            random_seed=random_seed + idx,
            observation_noise=observation_noise,
        )

        turns = []
        for _ in range(max_turns - 1):
            turns.append(system.chat(session_id, "请继续"))
        final_turn = system.chat(session_id, "请给出建议")
        turns.append(final_turn)

        true_diag = system.true_diagnosis(session_id)
        if final_turn.diagnosis in true_diag or true_diag in final_turn.diagnosis:
            diagnosis_correct += 1

        ordered_tests = [
            t.tool_action.payload.get("test", "")
            for t in turns
            if t.tool_action.kind.value == "order_test"
        ]
        ordered_set = {str(x) for x in ordered_tests if x}
        key_tests = key_tests_map.get(case_id, set())

        key_hit = len(ordered_set.intersection(key_tests)) / len(key_tests) if key_tests else 0.0
        key_test_hit_sum += key_hit

        non_key = 0
        if ordered_set:
            non_key = len([t for t in ordered_set if t not in key_tests])
            over_testing_sum += non_key / len(ordered_set)
        else:
            over_testing_sum += 1.0

        if final_turn.dangerous_miss:
            dangerous_miss_count += 1

        episodes_detail.append(
            ReplayEpisode(
                episode_id=idx,
                case_id=case_id,
                predicted_diagnosis=final_turn.diagnosis,
                true_diagnosis=true_diag,
                diagnosis_correct=1 if (final_turn.diagnosis in true_diag or true_diag in final_turn.diagnosis) else 0,
                ordered_tests=sorted(ordered_set),
                key_test_hit_rate=key_hit,
                over_testing_rate=(non_key / len(ordered_set)) if ordered_set else 1.0,
                dangerous_miss=1 if final_turn.dangerous_miss else 0,
            )
        )

    metrics = ReplayMetrics(
        episodes=episodes,
        diagnosis_accuracy=diagnosis_correct / episodes,
        key_test_hit_rate=key_test_hit_sum / episodes,
        over_testing_rate=over_testing_sum / episodes,
        dangerous_miss_rate=dangerous_miss_count / episodes,
    )
    return (metrics, episodes_detail)


def save_replay_report(
    metrics: ReplayMetrics,
    episodes: list[ReplayEpisode],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "replay_metrics.json"
    csv_path = out_dir / "replay_episodes.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "episodes": metrics.episodes,
                "diagnosis_accuracy": metrics.diagnosis_accuracy,
                "key_test_hit_rate": metrics.key_test_hit_rate,
                "over_testing_rate": metrics.over_testing_rate,
                "dangerous_miss_rate": metrics.dangerous_miss_rate,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "episode_id",
                "case_id",
                "predicted_diagnosis",
                "true_diagnosis",
                "diagnosis_correct",
                "ordered_tests",
                "key_test_hit_rate",
                "over_testing_rate",
                "dangerous_miss",
            ]
        )
        for ep in episodes:
            writer.writerow(
                [
                    ep.episode_id,
                    ep.case_id,
                    ep.predicted_diagnosis,
                    ep.true_diagnosis,
                    ep.diagnosis_correct,
                    "|".join(ep.ordered_tests),
                    ep.key_test_hit_rate,
                    ep.over_testing_rate,
                    ep.dangerous_miss,
                ]
            )

    return (json_path, csv_path)


def quality_gate(metrics: ReplayMetrics, thresholds: QualityThresholds) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if metrics.diagnosis_accuracy < thresholds.min_diagnosis_accuracy:
        failures.append(
            f"diagnosis_accuracy={metrics.diagnosis_accuracy:.3f} < {thresholds.min_diagnosis_accuracy:.3f}"
        )
    if metrics.key_test_hit_rate < thresholds.min_key_test_hit_rate:
        failures.append(
            f"key_test_hit_rate={metrics.key_test_hit_rate:.3f} < {thresholds.min_key_test_hit_rate:.3f}"
        )
    if metrics.over_testing_rate > thresholds.max_over_testing_rate:
        failures.append(
            f"over_testing_rate={metrics.over_testing_rate:.3f} > {thresholds.max_over_testing_rate:.3f}"
        )
    if metrics.dangerous_miss_rate > thresholds.max_dangerous_miss_rate:
        failures.append(
            f"dangerous_miss_rate={metrics.dangerous_miss_rate:.3f} > {thresholds.max_dangerous_miss_rate:.3f}"
        )
    return (len(failures) == 0, failures)
