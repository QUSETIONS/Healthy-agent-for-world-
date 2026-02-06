from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from medical_world_agent.eval import QualityThresholds, quality_gate, run_replay_evaluation


def main() -> None:
    metrics, _ = run_replay_evaluation(episodes=30, max_turns=4, random_seed=13, observation_noise=0.15)
    thresholds = QualityThresholds()
    passed, failures = quality_gate(metrics, thresholds)

    payload = {
        "metrics": {
            "diagnosis_accuracy": metrics.diagnosis_accuracy,
            "key_test_hit_rate": metrics.key_test_hit_rate,
            "over_testing_rate": metrics.over_testing_rate,
            "dangerous_miss_rate": metrics.dangerous_miss_rate,
        },
        "thresholds": {
            "min_diagnosis_accuracy": thresholds.min_diagnosis_accuracy,
            "min_key_test_hit_rate": thresholds.min_key_test_hit_rate,
            "max_over_testing_rate": thresholds.max_over_testing_rate,
            "max_dangerous_miss_rate": thresholds.max_dangerous_miss_rate,
        },
        "passed": passed,
        "failures": failures,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
