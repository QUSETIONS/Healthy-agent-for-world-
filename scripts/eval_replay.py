from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from medical_world_agent.eval import run_replay_evaluation, save_replay_report


def main() -> None:
    metrics, episodes = run_replay_evaluation(episodes=30, max_turns=4, random_seed=7, observation_noise=0.15)
    json_path, csv_path = save_replay_report(metrics, episodes, ROOT / "reports")
    print(
        json.dumps(
            {
                "episodes": metrics.episodes,
                "diagnosis_accuracy": metrics.diagnosis_accuracy,
                "key_test_hit_rate": metrics.key_test_hit_rate,
                "over_testing_rate": metrics.over_testing_rate,
                "dangerous_miss_rate": metrics.dangerous_miss_rate,
                "report_json": str(json_path),
                "report_csv": str(csv_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
