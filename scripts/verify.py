from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _run_step(args: list[str], label: str) -> None:
    print(f"[verify] {label}: {' '.join(args)}")
    completed = subprocess.run(args, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    _run_step([sys.executable, "-m", "compileall", "medical_world_agent", "tests"], "syntax")
    _run_step([sys.executable, "-m", "pytest"], "tests")
    _run_step([sys.executable, "scripts/stress_sessions.py"], "stress")
    print("[verify] all checks passed")


if __name__ == "__main__":
    main()
