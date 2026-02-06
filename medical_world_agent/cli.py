from __future__ import annotations

import argparse

from .orchestrator import MedicalAgentSystem


def main() -> None:
    parser = argparse.ArgumentParser(description="Medical world-model agent CLI")
    parser.add_argument("--case-id", default="chest_pain_001", help="Case ID")
    args = parser.parse_args()

    system = MedicalAgentSystem()
    session_id = system.start_session(args.case_id)
    print(f"Session started: {session_id}")
    print("Type 'exit' to quit.")

    while True:
        user_input = input("You> ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        turn = system.chat(session_id, user_input)
        print(f"Agent>\n{turn.message}\n")


if __name__ == "__main__":
    main()
