# Healthy Agent for World

Local medical multi-agent system with an internal world model for multi-turn diagnosis, tool calling, treatment recommendation, and replay evaluation.

## Features

- Multi-agent orchestration: triage, diagnosis, and safety agents
- Internal clinical world model with `reset()` / `step()` interaction loop
- Tool-based reasoning workflow: `ask_question`, `order_test`, `recommend_plan`
- Hard red-flag safety override for emergency cases
- Guideline retrieval layer for diagnosis-linked evidence references
- External updatable knowledge source (`knowledge_corpus_path` / `knowledge_corpus_url`)
- Confidence-ranked guideline evidence references
- Refusal-and-handoff policy for high-risk or low-confidence scenarios
- Explainable outputs: diagnosis confidence + evidence chain
- Audit logging for session/decision traceability
- API permission control via `X-API-Key`
- Clinical pathway templates with per-session progress view
- Probabilistic observation noise with seed control and scoped overrides
- Replay evaluation with persisted reports (JSON + CSV)
- Regression quality gate for release checks

## Project Structure

- `medical_world_agent/world_model.py`: case states, transitions, noisy observation sampling
- `medical_world_agent/subagents.py`: triage, diagnostic, safety logic
- `medical_world_agent/tools.py`: tool registry and dispatch
- `medical_world_agent/orchestrator.py`: session lifecycle + end-to-end turn execution
- `medical_world_agent/knowledge.py`: local guideline corpus and retriever
- `medical_world_agent/api.py`: FastAPI endpoints
- `medical_world_agent/cli.py`: local terminal interaction
- `medical_world_agent/eval.py`: replay evaluation and report writer
- `static/dashboard.html`: minimal web operations dashboard
- `reports/audit_log.jsonl`: append-only audit trail
- `scripts/eval_replay.py`: runnable evaluation entrypoint
- `tests/`: unit/integration tests

## Quick Start

1) Install dependencies

```bash
python -m pip install -e .[dev]
```

2) Run CLI

```bash
python -m medical_world_agent.cli --case-id chest_pain_001
```

3) Run API

```bash
uvicorn medical_world_agent.api:app --host 0.0.0.0 --port 8000
```

Open dashboard:

```text
http://localhost:8000/
```

Optional access control:

```bash
set HEALTHY_AGENT_API_KEY=your-secret-key
```

Then pass request header:

```text
X-API-Key: your-secret-key
```

4) Run tests

```bash
python -m pytest
```

5) Run replay evaluation and generate reports

```bash
python scripts/eval_replay.py
```

6) Run quality gate

```bash
python scripts/quality_gate.py
```

Generated files:

- `reports/replay_metrics.json`
- `reports/replay_episodes.csv`

## API Overview

- `POST /sessions/start`
  - body: `case_id`, optional `random_seed`, `observation_noise`, `noise_profile`, `knowledge_corpus_path`, `knowledge_corpus_url`, `evidence_top_k`
- `POST /sessions/{session_id}/chat`
  - body: `message`
  - response includes `diagnosis_confidence`, `evidence_chain`, `escalate_to_human`, `refusal`, `refusal_reason`
- `GET /sessions`
- `GET /sessions/{session_id}/turns`
- `GET /sessions/{session_id}/pathway`
- `GET /sessions/{session_id}/state`

## Noise Configuration

`noise_profile` supports layered overrides:

- `default`
- `case` (per case)
- `test` (per test)
- `case_test` (per case + test)
- `qa` (per question)
- `case_qa` (per case + question)

Example:

```json
{
  "default": 0.1,
  "case": {"resp_001": 0.2},
  "case_test": {"resp_001": {"cbc": 1.0}}
}
```

## External Knowledge Source

Provide an external guideline corpus as JSON list with fields:

- `guideline_id`
- `title`
- `source`
- `tags` (string array)
- `content`

Session-level options:

- `knowledge_corpus_path`: local JSON file path
- `knowledge_corpus_url`: remote JSON endpoint
- `evidence_top_k`: number of guideline references returned

## Safety Notes

- This project is for research and engineering validation.
- It is not a medical device and does not replace licensed clinical judgment.
