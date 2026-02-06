import json
from pathlib import Path

from medical_world_agent.knowledge import GuidelineRetriever


def test_guideline_retriever_returns_acs_guidance() -> None:
    retriever = GuidelineRetriever()
    hits = retriever.retrieve("胸痛 肌钙蛋白 ST段抬高 再灌注", top_k=2)
    assert hits
    assert any("冠脉" in h.snippet.title or "胸痛" in h.snippet.title for h in hits)
    assert hits[0].confidence >= hits[-1].confidence


def test_guideline_retriever_loads_external_corpus(tmp_path: Path) -> None:
    corpus_path = tmp_path / "guidelines.json"
    corpus = [
        {
            "guideline_id": "custom-001",
            "title": "自定义糖尿病门诊随访",
            "source": "Local KB",
            "tags": ["糖尿病", "血糖", "随访"],
            "content": "建议定期复查HbA1c并评估并发症风险。",
        }
    ]
    corpus_path.write_text(json.dumps(corpus, ensure_ascii=False), encoding="utf-8")

    retriever = GuidelineRetriever(corpus_path=str(corpus_path))
    hits = retriever.retrieve("糖尿病 血糖 HbA1c", top_k=1)
    assert hits
    assert hits[0].snippet.guideline_id == "custom-001"
