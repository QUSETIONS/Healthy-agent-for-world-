import json

from medical_world_agent.knowledge import GuidelineRetriever


def test_tfidf_prefers_more_relevant_document(tmp_path) -> None:
    corpus = [
        {
            "guideline_id": "doc-1",
            "title": "胸痛评估流程",
            "source": "Local",
            "tags": ["胸痛", "肌钙蛋白"],
            "content": "胸痛患者需结合肌钙蛋白和心电图进行再灌注评估。",
        },
        {
            "guideline_id": "doc-2",
            "title": "普通感冒处理",
            "source": "Local",
            "tags": ["咳嗽", "流涕"],
            "content": "多饮水，必要时对症治疗。",
        },
    ]
    corpus_path = tmp_path / "guidelines.json"
    corpus_path.write_text(json.dumps(corpus, ensure_ascii=False), encoding="utf-8")

    retriever = GuidelineRetriever(corpus_path=str(corpus_path))
    hits = retriever.retrieve("胸痛 肌钙蛋白 再灌注", top_k=2)
    assert hits
    assert hits[0].snippet.guideline_id == "doc-1"
    assert hits[0].confidence >= hits[-1].confidence


def test_tfidf_top_k_behavior_is_stable() -> None:
    retriever = GuidelineRetriever()
    hits_top_1 = retriever.retrieve("卒中 头颅ct 再灌注", top_k=1)
    hits_top_3 = retriever.retrieve("卒中 头颅ct 再灌注", top_k=3)

    assert len(hits_top_1) == 1
    assert 1 <= len(hits_top_3) <= 3
    assert hits_top_1[0].snippet.guideline_id == hits_top_3[0].snippet.guideline_id


def test_tfidf_external_corpus_compatibility(tmp_path) -> None:
    corpus = [
        {
            "guideline_id": "custom-001",
            "title": "自定义糖尿病门诊随访",
            "source": "Local KB",
            "tags": ["糖尿病", "血糖", "随访"],
            "content": "建议定期复查HbA1c并评估并发症风险。",
        }
    ]
    corpus_path = tmp_path / "guidelines.json"
    corpus_path.write_text(json.dumps(corpus, ensure_ascii=False), encoding="utf-8")

    retriever = GuidelineRetriever(corpus_path=str(corpus_path))
    hits = retriever.retrieve("糖尿病 血糖 HbA1c", top_k=1)
    assert hits
    assert hits[0].snippet.guideline_id == "custom-001"
