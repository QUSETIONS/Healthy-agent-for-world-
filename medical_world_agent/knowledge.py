from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any
from urllib.request import urlopen


@dataclass(frozen=True)
class GuidelineSnippet:
    guideline_id: str
    title: str
    source: str
    tags: tuple[str, ...]
    content: str


@dataclass(frozen=True)
class GuidelineHit:
    snippet: GuidelineSnippet
    score: int
    confidence: float


class GuidelineRetriever:
    def __init__(
        self,
        corpus_path: str | None = None,
        corpus_url: str | None = None,
        timeout_sec: float = 5.0,
    ) -> None:
        docs = _default_corpus()

        if corpus_path:
            loaded = _load_corpus_from_path(corpus_path)
            if loaded:
                docs = loaded

        if corpus_url:
            loaded = _load_corpus_from_url(corpus_url, timeout_sec=timeout_sec)
            if loaded:
                docs = loaded

        self._docs = docs

    def retrieve(self, query: str, top_k: int = 3) -> list[GuidelineHit]:
        terms = _tokenize(query)
        if not terms:
            return []

        scored: list[tuple[int, GuidelineSnippet]] = []
        for doc in self._docs:
            bag = _tokenize(f"{doc.title} {' '.join(doc.tags)} {doc.content}")
            score = sum(1 for term in terms if term in bag)
            if score > 0:
                scored.append((score, doc))

        scored.sort(key=lambda item: item[0], reverse=True)
        selected = scored[: max(1, top_k)]
        if not selected:
            return []

        max_score = max(score for score, _ in selected)
        hits: list[GuidelineHit] = []
        for score, doc in selected:
            confidence = score / max(1, max_score)
            hits.append(GuidelineHit(snippet=doc, score=score, confidence=round(confidence, 3)))
        return hits


def _tokenize(text: str) -> set[str]:
    cleaned = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text.lower())
    return {x for x in cleaned.split() if x}


def _load_corpus_from_path(path: str) -> tuple[GuidelineSnippet, ...]:
    file_path = Path(path)
    if not file_path.exists():
        return ()
    raw = file_path.read_text(encoding="utf-8")
    payload = json.loads(raw)
    return _parse_corpus_payload(payload)


def _load_corpus_from_url(url: str, timeout_sec: float = 5.0) -> tuple[GuidelineSnippet, ...]:
    with urlopen(url, timeout=timeout_sec) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return _parse_corpus_payload(payload)


def _parse_corpus_payload(payload: Any) -> tuple[GuidelineSnippet, ...]:
    if not isinstance(payload, list):
        return ()

    docs: list[GuidelineSnippet] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        guideline_id = str(item.get("guideline_id", "")).strip()
        title = str(item.get("title", "")).strip()
        source = str(item.get("source", "")).strip()
        content = str(item.get("content", "")).strip()
        tags_raw = item.get("tags", [])
        tags = tuple(str(x).strip() for x in tags_raw) if isinstance(tags_raw, list) else tuple()
        if guideline_id and title and source and content:
            docs.append(
                GuidelineSnippet(
                    guideline_id=guideline_id,
                    title=title,
                    source=source,
                    tags=tags,
                    content=content,
                )
            )
    return tuple(docs)


def _default_corpus() -> tuple[GuidelineSnippet, ...]:
    return (
        GuidelineSnippet(
            guideline_id="acs-001",
            title="急性冠脉综合征急诊流程",
            source="AHA/ESC chest pain pathway (summary)",
            tags=("胸痛", "心肌梗死", "ecg", "troponin", "再灌注"),
            content="对疑似急性冠脉综合征患者应立即完成12导联心电图，动态复查肌钙蛋白，并在适应证下尽快评估再灌注治疗。",
        ),
        GuidelineSnippet(
            guideline_id="cap-001",
            title="社区获得性肺炎门急诊处理",
            source="ATS/IDSA CAP guideline (summary)",
            tags=("肺炎", "咳嗽", "发热", "胸片", "抗感染"),
            content="对于疑似社区获得性肺炎应结合胸部影像与炎症指标，尽早启动经验性抗感染，并评估住院指征。",
        ),
        GuidelineSnippet(
            guideline_id="app-001",
            title="急性阑尾炎评估与外科会诊",
            source="WSES appendicitis guideline (summary)",
            tags=("阑尾炎", "右下腹痛", "超声", "外科"),
            content="迁移性右下腹痛伴炎症指标升高时，应尽快进行影像评估并启动外科会诊。",
        ),
        GuidelineSnippet(
            guideline_id="uti-001",
            title="急性下尿路感染诊治原则",
            source="EAU UTI guideline (summary)",
            tags=("尿路感染", "尿频", "尿痛", "尿常规", "尿培养"),
            content="急性下尿路感染推荐结合尿常规与必要时尿培养，抗菌治疗需结合耐药风险与培养结果优化。",
        ),
        GuidelineSnippet(
            guideline_id="stroke-001",
            title="急性缺血性卒中绿色通道",
            source="AHA/ASA stroke guideline (summary)",
            tags=("卒中", "言语不清", "肢体无力", "头颅ct", "再灌注"),
            content="疑似急性卒中患者应立即完成神经功能评分与颅脑影像以排除出血，尽快评估再灌注时窗。",
        ),
    )
