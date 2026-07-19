"""PDF 检索候选质量校准：只描述候选质量，不直接决定是否回答。"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .retriever import RetrievalHit


@dataclass
class CalibrationModel:
    score_type: str
    bias: float
    score_weight: float
    rank_weight: float
    overlap_weight: float
    samples: int = 0
    positives: int = 0

    def predict(self, score: float, rank: int, overlap: bool) -> float:
        value = (
            self.bias
            + self.score_weight * float(score)
            + self.rank_weight / max(int(rank), 1)
            + self.overlap_weight * (1.0 if overlap else 0.0)
        )
        value = max(min(value, 30.0), -30.0)
        return 1.0 / (1.0 + math.exp(-value))


class RetrievalQualityCalibrator:
    """用检索快照中的 chunk 命中标注拟合候选质量概率。"""

    VERSION = "pdf_retrieval_quality_v1"

    def __init__(self, models: dict[str, CalibrationModel] | None = None, *, source: str = "heuristic"):
        self.models = models or {}
        self.source = source

    @classmethod
    def from_result_files(cls, paths: Iterable[str | Path]) -> "RetrievalQualityCalibrator":
        rows: list[dict[str, Any]] = []
        for path in paths:
            rows.extend(_load_rows(Path(path)))
        return cls.fit(rows)

    @classmethod
    def fit(cls, rows: Iterable[dict[str, Any]]) -> "RetrievalQualityCalibrator":
        samples: dict[str, list[tuple[float, int, bool, int]]] = {}
        for row in rows:
            relevant = _relevant_keys(row)
            retrieved = row.get("retrieved") or []
            if not isinstance(retrieved, list) or not retrieved:
                continue
            for item in retrieved:
                if not isinstance(item, dict):
                    continue
                source, score = _item_score(item)
                rank = _as_int(item.get("rank"), 1)
                overlap = item.get("vector_rank") is not None and item.get("bm25_rank") is not None
                label = 1 if _retrieved_key(item) in relevant else 0
                samples.setdefault(source, []).append((score, rank, overlap, label))

        models = {
            source: _fit_logistic(source, values)
            for source, values in samples.items()
            if values
        }
        return cls(models, source="golden_set")

    @classmethod
    def load(cls, path: str | Path) -> "RetrievalQualityCalibrator":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        models = {
            name: CalibrationModel(**values)
            for name, values in (payload.get("models") or {}).items()
        }
        return cls(models, source=str(payload.get("source") or "golden_set"))

    def dump(self, path: str | Path) -> None:
        payload = {
            "version": self.VERSION,
            "source": self.source,
            "models": {name: asdict(model) for name, model in self.models.items()},
        }
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def predict_hit(self, hit: RetrievalHit, *, rank: int = 1) -> float:
        source, score = _hit_score(hit)
        model = self.models.get(source) or _heuristic_model(source)
        vector_rank = hit.metadata.get("vector_rank")
        bm25_rank = hit.metadata.get("bm25_rank")
        return model.predict(
            score,
            rank,
            vector_rank is not None and bm25_rank is not None,
        )

    def annotate(self, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        for rank, hit in enumerate(hits, 1):
            quality = self.predict_hit(hit, rank=rank)
            hit.metadata["retrieval_quality"] = quality
            hit.metadata["retrieval_quality_source"] = self.source
            hit.metadata["retrieval_quality_version"] = self.VERSION
        return hits


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return [row for row in payload["rows"] if isinstance(row, dict)]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


def _fit_logistic(source: str, values: list[tuple[float, int, bool, int]]) -> CalibrationModel:
    positives = sum(label for _, _, _, label in values)
    prior = (positives + 1.0) / (len(values) + 2.0)
    bias = math.log(prior / (1.0 - prior))
    weights = [bias, 1.0, 1.0, 0.5]
    learning_rate = 0.05
    for _ in range(800):
        gradient = [0.0, 0.0, 0.0, 0.0]
        for score, rank, overlap, label in values:
            features = [1.0, score, 1.0 / max(rank, 1), 1.0 if overlap else 0.0]
            prediction = _sigmoid(sum(weight * feature for weight, feature in zip(weights, features)))
            error = prediction - label
            for index, feature in enumerate(features):
                gradient[index] += error * feature
        scale = 1.0 / max(len(values), 1)
        for index in range(4):
            regularization = 0.01 * weights[index] if index else 0.0
            weights[index] -= learning_rate * (gradient[index] * scale + regularization)
    return CalibrationModel(
        score_type=source,
        bias=weights[0],
        score_weight=weights[1],
        rank_weight=weights[2],
        overlap_weight=weights[3],
        samples=len(values),
        positives=positives,
    )


def _heuristic_model(source: str) -> CalibrationModel:
    if source == "rerank":
        return CalibrationModel(source, -2.0, 5.0, 1.0, 0.5)
    if source == "rrf":
        return CalibrationModel(source, -2.0, 20.0, 2.0, 0.5)
    if source == "weighted":
        return CalibrationModel(source, -2.0, 4.0, 1.0, 0.5)
    return CalibrationModel(source, -2.0, 2.0, 1.0, 0.5)


def _item_score(item: dict[str, Any]) -> tuple[str, float]:
    for source, key in (
        ("rerank", "rerank_score"),
        ("rrf", "rrf_score"),
        ("vector", "vector_score"),
        ("bm25", "bm25_score"),
    ):
        value = _as_float(item.get(key))
        if value is not None:
            return source, value
    hybrid_score = _as_float(item.get("hybrid_score"))
    if hybrid_score is not None:
        # 评测快照的旧格式没有 score_type；RRF 的分数通常远低于加权融合。
        return ("rrf" if 0.0 <= hybrid_score <= 0.2 else "weighted"), hybrid_score
    return "unknown", _as_float(item.get("score")) or 0.0


def _hit_score(hit: RetrievalHit) -> tuple[str, float]:
    source = str(hit.score_type or hit.metadata.get("score_source") or "unknown")
    score = _as_float(hit.metadata.get(f"{source}_score"))
    if score is None:
        score = float(hit.score or 0.0)
    return source, score


def _relevant_keys(row: dict[str, Any]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for item in row.get("relevant_chunks") or []:
        if not isinstance(item, dict):
            continue
        doc_id = str(item.get("doc_id") or "")
        chunk_id = str(item.get("chunk_id") or item.get("node_id") or "")
        if doc_id and chunk_id:
            keys.add((doc_id, chunk_id.lstrip("0") or "0"))
    return keys


def _retrieved_key(item: dict[str, Any]) -> tuple[str, str]:
    doc_id = str(item.get("doc_id") or "")
    chunk_id = str(item.get("chunk_index") or "")
    if not chunk_id:
        node_id = str(item.get("node_id") or "")
        chunk_id = node_id.rsplit(":", 1)[-1].lstrip("0") or "0"
    return doc_id, chunk_id.lstrip("0") or "0"


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _sigmoid(value: float) -> float:
    value = max(min(value, 30.0), -30.0)
    return 1.0 / (1.0 + math.exp(-value))


__all__ = ["CalibrationModel", "RetrievalQualityCalibrator"]
