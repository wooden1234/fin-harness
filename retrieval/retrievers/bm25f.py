"""可显式控制字段权重和长度归一化的 BM25F 检索器。"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from retrieval.clients.es_client import create_es_client, index_name
from retrieval.core.collections import get_collection_registry, get_table_name
from retrieval.core.filters import (
    MetadataFilters,
    filter_categories,
    filters_for_category,
    merge_filters,
)
from retrieval.retrievers.es_bm25 import (
    _filter_clauses,
    _filtered_categories,
    _hit_metadata,
    _text_query,
)

Tokenizer = Callable[[str], list[str]]


def default_tokenizer(text: str) -> list[str]:
    """使用与 ES 无关的 Unicode 词元化，便于离线测试和显式评分。"""
    return re.findall(r"\w+", str(text or "").lower(), flags=re.UNICODE)


def jieba_search_tokenizer(text: str) -> list[str]:
    """使用结巴搜索模式近似 IK 的细粒度中文切词。"""
    import jieba

    return [token.strip().lower() for token in jieba.lcut_for_search(str(text or "")) if token.strip()]


@dataclass(frozen=True)
class BM25FConfig:
    """BM25F 的可调参数；字段权重和 b 均按字段分别配置。"""

    field_weights: dict[str, float] = field(
        default_factory=lambda: {
            "title": 5.0,
            "section": 4.0,
            "source": 2.0,
            "leaf_text": 1.0,
        }
    )
    field_b: dict[str, float] = field(
        default_factory=lambda: {
            "title": 0.2,
            "section": 0.4,
            "source": 0.3,
            "leaf_text": 0.75,
        }
    )
    k1: float = 1.2


class BM25FScorer:
    """对一批候选文档执行严格的字段级 BM25F 计算。"""

    def __init__(self, config: BM25FConfig | None = None, tokenizer: Tokenizer | None = None):
        self.config = config or BM25FConfig()
        if self.config.k1 < 0:
            raise ValueError("k1 必须大于等于 0")
        if any(value < 0 for value in self.config.field_weights.values()):
            raise ValueError("字段权重不能为负数")
        if any(not 0 <= value <= 1 for value in self.config.field_b.values()):
            raise ValueError("字段 b 必须位于 0 到 1 之间")
        self.tokenizer = tokenizer or default_tokenizer

    def score_documents(
        self,
        query: str,
        documents: list[dict[str, Any]],
    ) -> list[float]:
        fields = tuple(self.config.field_weights)
        query_terms = set(self.tokenizer(query))
        tokenized = [
            {name: self.tokenizer(str(document.get(name, ""))) for name in fields}
            for document in documents
        ]
        lengths = {
            name: [len(item[name]) for item in tokenized]
            for name in fields
        }
        averages = {
            name: (sum(values) / len(values) if values else 0.0)
            for name, values in lengths.items()
        }
        document_count = len(documents)
        document_frequency = {
            term: sum(
                any(term in item[name] for name in fields)
                for item in tokenized
            )
            for term in query_terms
        }

        scores: list[float] = []
        for item, document_lengths in zip(tokenized, zip(*(lengths[name] for name in fields))):
            score = 0.0
            for term in query_terms:
                df = document_frequency[term]
                if not df:
                    continue
                idf = math.log(1.0 + (document_count - df + 0.5) / (df + 0.5))
                weighted_tf = 0.0
                for field_name, field_length in zip(fields, document_lengths):
                    term_frequency = item[field_name].count(term)
                    if not term_frequency:
                        continue
                    average_length = averages[field_name]
                    normalization = 1.0
                    if average_length > 0:
                        b = self.config.field_b.get(field_name, 0.75)
                        normalization = (1.0 - b) + b * field_length / average_length
                    weighted_tf += self.config.field_weights[field_name] * term_frequency / normalization
                if weighted_tf:
                    score += idf * ((self.config.k1 + 1.0) * weighted_tf) / (
                        self.config.k1 + weighted_tf
                    )
            scores.append(score)
        return scores


class ElasticsearchBM25FRetriever:
    """ES 候选召回后执行显式 BM25F 重排的检索器。"""

    def __init__(
        self,
        categories: list[str] | None = None,
        metadata_filters: MetadataFilters | None = None,
        *,
        candidate_multiplier: int = 10,
        config: BM25FConfig | None = None,
        tokenizer: Tokenizer | None = None,
        client: Any | None = None,
    ):
        registry = get_collection_registry()
        self.categories = list(registry) if categories is None else list(categories)
        unknown = [category for category in self.categories if category not in registry]
        if unknown:
            raise ValueError(f"未知 categories={unknown}，可选: {', '.join(sorted(registry))}")
        self.metadata_filters = metadata_filters or {}
        self.candidate_multiplier = max(int(candidate_multiplier), 1)
        self.scorer = BM25FScorer(config, tokenizer=tokenizer)
        self._client = client or create_es_client()

    def search(
        self,
        query: str,
        *,
        top_k: int,
        metadata_filters: MetadataFilters | None = None,
    ) -> list[Any]:
        from retrieval.retrievers.retriever import RetrievalHit

        filters = merge_filters(self.metadata_filters, metadata_filters)
        categories = _filtered_categories(self.categories, filters)
        if not categories:
            return []
        candidates: list[tuple[dict[str, Any], str, dict[str, Any]]] = []
        size = max(int(top_k), 1) * self.candidate_multiplier
        for category in categories:
            response = self._client.search(
                index=[index_name(category)],
                body={
                    "query": {"bool": {"must": [_text_query(query)], "filter": _filter_clauses(filters_for_category(filters, category))}},
                    "size": size,
                },
            )
            for raw_hit in response.get("hits", {}).get("hits", []):
                source = raw_hit.get("_source") or {}
                candidates.append((source, category, raw_hit))
        sources = [item[0] for item in candidates]
        scores = self.scorer.score_documents(query, sources)
        hits: list[Any] = []
        for (source, category, raw_hit), score in zip(candidates, scores):
            metadata = _hit_metadata(source)
            hit_category = str(source.get("category") or metadata.get("category") or category)
            collection = str(source.get("collection") or metadata.get("collection") or get_table_name(hit_category))
            metadata.update({"category": hit_category, "collection": collection, "bm25f_score": score, "es_index": raw_hit.get("_index")})
            hits.append(RetrievalHit(text=str(source.get("leaf_text") or source.get("text") or ""), score=score, metadata=metadata, node_id=str(source.get("node_id") or "") or None, category=hit_category or None, collection=collection or None, score_type="bm25f"))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]
