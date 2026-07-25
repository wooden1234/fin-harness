"""PDF 查询元数据提取：仅使用确定性规则和实体词典。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field, field_validator

from app.core.config import settings
from retrieval.core.filters import MetadataFilters
from retrieval.retrievers.query_constraints import parse_query_constraints
from retrieval.retrievers.json_utils import parse_json_payload

_FIELDS = ("year", "ticker", "doc_id", "source", "issuer")
_DOC_ID_PATTERN = re.compile(r"^PDF-[A-Z0-9-]+$", flags=re.IGNORECASE)
_TICKER_PATTERN = re.compile(r"^\d{6}$")
_GENERIC_SOURCE_PATTERN = re.compile(
    r"^(报告|年报|年度报告|研报|研究报告|白皮书|政策|政策文件)$"
)
_DOCUMENT_YEAR_PATTERN = re.compile(
    r"20\d{2}年?(?:第?[一二三四]季度)?(?:年报|年度报告|财报|研究报告|研报|白皮书|政策|规划|通知|办法|报告|版本|版)"
)
_YEAR_TARGET_PATTERN = re.compile(
    r"20\d{2}年?(?:收入|营收|利润|GMV|市场规模|资本开支|不良贷款率|占比|数量|价格|采购单价)"
)
_YEAR_PREDICTION_PATTERN = re.compile(
    r"(?:预计|预测|目标|达到|之后|以后).{0,12}20\d{2}年?(?:收入|营收|利润|GMV|市场规模|资本开支|占比|数量|价格|采购单价)"
)
_HISTORICAL_CUTOFF_YEAR_PATTERN = re.compile(r"截至\s*20\d{2}年?(?:底|年底|年末|年终)")


class QueryFilterDecision(BaseModel):
    """大模型从用户问题中提取的候选过滤字段。"""

    year: int | None = Field(default=None, ge=1900, le=2100)
    ticker: str | None = None
    doc_id: str | None = None
    source: str | None = None
    issuer: str | None = None
    confidence: dict[str, float] = Field(default_factory=dict)

    @field_validator("year", mode="before")
    @classmethod
    def _normalize_year(cls, value: Any) -> int | None:
        try:
            year = int(value)
        except (TypeError, ValueError):
            return None
        return year if 1900 <= year <= 2100 else None

    @field_validator("ticker", mode="before")
    @classmethod
    def _validate_ticker(cls, value: str | None) -> str | None:
        if value is None or not isinstance(value, str):
            return None
        value = value.strip()
        return value if _TICKER_PATTERN.fullmatch(value) else None

    @field_validator("doc_id", mode="before")
    @classmethod
    def _validate_doc_id(cls, value: str | None) -> str | None:
        if value is not None and isinstance(value, str):
            value = value.strip().upper()
            return value if _DOC_ID_PATTERN.fullmatch(value) else None
        return None

    @field_validator("source", "issuer", mode="before")
    @classmethod
    def _normalize_text(cls, value: str | None) -> str | None:
        if value is None or not isinstance(value, str):
            return None
        value = value.strip()
        return value if value and len(value) <= 128 else None

    @field_validator("source", mode="before")
    @classmethod
    def _reject_generic_source(cls, value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        value = value.strip()
        return value if value and not _GENERIC_SOURCE_PATTERN.fullmatch(value) else None

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, value: Any) -> dict[str, float]:
        if not isinstance(value, dict):
            return {}
        result: dict[str, float] = {}
        for key, raw_score in value.items():
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                continue
            if 0.0 <= score <= 1.0:
                result[str(key)] = score
        return result


@dataclass(frozen=True)
class QueryFilterExtraction:
    filters: MetadataFilters
    used_llm: bool
    reason: str


class QueryFilterExtractor:
    def __init__(
        self,
        *,
        llm: BaseChatModel | None = None,
        min_confidence: float | None = None,
    ) -> None:
        self._llm = llm
        self.min_confidence = (
            settings.PDF_QUERY_FILTER_MIN_CONFIDENCE
            if min_confidence is None
            else min_confidence
        )
        self._cache: dict[tuple[str, tuple[str, ...], tuple[tuple[str, Any], ...]], QueryFilterExtraction] = {}

    @staticmethod
    def _parse_text(text: str) -> QueryFilterDecision:
        raw = str(text or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
            raw = re.sub(r"\s*```$", "", raw)
        payload = parse_json_payload(raw)
        if isinstance(payload, list):
            if len(payload) == 1 and isinstance(payload[0], dict):
                payload = payload[0]
            else:
                payload = {
                    str(item.get("field") or item.get("name")): item.get("value")
                    for item in payload
                    if isinstance(item, dict) and (item.get("field") or item.get("name"))
                }
        if not isinstance(payload, dict):
            raise ValueError("query filter payload must be a JSON object")
        nested_filters = payload.get("filters")
        if isinstance(nested_filters, dict):
            payload = {**nested_filters, **{k: v for k, v in payload.items() if k != "filters"}}
        confidence = payload.setdefault("confidence", {})
        if not isinstance(confidence, dict):
            confidence = {}
            payload["confidence"] = confidence
        for field in _FIELDS:
            value = payload.get(field)
            if isinstance(value, dict):
                if "confidence" in value:
                    confidence[field] = value["confidence"]
                payload[field] = value.get("value")
        return QueryFilterDecision.model_validate(payload)

    @staticmethod
    def _allow_year_filter(query: str, knowledge_bases: list[str]) -> bool:
        """仅在年份像文档版本时下推，避免误把正文事实年份当成文档年份。"""
        if _YEAR_PREDICTION_PATTERN.search(query):
            return False
        if _DOCUMENT_YEAR_PATTERN.search(query):
            return True
        # “截至某年底”通常是正文事实的时间边界，不是文档版本年份。
        if _HISTORICAL_CUTOFF_YEAR_PATTERN.search(query):
            return False
        categories = set(knowledge_bases)
        if categories and categories <= {"annual_reports"}:
            return True
        if "research_reports" in categories and not _YEAR_TARGET_PATTERN.search(query):
            return True
        return False

    def extract_rules(
        self,
        query: str,
        *,
        knowledge_bases: list[str] | None = None,
        user_filters: MetadataFilters | None = None,
    ) -> QueryFilterExtraction:
        """仅执行确定性规则和实体词典，不调用大模型。"""
        categories = [str(value) for value in (knowledge_bases or []) if value]
        plan = parse_query_constraints(
            str(query or "").strip(),
            knowledge_bases=categories,
            user_filters=user_filters,
        )
        return QueryFilterExtraction(dict(plan.filters), False, "rules")

    def extract(
        self,
        query: str,
        *,
        knowledge_bases: list[str] | None = None,
        user_filters: MetadataFilters | None = None,
    ) -> QueryFilterExtraction:
        categories = tuple(str(value) for value in (knowledge_bases or []) if value)
        normalized_user_filters = tuple(
            sorted((str(field), repr(value)) for field, value in (user_filters or {}).items())
        )
        key = (query.strip(), categories, normalized_user_filters)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        result = self.extract_rules(
            key[0], knowledge_bases=list(categories), user_filters=user_filters
        )
        self._cache[key] = result
        return result

    async def aextract(
        self,
        query: str,
        *,
        knowledge_bases: list[str] | None = None,
        user_filters: MetadataFilters | None = None,
    ) -> QueryFilterExtraction:
        """异步提取入口，避免异步检索节点调用同步 LLM。"""
        categories = tuple(str(value) for value in (knowledge_bases or []) if value)
        normalized_user_filters = tuple(
            sorted((str(field), repr(value)) for field, value in (user_filters or {}).items())
        )
        key = (query.strip(), categories, normalized_user_filters)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        result = self.extract_rules(
            key[0], knowledge_bases=list(categories), user_filters=user_filters
        )
        self._cache[key] = result
        return result


_default_extractor: QueryFilterExtractor | None = None


def get_query_filter_extractor() -> QueryFilterExtractor:
    global _default_extractor
    if _default_extractor is None:
        _default_extractor = QueryFilterExtractor()
    return _default_extractor
