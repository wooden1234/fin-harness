"""PDF 查询元数据提取：优先使用大模型，失败时不施加字段过滤。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError, field_validator

from app.core.config import settings
from app.core.logger import get_logger
from retrieval.core.filters import (
    MetadataFilters,
    supported_filter_keys,
)
from retrieval.retrievers.query_constraints import (
    normalize_query_entity_filters,
    parse_query_constraints,
)
from retrieval.retrievers.json_utils import parse_json_payload

logger = get_logger(service="query_filter_extractor")

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


def _system_prompt() -> str:
    return """你是金融 PDF 知识库的查询元数据提取器。

只从用户问题中提取明确出现或明确指代的过滤字段，不要根据常识、公司列表或知识库内容猜测。
字段含义：
- year：用于过滤文档的报告/财务年度；annual_reports 中出现“2024年营业收入/2025年研发费用”等明确财务年份时必须提取
- year：不要提取“预计到2028年”“到2027年目标”“2026年市场规模预测”等正文预测/目标年份，除非问题明确询问该年份对应的报告版本
- year：不要提取“截至2024年底/截至2024年末”等正文事实时间边界；这不是文档报告年份
- ticker：六位股票代码
- doc_id：问题中明确出现的 PDF 文档 ID，格式通常为 PDF-...
- source：问题中明确指定的具体来源、报告名称或文件名；“报告/年报/研报/白皮书”等通用词不能作为 source
- issuer：问题中明确指定的发布机构、研究机构或发行方

无法确定时返回 null。confidence 为每个非空字段给出 0 到 1 的置信度。
字符串内部不要使用未转义的 ASCII 双引号，如需引用词语请使用中文引号“”。
不要输出 category；文档类型由另一个路由器处理。只输出 JSON。"""


def _human_prompt(query: str, categories: tuple[str, ...]) -> str:
    category_text = ", ".join(categories) if categories else "未确定"
    return (
        f"已确定的文档类别：{category_text}\n"
        f"用户问题：\n{query}\n\n"
        "请只提取适合作为文档元数据过滤条件的字段。"
    )


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

    def _get_llm(self) -> BaseChatModel:
        if self._llm is None:
            from agents.llm import get_router_llm

            self._llm = get_router_llm()
        return self._llm

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
    def _response_text(response: Any) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, dict):
            if isinstance(content.get("text"), str):
                return content["text"]
            return json.dumps(content, ensure_ascii=False)
        if isinstance(content, list):
            return "".join(
                block.get("text", str(block)) if isinstance(block, dict) else str(block)
                for block in content
            )
        return str(content)

    def _invoke(self, query: str, categories: tuple[str, ...]) -> QueryFilterDecision:
        llm = self._get_llm()
        messages = [
            SystemMessage(content=_system_prompt()),
            HumanMessage(content=_human_prompt(query, categories)),
        ]
        response: Any = None
        try:
            response = llm.invoke(messages)
            return self._parse_text(self._response_text(response))
        except Exception as exc:
            detail = ""
            if isinstance(exc, ValidationError):
                detail = "; ".join(
                    f"{'.'.join(str(item) for item in error.get('loc', ())) or 'root'}:{error.get('type', 'invalid')}"
                    for error in exc.errors()
                )
            else:
                detail = str(exc)
            logger.warning(
                "query_filter_extractor invoke or parse failed; use no metadata filters "
                "error={} detail={} response_preview={!r}",
                type(exc).__name__,
                detail or "unavailable",
                self._response_text(response)[:300] if response is not None else "",
            )
            raise

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

    def _validated_filters(
        self,
        decision: QueryFilterDecision,
        *,
        knowledge_bases: list[str],
        query: str,
    ) -> MetadataFilters:
        allowed = supported_filter_keys(knowledge_bases or None)
        result: MetadataFilters = {}
        for field in _FIELDS:
            value: Any = getattr(decision, field)
            if value in (None, "") or field not in allowed:
                continue
            if field == "year" and not self._allow_year_filter(query, knowledge_bases):
                continue
            confidence = float(decision.confidence.get(field, 0.0))
            if confidence < self.min_confidence:
                continue
            result[field] = value
        return result

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

        rule_plan = parse_query_constraints(
            key[0],
            knowledge_bases=list(categories),
            user_filters=user_filters,
        )
        rule_filters = dict(rule_plan.filters)

        # L1 已得到明确硬约束且没有字段语义歧义时，不调用 LLM。
        if rule_filters and not rule_plan.unresolved:
            result = QueryFilterExtraction(rule_filters, False, "rules")
            self._cache[key] = result
            return result

        try:
            decision = self._invoke(key[0], categories)
            filters = self._validated_filters(
                decision,
                knowledge_bases=list(categories),
                query=key[0],
            )
            filters = normalize_query_entity_filters(
                filters,
                knowledge_bases=list(categories),
            )
            filters = {**filters, **rule_filters}
            if filters:
                result = QueryFilterExtraction(filters, True, "rules_then_llm")
            else:
                result = QueryFilterExtraction(
                    {},
                    False,
                    "llm_empty_or_low_confidence; no metadata filters",
                )
        except Exception as exc:
            logger.warning(
                "query_filter_extractor failed; use no metadata filters error={}",
                type(exc).__name__,
            )
            result = QueryFilterExtraction({}, False, "llm_failed; no metadata filters")

        self._cache[key] = result
        return result


_default_extractor: QueryFilterExtractor | None = None


def get_query_filter_extractor() -> QueryFilterExtractor:
    global _default_extractor
    if _default_extractor is None:
        _default_extractor = QueryFilterExtractor()
    return _default_extractor
