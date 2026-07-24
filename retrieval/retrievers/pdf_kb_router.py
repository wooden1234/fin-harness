"""PDF 知识库 LLM 路由器：只选文档库类型，可多选。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator

from app.core.config import settings
from app.core.logger import get_logger
from retrieval.core.filters import routable_kb_ids
from retrieval.retrievers.pdf_kb_router_prompts import (
    build_pdf_kb_route_human_prompt,
    build_pdf_kb_route_system_prompt,
)
from retrieval.retrievers.query_constraints import parse_query_constraints
from retrieval.retrievers.json_utils import parse_json_payload

logger = get_logger(service="pdf_kb_router")

_RESEARCH_CUE_RE = re.compile(
    r"研报|研究报告|券商|证券公司|机构观点|目标价|投资评级|估值方法|盈利预测"
)
_ANNUAL_REPORT_RULE_RE = re.compile(r"年报|年度报告|年度财务报告|财报|财务报告")
_RESEARCH_REPORT_RULE_RE = re.compile(
    r"研报|研究报告|券商报告|证券公司报告|机构观点|目标价|投资评级|估值方法|盈利预测"
)
_WHITEPAPER_RULE_RE = re.compile(r"白皮书|技术白皮书|产业白皮书")
_POLICY_RULE_RE = re.compile(
    r"政策文件|监管文件|行动方案|指导意见|管理办法|实施办法|"
    r"《[^》]{1,80}(?:通知|规划|条例|指引|办法|意见|方案)》|"
    r"(?:国务院|政府|证监会|发改委|工信部|人民银行|监管部门)"
    r".{0,16}(?:通知|规划|条例|指引|政策|办法|意见|方案)"
)
_MACRO_REPORT_RULE_RE = re.compile(
    r"货币政策执行报告|金融稳定报告|区域金融运行报告|"
    r"(?:中国人民银行|央行).{0,12}(?:报告|金融运行|货币政策)"
)
_ANNUAL_FACT_RULE_RE = re.compile(
    r"营业收入|营收|净利润|归母净利|利润|研发费用|研发投入|毛利率|"
    r"资产负债|总资产|净资产|现金流|应收账款|坏账准备|存货|商誉|"
    r"每股收益|分红|股利|关联交易|公司治理|风险因素|审计意见|"
    r"财务指标|财务数据|经营数据|附注|期末余额|账面余额|计提比例"
)
_RESEARCH_INSTITUTION_CONTEXT_RE = re.compile(
    r"观点|认为|预测|评级|估值|目标价|盈利预测|投资建议|研究结论"
)
_REALTIME_MARKER_RE = re.compile(
    r"当前|实时|今日|今天|现价|最新价|盘中|近\s*\d+\s*(?:日|天)|本周"
)
_REALTIME_DATA_RE = re.compile(
    r"股价|行情|涨跌幅?|资金流|成交量|成交额|换手率|盘口|K线|"
    r"MACD|RSI|技术指标|市盈率|市净率"
)
_DOC_ID_CATEGORY_PREFIXES: tuple[tuple[str, str], ...] = (
    ("PDF-MACRO-", "macro_research"),
    ("PDF-AR-", "annual_reports"),
    ("PDF-RR-", "research_reports"),
    ("PDF-WP-", "industry_whitepapers"),
    ("PDF-POL-", "policy"),
)


class PdfKbRouteDecision(BaseModel):
    supported: bool = Field(
        default=True,
        description="问题是否有合理的 PDF 文档证据基础；明确不支持时为 false",
    )
    categories: list[str] = Field(
        default_factory=list,
        description="相关知识库 ID 列表；无法可靠判断时返回空列表，由路由器统一回退全库",
    )
    uncertain: bool = Field(
        default=False,
        description="无法确定文档类型时为 true，此时应检索全部库",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="对本次路由结果的整体置信度，0~1",
    )
    reason: str = Field(
        default="",
        description="一句中文说明为何选择这些库",
    )

    @field_validator("categories", mode="before")
    @classmethod
    def _normalize_categories(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            return [value]
        if not isinstance(value, list):
            return []
        return [str(item) for item in value if item]

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, value: Any) -> float:
        try:
            return min(max(float(value), 0.0), 1.0)
        except (TypeError, ValueError):
            return 0.0

    @field_validator("uncertain", mode="before")
    @classmethod
    def _normalize_uncertain(cls, value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes"}
        return bool(value)

    @field_validator("supported", mode="before")
    @classmethod
    def _normalize_supported(cls, value: Any) -> bool:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"false", "0", "no", "否"}:
                return False
            if normalized in {"true", "1", "yes", "是"}:
                return True
        return bool(value)

    @field_validator("reason", mode="before")
    @classmethod
    def _normalize_reason(cls, value: Any) -> str:
        return str(value or "")


@dataclass(frozen=True)
class PdfKbRouteResult:
    supported: bool
    categories: tuple[str, ...]
    confidence: float
    reason: str
    uncertain: bool
    fallback_all: bool
    raw: PdfKbRouteDecision
    source: str = "llm"

    @property
    def category(self) -> str | None:
        return self.categories[0] if self.categories else None


class PdfKbRouter:
    def __init__(
        self,
        *,
        llm: BaseChatModel | None = None,
        min_confidence: float = 0.5,
        unsupported_min_confidence: float | None = None,
        system_prompt: str | None = None,
        fallback_to_all_when_uncertain: bool = True,
    ) -> None:
        self._llm = llm
        self.min_confidence = min_confidence
        self.unsupported_min_confidence = (
            settings.PDF_KB_UNSUPPORTED_MIN_CONFIDENCE
            if unsupported_min_confidence is None
            else unsupported_min_confidence
        )
        self._system_prompt_override = system_prompt
        self.fallback_to_all_when_uncertain = fallback_to_all_when_uncertain
        self._cache: dict[str, PdfKbRouteResult] = {}

    def _get_llm(self) -> BaseChatModel:
        if self._llm is None:
            from agents.llm import get_router_llm

            self._llm = get_router_llm()
        return self._llm

    def _resolve_system_prompt(self) -> str:
        if self._system_prompt_override is not None:
            return self._system_prompt_override
        return build_pdf_kb_route_system_prompt()

    def _normalize_categories(self, values: list[str] | None) -> list[str]:
        allowed = set(routable_kb_ids())
        ordered: list[str] = []
        seen: set[str] = set()
        for value in values or []:
            if value in allowed and value not in seen:
                ordered.append(value)
                seen.add(value)
        return ordered

    @staticmethod
    def _augment_categories(query: str, categories: list[str]) -> list[str]:
        """将高精度规则候选与模型候选合并，不覆盖模型判断。"""
        annual_plan = parse_query_constraints(
            query,
            knowledge_bases=["annual_reports"],
        )
        has_annual_identity = bool(
            annual_plan.filters.get("ticker") and annual_plan.filters.get("year")
        )
        if (
            has_annual_identity
            and not _RESEARCH_CUE_RE.search(query)
            and "annual_reports" not in categories
        ):
            return ["annual_reports", *categories]
        return categories

    def _build_messages(self, query: str) -> list[SystemMessage | HumanMessage]:
        return [
            SystemMessage(content=self._resolve_system_prompt()),
            HumanMessage(content=build_pdf_kb_route_human_prompt(query)),
        ]

    @staticmethod
    def _rule_decision(query: str) -> PdfKbRouteDecision | None:
        """只处理明确体裁线索；歧义问题继续交给 LLM。"""
        normalized = str(query or "").strip()
        if not normalized:
            return None

        identity_plan = parse_query_constraints(normalized)
        doc_id = str(identity_plan.filters.get("doc_id") or "").upper()
        for prefix, category in _DOC_ID_CATEGORY_PREFIXES:
            if doc_id.startswith(prefix):
                return PdfKbRouteDecision(
                    categories=[category],
                    uncertain=False,
                    confidence=1.0,
                    reason=f"规则路由：问题明确指向已登记文档 {doc_id}",
                )

        # 官方宏观报告即使包含“政策”字样，也应优先进入宏观报告库。
        if _MACRO_REPORT_RULE_RE.search(normalized):
            return PdfKbRouteDecision(
                categories=["macro_research"],
                uncertain=False,
                confidence=0.99,
                reason="规则路由：明确指向央行或官方宏观金融报告",
            )

        matches: list[str] = []
        reasons: list[str] = []
        for pattern, category, label in (
            (_ANNUAL_REPORT_RULE_RE, "annual_reports", "年报或财报"),
            (_RESEARCH_REPORT_RULE_RE, "research_reports", "研报或机构研究"),
            (_WHITEPAPER_RULE_RE, "industry_whitepapers", "白皮书"),
            (_POLICY_RULE_RE, "policy", "政策或监管文件"),
        ):
            if pattern.search(normalized):
                matches.append(category)
                reasons.append(label)
        if not matches:
            if _REALTIME_MARKER_RE.search(normalized) and _REALTIME_DATA_RE.search(normalized):
                return PdfKbRouteDecision(
                    supported=False,
                    categories=[],
                    uncertain=True,
                    confidence=0.99,
                    reason="规则路由：问题明确依赖实时行情或市场数据",
                )

            annual_plan = parse_query_constraints(
                normalized,
                knowledge_bases=["annual_reports"],
            )
            if (
                annual_plan.filters.get("ticker")
                and annual_plan.filters.get("year")
                and _ANNUAL_FACT_RULE_RE.search(normalized)
                and not _RESEARCH_CUE_RE.search(normalized)
            ):
                return PdfKbRouteDecision(
                    categories=["annual_reports"],
                    uncertain=False,
                    confidence=0.97,
                    reason="规则路由：公司实体、财年和财务披露指标均明确",
                )

            research_plan = parse_query_constraints(
                normalized,
                knowledge_bases=["research_reports"],
            )
            if (
                research_plan.filters.get("issuer")
                and _RESEARCH_INSTITUTION_CONTEXT_RE.search(normalized)
            ):
                return PdfKbRouteDecision(
                    categories=["research_reports"],
                    uncertain=False,
                    confidence=0.97,
                    reason="规则路由：研究机构与观点或预测语义均明确",
                )
            return None
        return PdfKbRouteDecision(
            categories=matches,
            uncertain=False,
            confidence=0.98,
            reason=f"规则路由：明确包含{'、'.join(reasons)}体裁线索",
        )

    def _parse_decision_text(self, text: str) -> PdfKbRouteDecision:
        raw = str(text or "").strip()
        if not raw:
            raise ValueError("empty llm response")
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
            raw = re.sub(r"\s*```$", "", raw)

        # 使用 JSONDecoder 逐个尝试对象起点，避免贪婪正则跨越多个对象或正文中的大括号。
        payload = parse_json_payload(raw)
        if not isinstance(payload, dict):
            raise ValueError("route payload must be a JSON object")
        return PdfKbRouteDecision.model_validate(payload)

    @staticmethod
    def _response_text(response: Any) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, dict):
            if isinstance(content.get("text"), str):
                return content["text"]
            return json.dumps(content, ensure_ascii=False)
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
                else:
                    parts.append(str(block))
            return "".join(parts)
        return str(content)

    async def _ainvoke_decision(self, query: str) -> PdfKbRouteDecision:
        llm = self._get_llm()
        messages = self._build_messages(query)
        response: Any = None
        try:
            response = await llm.ainvoke(messages)
            return self._parse_decision_text(self._response_text(response))
        except Exception as exc:
            response_text = self._response_text(response) if response is not None else ""
            logger.warning(
                "pdf_kb_router invoke failed; fallback to all categories "
                "error={} detail={} response_preview={!r}",
                type(exc).__name__,
                str(exc),
                response_text[:300],
            )
            return PdfKbRouteDecision(
                categories=[],
                uncertain=True,
                confidence=0.0,
                reason="llm_parse_failed",
            )

    def _invoke_decision(self, query: str) -> PdfKbRouteDecision:
        """保留同步兼容入口，异步图应使用 ``_ainvoke_decision``。"""
        llm = self._get_llm()
        messages = self._build_messages(query)
        response: Any = None
        try:
            response = llm.invoke(messages)
            return self._parse_decision_text(self._response_text(response))
        except Exception as exc:
            response_text = self._response_text(response) if response is not None else ""
            logger.warning(
                "pdf_kb_router invoke failed; fallback to all categories "
                "error={} detail={} response_preview={!r}",
                type(exc).__name__,
                str(exc),
                response_text[:300],
            )
            return PdfKbRouteDecision(
                categories=[],
                uncertain=True,
                confidence=0.0,
                reason="llm_parse_failed",
            )

    def route(self, query: str) -> PdfKbRouteResult:
        key = query.strip()
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        rule_decision = self._rule_decision(key)
        decision = rule_decision or self._invoke_decision(key)
        result = self._build_route_result(
            key,
            decision,
            source="rules" if rule_decision is not None else "llm",
        )
        self._cache[key] = result
        return result

    async def aroute(self, query: str) -> PdfKbRouteResult:
        """异步路由入口，避免在异步检索节点中执行同步 LLM 请求。"""
        key = query.strip()
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        rule_decision = self._rule_decision(key)
        decision = rule_decision or await self._ainvoke_decision(key)
        result = self._build_route_result(
            key,
            decision,
            source="rules" if rule_decision is not None else "llm",
        )
        self._cache[key] = result
        return result

    def _build_route_result(
        self,
        key: str,
        decision: PdfKbRouteDecision,
        *,
        source: str = "llm",
    ) -> PdfKbRouteResult:
        """根据模型结果构造路由结果，供同步和异步入口复用。"""
        if (
            not decision.supported
            and float(decision.confidence) >= self.unsupported_min_confidence
        ):
            return PdfKbRouteResult(
                supported=False,
                categories=(),
                confidence=float(decision.confidence),
                reason=str(decision.reason or "pdf_knowledge_base_unsupported"),
                uncertain=True,
                fallback_all=False,
                raw=decision,
                source=source,
            )
        if not decision.supported:
            decision = decision.model_copy(
                update={
                    "supported": True,
                    "uncertain": True,
                    "reason": f"低置信度拒答，继续召回：{decision.reason}",
                }
            )
        categories = self._normalize_categories(list(decision.categories or []))
        categories = self._augment_categories(key, categories)
        uncertain = bool(decision.uncertain)
        fallback_all = False
        all_ids = list(routable_kb_ids())
        if self.fallback_to_all_when_uncertain and (
            uncertain or not categories or float(decision.confidence) < self.min_confidence
        ):
            categories = all_ids
            fallback_all = True
            uncertain = True
        return PdfKbRouteResult(
            supported=True,
            categories=tuple(categories),
            confidence=float(decision.confidence),
            reason=str(decision.reason or ""),
            uncertain=uncertain,
            fallback_all=fallback_all,
            raw=decision,
            source=source,
        )

    async def aroute_categories(self, query: str) -> list[str]:
        return list((await self.aroute(query)).categories)

    def route_categories(self, query: str) -> list[str]:
        return list(self.route(query).categories)


_default_router: PdfKbRouter | None = None


def get_pdf_kb_router(*, min_confidence: float = 0.5) -> PdfKbRouter:
    global _default_router
    if _default_router is None or _default_router.min_confidence != min_confidence:
        _default_router = PdfKbRouter(min_confidence=min_confidence)
    return _default_router


def route_pdf_kb_categories(
    query: str,
    *,
    min_confidence: float = 0.5,
    router: PdfKbRouter | None = None,
) -> list[str]:
    active = router or get_pdf_kb_router(min_confidence=min_confidence)
    return active.route_categories(query)
