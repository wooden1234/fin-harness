"""PDF 知识库 LLM 路由器：只选文档库类型，可多选。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

from app.core.logger import get_logger
from retrieval.core.filters import routable_kb_ids
from retrieval.retrievers.pdf_kb_router_prompts import (
    build_pdf_kb_route_human_prompt,
    build_pdf_kb_route_system_prompt,
)

logger = get_logger(service="pdf_kb_router")


class PdfKbRouteDecision(BaseModel):
    categories: list[str] = Field(
        default_factory=list,
        description="相关知识库 ID 列表（可多个）；无法判断时应返回全部可选库",
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


@dataclass(frozen=True)
class PdfKbRouteResult:
    categories: tuple[str, ...]
    confidence: float
    reason: str
    uncertain: bool
    fallback_all: bool
    raw: PdfKbRouteDecision

    @property
    def category(self) -> str | None:
        return self.categories[0] if self.categories else None


class PdfKbRouter:
    def __init__(
        self,
        *,
        llm: BaseChatModel | None = None,
        min_confidence: float = 0.5,
        system_prompt: str | None = None,
        fallback_to_all_when_uncertain: bool = True,
    ) -> None:
        self._llm = llm
        self.min_confidence = min_confidence
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

    def _build_messages(self, query: str) -> list[SystemMessage | HumanMessage]:
        return [
            SystemMessage(content=self._resolve_system_prompt()),
            HumanMessage(content=build_pdf_kb_route_human_prompt(query)),
        ]

    def _parse_decision_text(self, text: str) -> PdfKbRouteDecision:
        raw = str(text or "").strip()
        if not raw:
            raise ValueError("empty llm response")
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
            raw = re.sub(r"\s*```$", "", raw)
        try:
            return PdfKbRouteDecision.model_validate_json(raw)
        except ValidationError:
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if match is None:
                raise
            return PdfKbRouteDecision.model_validate_json(match.group(0))

    def _invoke_decision(self, query: str) -> PdfKbRouteDecision:
        llm = self._get_llm()
        messages = self._build_messages(query)
        structured = llm.with_structured_output(PdfKbRouteDecision, method="json_mode")

        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                return cast(PdfKbRouteDecision, structured.invoke(messages))
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "pdf_kb_router structured invoke failed attempt={} error={}",
                    attempt + 1,
                    type(exc).__name__,
                )

        try:
            response = llm.invoke(messages)
            content = getattr(response, "content", response)
            if isinstance(content, list):
                content = "".join(
                    block.get("text", str(block)) if isinstance(block, dict) else str(block)
                    for block in content
                )
            return self._parse_decision_text(str(content))
        except Exception as exc:
            logger.warning(
                "pdf_kb_router plain invoke failed error={} prior={}",
                type(exc).__name__,
                type(last_exc).__name__ if last_exc else None,
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

        decision = self._invoke_decision(key)

        categories = self._normalize_categories(list(decision.categories or []))
        uncertain = bool(decision.uncertain)
        fallback_all = False
        all_ids = list(routable_kb_ids())

        if self.fallback_to_all_when_uncertain and (
            uncertain
            or not categories
            or float(decision.confidence) < self.min_confidence
        ):
            categories = all_ids
            fallback_all = True
            uncertain = True

        result = PdfKbRouteResult(
            categories=tuple(categories),
            confidence=float(decision.confidence),
            reason=str(decision.reason or ""),
            uncertain=uncertain,
            fallback_all=fallback_all,
            raw=decision,
        )
        self._cache[key] = result
        return result

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
