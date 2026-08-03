"""Main DeepAgent 动态预算和工具配额。"""

from __future__ import annotations

import asyncio
import time
from collections import Counter
from dataclasses import dataclass, field

from agents.main_deep_agent.entities import normalize_entity
from app.core.config import settings


TOOL_SOURCE_FAMILY = {
    "weather.get": "weather",
    "web.search": "web",
    "iwencai.query": "market",
    "iwencai.screen": "market",
    "iwencai.compare_entities": "market",
    "iwencai.market.query": "market",
    "iwencai.industry.query": "market",
    "iwencai.index.query": "market",
    "iwencai.rating.query": "research",
    "iwencai.announcement.search": "research",
    "iwencai.report.search": "research",
    "iwencai.fund.screen": "market",
    "knowledge.faq.search": "knowledge",
    "knowledge.pdf.catalog": "knowledge",
    "knowledge.pdf.search": "knowledge",
    "knowledge.fact.lookup": "financial",
    "calculation.run": "calculation",
}

FAMILY_LIMITS = {
    "web": 6,
    "market": 3,
    "research": 3,
    "knowledge": 2,
    "financial": 3,
    "weather": 1,
    "calculation": 3,
}


@dataclass(slots=True)
class MainAgentBudgetController:
    """根据实际工具行为升级预算，不承担语义路由。"""

    started_monotonic: float
    timeout_scope: asyncio.Timeout | None = None
    tool_calls: int = 0
    tool_counts: Counter[str] = field(default_factory=Counter)
    tool_id_counts: Counter[str] = field(default_factory=Counter)
    source_families: set[str] = field(default_factory=set)
    budget_tier: str = "direct"
    model_rounds: int = 0
    soft_seconds: float | None = None
    hard_seconds: float = 22.0
    stop_new_tools: bool = False
    web_entity_counts: Counter[str] = field(default_factory=Counter)
    finalization_reason: str = ""
    finalization_started_at: float | None = None

    def attach(self, timeout_scope: asyncio.Timeout) -> None:
        self.timeout_scope = timeout_scope
        self._reschedule()

    def _reschedule(self) -> None:
        if self.timeout_scope is not None:
            self.timeout_scope.reschedule(self.started_monotonic + self.hard_seconds)

    def elapsed(self) -> float:
        return max(0.0, time.monotonic() - self.started_monotonic)

    def soft_expired(self) -> bool:
        return self.soft_seconds is not None and self.elapsed() >= self.soft_seconds

    def remaining_hard_seconds(self) -> float:
        """返回当前动态硬截止的剩余秒数。"""
        deadline = (
            self.timeout_scope.when()
            if self.timeout_scope is not None
            else self.started_monotonic + self.hard_seconds
        )
        if deadline is None:
            return 0.0
        return max(0.0, deadline - time.monotonic())

    def quality_revision_reserve_seconds(self) -> float | None:
        """使用软硬截止之间的收尾窗口作为无工具回修预算。"""
        if self.soft_seconds is None:
            return None
        return max(0.0, self.hard_seconds - self.soft_seconds)

    def can_start_quality_revision(self) -> bool:
        """仅在完整收尾窗口仍可用时启动一次质量回修。"""
        reserve = self.quality_revision_reserve_seconds()
        return reserve is not None and self.remaining_hard_seconds() >= reserve

    @property
    def web_limit(self) -> int:
        """Web 配额只反映工具成本，不依赖问题语义或实体识别结果。"""
        return FAMILY_LIMITS["web"]

    def request_finalization(self, reason: str) -> None:
        """进入无工具成稿阶段，并保留首次触发原因和时间。"""
        self.stop_new_tools = True
        if not self.finalization_reason:
            self.finalization_reason = reason
            self.finalization_started_at = self.elapsed()

    def authorize(self, tool_id: str, *, entity: str = "") -> tuple[bool, str]:
        """检查软截止、总量、族配额和高级财务调用约束。"""
        if self.stop_new_tools:
            return False, self.finalization_reason or "finalization_started"
        if self.soft_expired():
            self.request_finalization("soft_deadline_no_new_tools")
            return False, "soft_deadline_no_new_tools"
        if (
            self.budget_tier == "deep_research"
            and self.elapsed() >= float(settings.MAIN_AGENT_RESEARCH_TOOL_CUTOFF_SEC)
        ):
            self.request_finalization("research_tool_cutoff")
            return False, "research_tool_cutoff"
        if self.tool_calls >= int(settings.MAIN_AGENT_MAX_TOOL_CALLS):
            self.request_finalization("tool_call_budget_exhausted")
            return False, "tool_call_budget_exhausted"
        family = TOOL_SOURCE_FAMILY.get(tool_id, "unknown")
        family_limit = self.web_limit if family == "web" else FAMILY_LIMITS.get(family, 1)
        if self.tool_counts[family] >= family_limit:
            reason = f"tool_family_budget_exhausted:{family}"
            return False, reason
        if tool_id == "knowledge.fact.lookup" and self.tool_id_counts[tool_id] >= 2:
            return False, "narrow_finance_call_exhausted"
        return True, ""

    def register(self, tool_id: str, *, entity: str = "") -> None:
        """登记一次已授权调用，并在首次、第二来源族时升级预算。"""
        family = TOOL_SOURCE_FAMILY.get(tool_id, "unknown")
        self.tool_calls += 1
        self.tool_counts[family] += 1
        self.tool_id_counts[tool_id] += 1
        if family == "web" and entity:
            self.web_entity_counts[normalize_entity(entity)] += 1
        if self.budget_tier == "direct":
            self.budget_tier = "standard"
            self.soft_seconds = float(settings.MAIN_AGENT_STANDARD_SOFT_DEADLINE_SEC)
            self.hard_seconds = float(settings.MAIN_AGENT_STANDARD_HARD_DEADLINE_SEC)
        if family not in {"calculation", "unknown"}:
            self.source_families.add(family)
        if len(self.independent_source_families) >= 2:
            self.budget_tier = "deep_research"
            self.soft_seconds = float(settings.MAIN_AGENT_RESEARCH_SOFT_DEADLINE_SEC)
            self.hard_seconds = float(settings.MAIN_AGENT_RESEARCH_HARD_DEADLINE_SEC)
        if family == "web" and sum(
            count > 0 for count in self.web_entity_counts.values()
        ) >= 2:
            self.budget_tier = "deep_research"
            self.soft_seconds = float(settings.MAIN_AGENT_RESEARCH_SOFT_DEADLINE_SEC)
            self.hard_seconds = float(settings.MAIN_AGENT_RESEARCH_HARD_DEADLINE_SEC)
        self._reschedule()

    @property
    def independent_source_families(self) -> set[str]:
        return set(self.source_families)


__all__ = ["MainAgentBudgetController", "TOOL_SOURCE_FAMILY"]
