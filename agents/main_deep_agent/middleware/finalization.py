"""Main DeepAgent 的无工具收尾中间件。"""

from __future__ import annotations

import re

from langchain.agents.middleware import AgentMiddleware
from langgraph.config import get_stream_writer

from agents.main_deep_agent.middleware.budget import MainAgentBudgetController
from agents.main_deep_agent.state import MainAgentProgressJournal

_CUMULATIVE_METRIC_RE = re.compile(r"累计")
_ANNUAL_METRIC_RE = re.compile(r"营业收入|营业额|收入|净利润|归母|净利")
_DATE_BRACKET_RE = re.compile(r"\[\d{8}\]")


class MainAgentFailureFinalizationMiddleware(AgentMiddleware):
    """进入收尾后仅移除业务工具，保留结构化响应工具。"""

    def __init__(
        self,
        budget: MainAgentBudgetController,
        *,
        journal: MainAgentProgressJournal | None = None,
        business_tool_names: set[str] | None = None,
        idle_wraps_with_evidence: int = 2,
        idle_wraps_after_tools: int = 3,
    ) -> None:
        self.budget = budget
        self.journal = journal
        self.business_tool_names = set(business_tool_names or set())
        self.idle_wraps_with_evidence = max(1, int(idle_wraps_with_evidence))
        self.idle_wraps_after_tools = max(self.idle_wraps_with_evidence, int(idle_wraps_after_tools))
        self._event_emitted = False
        self._wraps_since_tool = 0
        self._last_tool_calls = 0

    def _has_cumulative_finance_facts(self) -> bool:
        if self.journal is None:
            return False
        hits = 0
        for item in self.journal.evidence.values():
            for fact in list(item.metadata.get("facts") or []):
                if not isinstance(fact, dict):
                    continue
                metric = str(fact.get("metric") or "")
                if (
                    _CUMULATIVE_METRIC_RE.search(metric)
                    and _ANNUAL_METRIC_RE.search(metric)
                    and _DATE_BRACKET_RE.search(metric)
                ):
                    hits += 1
                    if hits >= 2:
                        return True
        return False

    def _maybe_force_finalization(self) -> None:
        """有财务累计证据却迟迟不调终稿时，强制进入无工具收尾，避免递归空转。"""
        if self.budget.stop_new_tools:
            return
        tool_calls = int(self.budget.tool_calls)
        if tool_calls < 1:
            self._wraps_since_tool = 0
            self._last_tool_calls = 0
            return
        if tool_calls > self._last_tool_calls:
            # 本轮模型调用发生在工具结果返回之后，计为第 1 次空闲包装。
            self._last_tool_calls = tool_calls
            self._wraps_since_tool = 1
        else:
            self._wraps_since_tool += 1
        if self._has_cumulative_finance_facts():
            if self._wraps_since_tool >= self.idle_wraps_with_evidence:
                self.budget.request_finalization("idle_with_finance_evidence")
            return
        if self._wraps_since_tool >= self.idle_wraps_after_tools:
            self.budget.request_finalization("idle_after_tools")

    async def awrap_model_call(self, request, handler):
        self._maybe_force_finalization()
        if self.budget.stop_new_tools:
            if not self._event_emitted:
                self._event_emitted = True
                try:
                    get_stream_writer()(
                        {
                            "kind": "main_finalization",
                            "status": "running",
                        }
                    )
                except RuntimeError:
                    pass
            tools = [
                tool
                for tool in list(getattr(request, "tools", []) or [])
                if getattr(tool, "name", "") not in self.business_tool_names
            ]
            current_system = str(getattr(request, "system_message", "") or "")
            return await handler(
                request.override(
                    tools=tools,
                    system_message=(
                        current_system
                        + "\n现在进入无工具收尾阶段。必须使用 MainAgentResponse 结构，"
                        "只基于已取得 Evidence 成稿；完整财年优先用指标名含「累计」且带"
                        "[YYYYMMDD] 的字段，禁止改用单季度或最新价凑表；"
                        "不得发起新工具调用或用模型记忆补事实。"
                    ),
                )
            )
        return await handler(request)


__all__ = ["MainAgentFailureFinalizationMiddleware"]
