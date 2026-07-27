"""确定性市场数据采集工作流的静态边界。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MarketAcquisitionWorkflowSpec:
    """只允许访问已明确选择的结构化市场数据来源。"""

    workflow_id: str = "market_acquisition_workflow"
    default_task_id: str = "market-acquire"
    tool_ids: tuple[str, ...] = (
        "iwencai.market.query",
        "iwencai.industry.query",
        "iwencai.index.query",
        "iwencai.fund.screen",
    )
    skill_by_tool: tuple[tuple[str, str], ...] = (
        ("iwencai.market.query", "market-quotes"),
        ("iwencai.industry.query", "industry-data"),
        ("iwencai.index.query", "index-data"),
        ("iwencai.fund.screen", "fund-screening"),
    )

    def skill_for_tool(self, tool_id: str) -> str | None:
        return dict(self.skill_by_tool).get(tool_id)


MARKET_ACQUISITION_WORKFLOW_SPEC = MarketAcquisitionWorkflowSpec()


__all__ = [
    "MARKET_ACQUISITION_WORKFLOW_SPEC",
    "MarketAcquisitionWorkflowSpec",
]
