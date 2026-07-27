"""选股 Agent 的静态配置。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StockScreeningSpec:
    """选股工作流声明。"""

    agent_id: str = "stock_screening_agent"
    skills: tuple[str, ...] = ("stock-screening",)
    busy_answer: str = "问财选股暂时无法完成，请稍后重试。"
    default_task_id: str = "stock-screening"
    max_retries: int = 1

STOCK_SCREENING_SPEC = StockScreeningSpec()


__all__ = [
    "STOCK_SCREENING_SPEC",
    "StockScreeningSpec",
]
