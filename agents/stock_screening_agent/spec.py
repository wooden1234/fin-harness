"""选股 Agent 的静态配置。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StockScreeningSpec:
    """选股 Deep Agent 声明。"""

    agent_id: str = "stock_screening_agent"
    skills: tuple[str, ...] = ("stock-screening",)
    role_prompt: str = "你是一个只读的 A 股选股 Agent。"
    busy_answer: str = "问财选股暂时无法完成，请稍后重试。"
    evidence_source_type: str = "iwencai"
    evidence_provider: str = "hithink-astock-selector"
    evidence_title: str = "同花顺问财"
    default_task_id: str = "stock-screening"

STOCK_SCREENING_SPEC = StockScreeningSpec()


__all__ = [
    "STOCK_SCREENING_SPEC",
    "StockScreeningSpec",
]
