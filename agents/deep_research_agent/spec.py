"""深度研究 Deep Agent 的静态配置。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DeepResearchSpec:
    """限制深度研究可读取的 Skill、工具预算和输出边界。"""

    agent_id: str = "deep_research_agent"
    default_task_id: str = "deep-research"
    skills: tuple[str, ...] = (
        "stock-screening",
        "market-quotes",
        "industry-data",
        "index-data",
        "announcement-search",
        "research-report-search",
        "institution-rating",
    )
    max_tool_calls: int = 15
    recursion_limit: int = 30
    busy_answer: str = "深度研究暂时无法完成，请稍后重试。"
    role_prompt: str = (
        "你是受治理的金融深度研究 Agent。你的结论必须来自已绑定工具返回的证据，"
        "先制定简短研究步骤，再检索、交叉验证并报告证据缺口。"
    )


DEEP_RESEARCH_SPEC = DeepResearchSpec()


__all__ = ["DEEP_RESEARCH_SPEC", "DeepResearchSpec"]
