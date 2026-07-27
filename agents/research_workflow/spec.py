"""独立金融研究工作流的静态配置。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ResearchWorkflowSpec:
    """定义研究工作流的稳定标识和默认数据范围。"""

    workflow_id: str = "research_workflow"
    default_task_id: str = "research"
    default_data_sources: tuple[str, ...] = (
        "market",
        "research",
        "finance_rag",
    )


RESEARCH_WORKFLOW_SPEC = ResearchWorkflowSpec()


__all__ = ["RESEARCH_WORKFLOW_SPEC", "ResearchWorkflowSpec"]
