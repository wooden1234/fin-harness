"""确定性研究资料检索工作流的静态边界。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ResearchRetrievalWorkflowSpec:
    """只允许访问已明确选择的公开研究文档来源。"""

    workflow_id: str = "research_retrieval_workflow"
    default_task_id: str = "research-search"
    tool_ids: tuple[str, ...] = (
        "iwencai.announcement.search",
        "iwencai.report.search",
        "iwencai.rating.query",
    )
    skill_by_tool: tuple[tuple[str, str], ...] = (
        ("iwencai.announcement.search", "announcement-search"),
        ("iwencai.report.search", "research-report-search"),
        ("iwencai.rating.query", "institution-rating"),
    )

    def skill_for_tool(self, tool_id: str) -> str | None:
        return dict(self.skill_by_tool).get(tool_id)


RESEARCH_RETRIEVAL_WORKFLOW_SPEC = ResearchRetrievalWorkflowSpec()


__all__ = [
    "RESEARCH_RETRIEVAL_WORKFLOW_SPEC",
    "ResearchRetrievalWorkflowSpec",
]
