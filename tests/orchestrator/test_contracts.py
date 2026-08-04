"""统一数据契约测试。"""

import pytest
from pydantic import ValidationError

from agents.orchestrator.contracts import (
    AgentResult,
    Evidence,
    ExecutionDecision,
    RequestProfile,
    TaskPlan,
    TaskSpec,
)


def _task(task_id: str, *, depends_on: list[str] | None = None) -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        objective=f"执行任务 {task_id}",
        agent_id="test_agent",
        depends_on=depends_on or [],
    )


def test_request_profile_defaults_are_stable() -> None:
    profile = RequestProfile(
        original_query="筛选新能源股票",
        execution=ExecutionDecision(mode="deep_research", budget_tier="research"),
    )

    assert profile.domain == "finance"
    assert profile.execution.budget_tier == "research"
    assert profile.constraints == {}


def test_task_plan_accepts_dependency_dag() -> None:
    plan = TaskPlan(
        plan_id="plan-1",
        query="研究公司",
        tasks=[_task("screen"), _task("research", depends_on=["screen"])],
    )

    assert [task.task_id for task in plan.tasks] == ["screen", "research"]


@pytest.mark.parametrize(
    ("tasks", "error"),
    [
        ([_task("a"), _task("a")], "task_id_must_be_unique"),
        ([_task("a", depends_on=["missing"])], "unknown_task_dependency"),
        ([_task("a", depends_on=["a"])], "task_cannot_depend_on_itself"),
        (
            [_task("a", depends_on=["b"]), _task("b", depends_on=["a"])],
            "cyclic_task_dependency",
        ),
    ],
)
def test_task_plan_rejects_invalid_dependencies(
    tasks: list[TaskSpec],
    error: str,
) -> None:
    with pytest.raises(ValidationError, match=error):
        TaskPlan(plan_id="plan-1", query="测试", tasks=tasks)


def test_evidence_validates_confidence_range() -> None:
    with pytest.raises(ValidationError):
        Evidence(evidence_id="e-1", source_type="web", confidence=1.1)


def test_agent_result_keeps_structured_data_and_suggested_tasks() -> None:
    result = AgentResult(
        task_id="finance",
        agent_id="finance_agent",
        status="completed",
        structured_data={"count": 3},
        suggested_tasks=[_task("research", depends_on=["screen"])],
    )

    assert result.structured_data["count"] == 3
    assert result.suggested_tasks[0].task_id == "research"
