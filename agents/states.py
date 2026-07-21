"""LangGraph 共享状态：FinAgentState 由各领域 mixin 组合而成。

所有类型均通过本模块重导出，现有 `from agents.states import ...` 零改动。
实际定义分散在各组件文件夹的 state.py / models.py 中。
"""

from __future__ import annotations

from typing_extensions import TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from typing import Annotated

# ─── 共享类型 ───
from app.shared import (
    AgentRoute,
    Citation,
    CoreState,
    ConversationState,
    CoverageStatus,
    RiskLevel,
    SubTaskIntent,
    TaskResult,
)

# ─── 各领域 State Mixin ───
from app.state_mixins import (
    ComplianceState,
    FinancialQueryState,
    GuardrailsState,
    PlannerState,
    RiskTriageState,
    SupervisorState,
    WorkerOutputState,
)

# ─── 各领域 Pydantic 模型（重导出，保持向后兼容）───
from app.shared import PlannerOutput, Router, SubTask, SubTaskType


# ─── 组合：FinAgentState ───
class FinAgentState(
    CoreState,
    SupervisorState,
    RiskTriageState,
    GuardrailsState,
    ComplianceState,
    PlannerState,
    FinancialQueryState,
    WorkerOutputState,
):
    """主图全局状态。

    通过 TypedDict 多重继承，将各组件领域字段组合到一个类型中。
    各 mixin 独立定义在自己组件的 state.py 里，本文件只做组合 + 重导出。
    """
    pass


# ─── 入口更窄 ───
class FinAgentInput(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# ─── 重导出（所有类型仍可从 app.agents.states import）───
__all__ = [
    "AgentRoute",
    "Citation",
    "ComplianceState",
    "CoreState",
    "ConversationState",
    "CoverageStatus",
    "FinAgentInput",
    "FinAgentState",
    "FinancialQueryState",
    "GuardrailsState",
    "PlannerOutput",
    "PlannerState",
    "RiskLevel",
    "RiskTriageState",
    "Router",
    "SubTask",
    "SubTaskIntent",
    "SubTaskType",
    "SupervisorState",
    "TaskResult",
    "WorkerOutputState",
]
