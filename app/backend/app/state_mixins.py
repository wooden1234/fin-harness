"""各领域 State TypedDict Mixin。

集中导入各组件文件夹下的 state.py，统一 export 供 `states.py` 组合。
本文件不从 `agents.states` 或 `agents.__init__` 导入，避免循环依赖。

各 mixin 归属：
  - state_mixins.py 直接定义：SupervisorState / GuardrailsState
  - finance_agent/state.py：      PlannerState / WorkerOutputState
  - financial_query_agent/state.py：FinancialQueryState
"""

from __future__ import annotations

from typing import Any, NotRequired
from typing_extensions import TypedDict

from app.shared import AgentRoute


# ─── 兼容路由状态（名称保留，生产主图由 Analyzer 写入）───
class SupervisorState(TypedDict):
    """保留旧类型名，当前仅承载 Analyzer 产生的最终路由。"""
    route: NotRequired[AgentRoute]


# ─── Guardrails（直属组件，无大 agent 包裹）───
class GuardrailsState(TypedDict):
    """安全护栏节点写入的校验结果"""
    guardrail_decision: NotRequired[dict[str, Any]]
    guardrails_pass: NotRequired[bool]
    guardrails_reason: NotRequired[str]


# ─── Compliance（最终输出审查）───
class ComplianceState(TypedDict):
    """最终答案合规审查结果。"""
    compliance_action: NotRequired[str]
    compliance_reason_code: NotRequired[str]
    compliance_reason: NotRequired[str]


# ─── Finance Agent 领域（两个 state 在组件目录下）───
from agents.finance_agent.state import (  # noqa: E402
    PlannerState,
    WorkerOutputState,
)
from agents.finance_agent.financial_query_agent.state import (  # noqa: E402
    FinancialQueryState,
)


__all__ = [
    "ComplianceState",
    "FinancialQueryState",
    "GuardrailsState",
    "PlannerState",
    "SupervisorState",
    "WorkerOutputState",
]
