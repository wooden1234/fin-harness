"""各领域 State TypedDict Mixin。

集中导入各组件文件夹下的 state.py，统一 export 供 `states.py` 组合。
本文件不从 `agents.states` 或 `agents.__init__` 导入，避免循环依赖。

各 mixin 归属：
  - state_mixins.py 直接定义：SupervisorState / RiskTriageState / GuardrailsState
  - finance_agent/state.py：      PlannerState / WorkerOutputState
  - financial_query_agent/state.py：FinancialQueryState
"""

from __future__ import annotations

from typing import NotRequired
from typing_extensions import TypedDict

from app.shared import AgentRoute, RiskLevel


# ─── Supervisor（直属组件，无大 agent 包裹）───
class SupervisorState(TypedDict):
    """Supervisor 写入的路由信息"""
    route: NotRequired[AgentRoute]
    logic: NotRequired[str]


# ─── Risk Triage（直属组件，无大 agent 包裹）───
class RiskTriageState(TypedDict):
    """风险分级节点写入的风险信息"""
    risk_level: NotRequired[RiskLevel]
    risk_reason: NotRequired[str]
    risk_needs_human: NotRequired[bool]


# ─── Guardrails（直属组件，无大 agent 包裹）───
class GuardrailsState(TypedDict):
    """安全护栏节点写入的校验结果"""
    guardrails_pass: NotRequired[bool]
    guardrails_reason: NotRequired[str]


# ─── Finance Agent 领域（两个 state 在组件目录下）───
from agents.finance_agent.state import (  # noqa: E402
    PlannerState,
    WorkerOutputState,
)
from agents.finance_agent.financial_query_agent.state import (  # noqa: E402
    FinancialQueryState,
)


__all__ = [
    "FinancialQueryState",
    "GuardrailsState",
    "PlannerState",
    "RiskTriageState",
    "SupervisorState",
    "WorkerOutputState",
]
