"""Agent 运行时租户上下文。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import time
from typing import Literal


ExecutionUnitKind = Literal["deterministic", "agent", "workflow", "tool_skill"]


class RunHardDeadlineExceeded(TimeoutError):
    """本轮动态硬时限已经耗尽。"""


class RunSoftDeadlineExceeded(TimeoutError):
    """本轮动态软时限已经耗尽，应使用已有结果降级。"""


_DEFAULT_USER_READ_ONLY_TOOLS = (
    "weather.get",
    "iwencai.query",
    "iwencai.screen",
    "iwencai.market.query",
    "iwencai.industry.query",
    "iwencai.index.query",
    "iwencai.rating.query",
    "iwencai.announcement.search",
    "iwencai.report.search",
    "iwencai.fund.screen",
    "knowledge.faq.search",
    "knowledge.pdf.catalog",
    "knowledge.pdf.search",
)


@dataclass(frozen=True, slots=True)
class AgentRuntimeContext:
    """由认证主体派生的不可变上下文，禁止使用请求体中的身份字段。

    ``tenant_id`` / ``user_id`` 带默认值，供 ``langgraph dev`` / Studio 在未传
    context 时仍能启动；生产路径必须通过 ``from_user`` 注入真实身份。
    """

    tenant_id: str = "studio"
    user_id: str = "0"
    conversation_id: str | None = None
    run_id: str | None = None
    permissions: tuple[str, ...] = ()
    started_monotonic: float = field(default_factory=time.monotonic)
    deadline_monotonic: float | None = None
    soft_deadline_monotonic: float | None = None
    hard_deadline_monotonic: float | None = None
    budget_tier: str = "research"
    unit_timeouts: dict[str, float] = field(default_factory=dict)
    max_concurrency: int = 4
    task_semaphore: asyncio.Semaphore = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        concurrency = max(1, int(self.max_concurrency))
        object.__setattr__(self, "max_concurrency", concurrency)
        object.__setattr__(self, "task_semaphore", asyncio.Semaphore(concurrency))

    def remaining_seconds(self) -> float | None:
        """返回动态硬时限与入口保险丝中更短的剩余时间。"""
        deadlines = [
            item
            for item in (self.deadline_monotonic, self.hard_deadline_monotonic)
            if item is not None
        ]
        if not deadlines:
            return None
        return max(0.0, min(deadlines) - time.monotonic())

    def soft_remaining_seconds(self) -> float | None:
        """返回软时限剩余时间；未完成难度分级时返回空。"""
        if self.soft_deadline_monotonic is None:
            return None
        return max(0.0, self.soft_deadline_monotonic - time.monotonic())

    def soft_deadline_exceeded(self) -> bool:
        """判断是否应停止扩张任务并使用已有结果降级回答。"""
        remaining = self.soft_remaining_seconds()
        return remaining is not None and remaining <= 0

    def configure_budget(
        self,
        *,
        budget_tier: str,
        soft_seconds: float,
        hard_seconds: float,
        unit_timeouts: dict[str, float],
    ) -> None:
        """分析完成后按确定性预算档位冻结本轮动态预算。"""
        soft = max(0.0, float(soft_seconds))
        hard = max(soft, float(hard_seconds))
        object.__setattr__(self, "budget_tier", budget_tier)
        object.__setattr__(
            self,
            "soft_deadline_monotonic",
            self.started_monotonic + soft,
        )
        object.__setattr__(
            self,
            "hard_deadline_monotonic",
            self.started_monotonic + hard,
        )
        object.__setattr__(
            self,
            "unit_timeouts",
            {
                str(kind): max(0.0, float(timeout))
                for kind, timeout in unit_timeouts.items()
            },
        )

    def execution_timeout_for(
        self,
        kind: ExecutionUnitKind,
        *,
        default_seconds: float,
    ) -> tuple[float, str]:
        """返回执行窗口及最先触发它的 unit、soft 或 hard 边界。"""
        configured = self.unit_timeouts.get(kind, default_seconds)
        candidates: list[tuple[str, float]] = [
            ("unit", max(0.0, float(configured))),
        ]
        soft_remaining = self.soft_remaining_seconds()
        hard_remaining = self.remaining_seconds()
        if soft_remaining is not None:
            candidates.append(("soft", soft_remaining))
        if hard_remaining is not None:
            candidates.append(("hard", hard_remaining))
        limit, timeout = min(
            candidates,
            key=lambda item: (
                item[1],
                {"hard": 0, "soft": 1, "unit": 2}[item[0]],
            ),
        )
        return timeout, limit

    @classmethod
    def from_user(
        cls,
        user: object,
        *,
        conversation_id: str | int | None = None,
        run_id: str | None = None,
        deadline_seconds: float | None = None,
        max_concurrency: int = 4,
    ) -> "AgentRuntimeContext":
        tenant_id = str(getattr(user, "tenant_id", "default"))
        user_id = str(getattr(user, "id"))
        role = str(getattr(user, "role", ""))
        if not tenant_id or tenant_id == "None":
            raise ValueError("认证主体缺少 tenant_id")
        return cls(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=str(conversation_id) if conversation_id is not None else None,
            run_id=run_id,
            permissions=("*",)
            if role == "admin"
            else (_DEFAULT_USER_READ_ONLY_TOOLS if role == "user" else ()),
            deadline_monotonic=(
                time.monotonic() + max(0.0, float(deadline_seconds))
                if deadline_seconds is not None
                else None
            ),
            max_concurrency=max_concurrency,
        )
