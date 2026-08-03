"""Main DeepAgent 的声明式组装契约。"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from agents.deep_agent_support import ensure_financial_deep_agent_profile
from agents.llm import get_faq_llm
from agents.main_deep_agent.config import MAIN_SKILL_SOURCES, build_main_backend
from agents.main_deep_agent.contracts import MainAgentResponse
from agents.main_deep_agent.middleware.budget import MainAgentBudgetController
from agents.main_deep_agent.middleware.finalization import (
    MainAgentFailureFinalizationMiddleware,
)
from agents.main_deep_agent.middleware.quality import MainEvidenceQualityMiddleware
from agents.main_deep_agent.prompts import build_main_system_prompt
from agents.main_deep_agent.state import MainAgentProgressJournal
from agents.main_deep_agent.tools.factory import build_main_tools
from agents.research_workflow.deep_agent.context_middleware import (
    GovernedResearchSummarizationMiddleware,
)
from agents.runtime_context import AgentRuntimeContext
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(service="main_deep_agent")


@dataclass(frozen=True, slots=True)
class MainDeepAgentAssembly:
    """集中描述 create_deep_agent 的可变组装部分。"""

    model: BaseChatModel
    tools: tuple[BaseTool, ...]
    system_prompt: str
    middleware: tuple[AgentMiddleware, ...] = ()
    skills: tuple[str, ...] = ()
    subagents: tuple[Any, ...] = ()
    memory: tuple[str, ...] = ()
    backend: Any = None
    state_schema: type | None = None
    context_schema: type | None = None
    response_format: Any = field(
        default_factory=lambda: ToolStrategy(
            MainAgentResponse,
            handle_errors=False,
        )
    )
    name: str = "main_deep_agent"

    def create(self):
        """按声明式配置创建 DeepAgent。"""
        from deepagents import create_deep_agent

        return create_deep_agent(
            model=self.model,
            tools=list(self.tools),
            system_prompt=self.system_prompt,
            middleware=list(self.middleware),
            skills=list(self.skills) or None,
            subagents=list(self.subagents) or None,
            memory=list(self.memory) or None,
            backend=self.backend,
            state_schema=self.state_schema,
            context_schema=self.context_schema,
            response_format=self.response_format,
            name=self.name,
        )


def build_main_deep_agent(
    *,
    context: AgentRuntimeContext,
    budget: MainAgentBudgetController,
    journal: MainAgentProgressJournal,
    investment_action_sensitive: bool,
    llm: BaseChatModel | None = None,
):
    """按声明式 Assembly 创建无 Shell、无写入、无子代理的 Main DeepAgent。"""
    ensure_financial_deep_agent_profile()
    active_llm = llm or get_faq_llm()
    governed_tools = build_main_tools(
        context=context,
        budget=budget,
        journal=journal,
    )
    assembly = MainDeepAgentAssembly(
        model=active_llm,
        tools=tuple(governed_tools),
        system_prompt=build_main_system_prompt(
            investment_action_sensitive=investment_action_sensitive,
        ),
        middleware=(
            MainAgentFailureFinalizationMiddleware(
                budget,
                business_tool_names={tool.name for tool in governed_tools},
            ),
            GovernedResearchSummarizationMiddleware(
                active_llm,
                journal=journal,
                max_compaction_rounds=2,
            ),
            # create_deep_agent 会在用户中间件外拼接内置中间件。当前依赖版本中
            # 质量门是唯一的 after_agent；新增同类钩子时必须复核逆序和跳转短路语义。
            MainEvidenceQualityMiddleware(
                journal=journal,
                budget=budget,
                investment_action_sensitive=investment_action_sensitive,
            ),
        ),
        skills=MAIN_SKILL_SOURCES,
        backend=build_main_backend(),
    )
    return assembly.create()


def _last_text(messages: Sequence[Any]) -> str:
    for message in reversed(list(messages)):
        if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
            text = message.content if isinstance(message.content, str) else str(message.content)
            if text.strip():
                return text.strip()
    return ""


async def run_main_deep_agent(
    *,
    messages: Sequence[Any],
    context: AgentRuntimeContext,
    config: RunnableConfig = None,
    investment_action_sensitive: bool,
    llm: BaseChatModel | None = None,
) -> tuple[MainAgentResponse | None, MainAgentProgressJournal, MainAgentBudgetController, str, str]:
    """执行 Main DeepAgent，并保留结构化失败和硬超时状态。"""
    journal = MainAgentProgressJournal()
    budget = MainAgentBudgetController(
        started_monotonic=context.started_monotonic,
        hard_seconds=float(settings.MAIN_AGENT_DIRECT_HARD_DEADLINE_SEC),
    )
    agent = build_main_deep_agent(
        context=context,
        budget=budget,
        journal=journal,
        investment_action_sensitive=investment_action_sensitive,
        llm=llm,
    )
    deadline = context.started_monotonic + float(
        settings.MAIN_AGENT_DIRECT_HARD_DEADLINE_SEC
    )
    result: Mapping[str, Any] = {}
    status = "completed"
    error = ""
    exception_fallback_text = ""
    try:
        async with asyncio.timeout_at(deadline) as timeout_scope:
            budget.attach(timeout_scope)
            invoke_config = dict(config or {})
            invoke_config["recursion_limit"] = int(settings.MAIN_AGENT_RECURSION_LIMIT)
            result = await agent.ainvoke(
                {"messages": list(messages)},
                config=invoke_config,
            )
            journal.set_agent_todos(result.get("todos") or [])
            budget.model_rounds = sum(
                isinstance(message, AIMessage)
                for message in list(result.get("messages") or [])
            )
    except TimeoutError:
        status = "hard_timeout"
        error = "main_agent_hard_deadline_exceeded"
    except Exception as exc:
        status = "structured_output_failed"
        error = type(exc).__name__
        failed_message = getattr(exc, "ai_message", None)
        if isinstance(failed_message, AIMessage):
            content = failed_message.content
            exception_fallback_text = (
                content if isinstance(content, str) else str(content)
            ).strip()[:800]
        logger.warning(
            "main agent failed: exception_type={} message={} tool_calls={} journal_entries={}",
            type(exc).__name__,
            str(exc)[:500],
            budget.tool_calls,
            len(journal.entries),
        )

    stale_quality_response = (
        journal.quality_revision_outcome == "no_new_structured_response"
    )
    raw_response = None if stale_quality_response else result.get("structured_response")
    response: MainAgentResponse | None = None
    if raw_response is not None:
        try:
            response = (
                raw_response
                if isinstance(raw_response, MainAgentResponse)
                else MainAgentResponse.model_validate(raw_response)
            )
        except ValueError:
            status = "structured_output_failed"
            error = "main_agent_response_validation_failed"

    fallback_text = _last_text(list(result.get("messages") or [])) or exception_fallback_text
    if stale_quality_response:
        status = "structured_output_failed"
        error = "quality_revision_missing_structured_response"
    if (
        response is None
        and not stale_quality_response
        and not journal.entries
        and fallback_text
    ):
        response = MainAgentResponse(mode="direct", direct_answer=fallback_text)
        status = "completed"
        error = ""
    return response, journal, budget, status, error



__all__ = [
    "MainDeepAgentAssembly",
    "build_main_deep_agent",
    "run_main_deep_agent",
]
