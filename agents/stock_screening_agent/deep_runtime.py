"""选股 Agent 的 Deep Agent 运行时。"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.runtime import Runtime

from agents.llm import get_faq_llm
from agents.orchestrator.contracts import AgentResult
from agents.runtime_context import AgentRuntimeContext
from agents.stock_screening_agent.result_mapper import (
    agent_result_from_tool_updates,
)
from agents.stock_screening_agent.skill_binding import (
    SkillBinding,
    resolve_skill_binding,
)
from agents.stock_screening_agent.spec import (
    STOCK_SCREENING_SPEC,
    StockScreeningSpec,
)
from harness.context import RunContext, build_run_context
from tools import execute_tool, get_langchain_tool, load_all_tools, validate_tool_ids

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PROFILE_LOCK = threading.Lock()
_PROFILE_READY = False
_EXCLUDED_BUILTIN_TOOLS = frozenset(
    {"execute", "task", "write_todos", "write_file", "edit_file"}
)


def _run_context_from_runtime(
    runtime: Runtime[AgentRuntimeContext] | None,
    *,
    agent_name: str,
) -> RunContext:
    context = runtime.context if runtime is not None else None
    return build_run_context(
        user_id=getattr(context, "user_id", None),
        tenant_id=getattr(context, "tenant_id", None),
        conversation_id=getattr(context, "conversation_id", None),
        permissions=tuple(getattr(context, "permissions", ()) or ()),
        metadata={"agent": agent_name, "run_id": getattr(context, "run_id", None)},
    )


def _ensure_harness_profile() -> None:
    global _PROFILE_READY
    if _PROFILE_READY:
        return
    with _PROFILE_LOCK:
        if _PROFILE_READY:
            return
        from deepagents import (
            GeneralPurposeSubagentProfile,
            HarnessProfile,
            register_harness_profile,
        )

        profile = HarnessProfile(
            excluded_tools=_EXCLUDED_BUILTIN_TOOLS,
            general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
            system_prompt_suffix=(
                "你是金融选股子 Agent。只使用已绑定的问财选股工具与 skills 说明；"
                "禁止执行 Shell、写文件或委派子 Agent。"
            ),
        )
        for key in (
            "deepseek",
            "openai",
            "anthropic",
            "google_genai",
            "fakelistchatmodel",
        ):
            register_harness_profile(key, profile)
        _PROFILE_READY = True


def _build_governed_tools(
    tool_ids: Sequence[str],
    *,
    run_context: RunContext,
    collector: list[dict[str, Any]],
) -> list[BaseTool]:
    load_all_tools()
    ids = [str(item) for item in tool_ids]
    validate_tool_ids(ids)
    allowed = frozenset(ids)
    tools: list[BaseTool] = []
    for tool_id in ids:
        base = get_langchain_tool(tool_id)

        async def _ainvoke(_tool_id: str = tool_id, **kwargs: Any) -> Any:
            result = await execute_tool(
                _tool_id,
                run_context,
                arguments=dict(kwargs),
                allowed_tool_ids=allowed,
            )
            collector.append(
                {
                    "tool_id": _tool_id,
                    "ok": result.ok,
                    "data": result.data,
                    "error": result.error,
                }
            )
            if result.ok:
                return result.data if result.data is not None else {"ok": True}
            return {"ok": False, "error": result.error or "tool_execution_failed"}

        tools.append(
            StructuredTool.from_function(
                coroutine=_ainvoke,
                name=base.name,
                description=base.description,
                args_schema=getattr(base, "args_schema", None),
            )
        )
    return tools


def _system_prompt(spec: StockScreeningSpec, binding: SkillBinding) -> str:
    bound = ", ".join(binding.tool_ids)
    skill_names = ", ".join(document.name for document in binding.documents)
    return f"""{spec.role_prompt}

请先查阅当前 Skill 说明：{skill_names}，再执行选股。
当前只允许使用已绑定的受控工具：{bound}。
不要执行 Shell，不要自行构造 HTTP 请求，不要编造问财返回的股票数据。
"""


def _answer_from_messages(messages: Sequence[Any], *, busy_answer: str) -> str:
    for message in reversed(list(messages)):
        if isinstance(message, AIMessage):
            if getattr(message, "tool_calls", None):
                content = str(message.content or "").strip()
                if not content:
                    continue
            text = str(message.content or "").strip()
            if text:
                return text
    return busy_answer


def build_deep_stock_screening_agent(
    *,
    spec: StockScreeningSpec = STOCK_SCREENING_SPEC,
    run_context: RunContext,
    collector: list[dict[str, Any]],
    llm: BaseChatModel | None = None,
    binding: SkillBinding | None = None,
):
    from deepagents import create_deep_agent
    from deepagents.backends.filesystem import FilesystemBackend
    from deepagents.middleware.filesystem import FilesystemPermission

    _ensure_harness_profile()
    active_binding = binding or resolve_skill_binding(spec.skills)
    skill_paths = list(active_binding.skill_paths)
    return create_deep_agent(
        model=llm or get_faq_llm(),
        tools=_build_governed_tools(
            active_binding.tool_ids,
            run_context=run_context,
            collector=collector,
        ),
        system_prompt=_system_prompt(spec, active_binding),
        skills=skill_paths,
        backend=FilesystemBackend(root_dir=str(_PROJECT_ROOT)),
        permissions=[
            FilesystemPermission(
                operations=["read"],
                paths=[f"{path}/**" for path in skill_paths],
                mode="allow",
            ),
            FilesystemPermission(
                operations=["write"],
                paths=["**"],
                mode="deny",
            ),
        ],
        name=spec.agent_id,
    )


async def run_stock_screening_deep_agent(
    state: Mapping[str, Any],
    *,
    query: str,
    config: RunnableConfig | None = None,
    runtime: Runtime[AgentRuntimeContext] | None = None,
    llm: BaseChatModel | None = None,
    spec: StockScreeningSpec = STOCK_SCREENING_SPEC,
) -> AgentResult:
    collector: list[dict[str, Any]] = []
    try:
        binding = resolve_skill_binding(spec.skills)
        agent = build_deep_stock_screening_agent(
            spec=spec,
            run_context=_run_context_from_runtime(runtime, agent_name=spec.agent_id),
            collector=collector,
            llm=llm,
            binding=binding,
        )
        messages = list(state.get("messages") or []) or [HumanMessage(content=query)]
        output = await agent.ainvoke({"messages": messages}, config=config)
    except Exception as exc:  # noqa: BLE001
        return AgentResult(
            task_id=spec.default_task_id,
            agent_id=spec.agent_id,
            status="failed",
            answer=spec.busy_answer,
            error_code=type(exc).__name__,
            gaps=[str(exc)],
            metadata={
                "query": query,
                "skills": list(spec.skills),
                "runtime": "deep_agent",
                "tool_results": collector,
            },
        )

    result = agent_result_from_tool_updates(
        query,
        {
            "messages": [
                AIMessage(
                    content=_answer_from_messages(
                        list(output.get("messages") or []),
                        busy_answer=spec.busy_answer,
                    )
                )
            ],
            "tool_results": collector,
        },
        spec=spec,
        binding=binding,
    )
    metadata = dict(result.metadata)
    metadata["runtime"] = "deep_agent"
    return result.model_copy(update={"metadata": metadata})


__all__ = [
    "build_deep_stock_screening_agent",
    "run_stock_screening_deep_agent",
]
