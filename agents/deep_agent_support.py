"""v2 Deep Agent 共用的权限、工具和 Harness 配置。"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from langgraph.runtime import Runtime

from agents.runtime_context import AgentRuntimeContext
from harness.context import RunContext, build_run_context
from tools import execute_tool, get_langchain_tool, load_all_tools, validate_tool_ids

_PROFILE_LOCK = threading.Lock()
_PROFILE_READY = False
_EXCLUDED_BUILTIN_TOOLS = frozenset(
    {"execute", "task", "write_file", "edit_file"}
)


def ensure_financial_deep_agent_profile() -> None:
    """注册所有金融 Deep Agent 共用的最小权限配置。"""
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
        from deepagents.middleware.summarization import SummarizationMiddleware

        profile = HarnessProfile(
            excluded_tools=_EXCLUDED_BUILTIN_TOOLS,
            # 项目注入唯一的受治理子类，精确排除框架默认实例。
            excluded_middleware=frozenset({SummarizationMiddleware}),
            general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
            system_prompt_suffix=(
                "你是受治理的金融专业 Agent。只能使用当前任务绑定的只读工具与 Skill；"
                "禁止执行 Shell、写文件或委派通用子 Agent。"
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


def run_context_from_runtime(
    runtime: Runtime[AgentRuntimeContext] | None,
    *,
    agent_name: str,
) -> RunContext:
    """把不可变 Agent 上下文转换为受治理 Tool 上下文。"""
    context = runtime.context if runtime is not None else None
    return build_run_context(
        user_id=getattr(context, "user_id", None),
        tenant_id=getattr(context, "tenant_id", None),
        conversation_id=getattr(context, "conversation_id", None),
        permissions=tuple(getattr(context, "permissions", ()) or ()),
        metadata={"agent": agent_name, "run_id": getattr(context, "run_id", None)},
    )


def build_governed_tools(
    tool_ids: Sequence[str],
    *,
    run_context: RunContext,
    collector: list[dict[str, Any]],
    max_tool_calls: int | None = None,
    allowed_research_question_ids: Sequence[str] | None = None,
) -> list[BaseTool]:
    """把注册 Tool 包装成带权限、审计和白名单的 Deep Agent Tool。"""
    load_all_tools()
    ids = [str(item) for item in tool_ids]
    validate_tool_ids(ids)
    allowed = frozenset(ids)
    allowed_question_ids = (
        frozenset(str(item) for item in allowed_research_question_ids)
        if allowed_research_question_ids is not None
        else None
    )
    governed_tools: list[BaseTool] = []
    for tool_id in ids:
        base = get_langchain_tool(tool_id)

        async def _ainvoke(_tool_id: str = tool_id, **kwargs: Any) -> Any:
            if max_tool_calls is not None and len(collector) >= max_tool_calls:
                return {"ok": False, "error": "tool_call_budget_exhausted"}
            if _tool_id in {"knowledge.faq.search", "knowledge.pdf.search"}:
                question_id = str(kwargs.get("research_question_id") or "")
                if allowed_question_ids is not None and question_id not in allowed_question_ids:
                    rejection = {
                        "tool_id": _tool_id,
                        "ok": False,
                        "data": None,
                        "error": "research_question_id_not_allowed",
                    }
                    collector.append(rejection)
                    return {"ok": False, "error": rejection["error"]}
            if _tool_id == "knowledge.pdf.search" and not any(
                item.get("tool_id") == "knowledge.pdf.catalog" and item.get("ok")
                for item in collector
            ):
                rejection = {
                    "tool_id": _tool_id,
                    "ok": False,
                    "data": None,
                    "error": "pdf_catalog_required_before_search",
                }
                collector.append(rejection)
                return {"ok": False, "error": rejection["error"]}
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

        governed_tools.append(
            StructuredTool.from_function(
                coroutine=_ainvoke,
                name=base.name,
                description=base.description,
                args_schema=getattr(base, "args_schema", None),
            )
        )
    return governed_tools


__all__ = [
    "build_governed_tools",
    "ensure_financial_deep_agent_profile",
    "run_context_from_runtime",
]
