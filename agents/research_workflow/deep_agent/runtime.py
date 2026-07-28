"""Research Workflow 内部受限 Deep Agent 深度研究运行时。"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langgraph.runtime import Runtime

from agents.deep_agent_support import (
    build_governed_tools,
    ensure_financial_deep_agent_profile,
    run_context_from_runtime,
)
from agents.research_workflow.deep_agent.spec import (
    DEEP_RESEARCH_SPEC,
    DeepResearchSpec,
)
from agents.llm import get_faq_llm
from agents.orchestrator.contracts import AgentResult, DeepResearchReport, Evidence
from agents.runtime_context import AgentRuntimeContext
from agents.stock_screening_agent.skill_binding import (
    SkillBinding,
    resolve_skill_binding,
)
from harness.context import RunContext

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _required_tools(binding: SkillBinding) -> tuple[str, ...]:
    """深度研究只绑定 Skill 必需工具，不自动启用可选 Web 兜底。"""
    return binding.required_tools


def _system_prompt(spec: DeepResearchSpec, binding: SkillBinding) -> str:
    skills = ", ".join(document.name for document in binding.documents)
    tools = ", ".join(_required_tools(binding))
    return f"""{spec.role_prompt}

当前可读取 Skill：{skills}。
当前只允许调用受治理工具：{tools}。
如果存在 `read_orchestrator_dependencies`，必须先读取来源 Agent 的完整结构化结果。
最多调用 {spec.max_tool_calls} 次工具；数据不足时返回缺口，不得继续猜测。
禁止使用模型记忆补充行情、财务数值、公告或研报内容。
不要把机构评级解释为确定收益或直接交易建议。
"""


def _last_answer(messages: Sequence[Any], *, fallback: str) -> str:
    for message in reversed(list(messages)):
        if isinstance(message, AIMessage):
            text = str(message.content or "").strip()
            if text and not getattr(message, "tool_calls", None):
                return text
    return fallback


def _evidence_from_collector(
    query: str,
    collector: list[dict[str, Any]],
) -> list[Evidence]:
    evidence: list[Evidence] = []
    for index, item in enumerate(collector):
        if not item.get("ok"):
            continue
        tool_id = str(item.get("tool_id") or "unknown")
        if tool_id == "orchestrator.dependencies":
            continue
        digest = hashlib.sha256(
            f"{query}:{tool_id}:{index}".encode("utf-8")
        ).hexdigest()[:12]
        evidence.append(
            Evidence(
                evidence_id=f"deep-research:{digest}",
                task_id=DEEP_RESEARCH_SPEC.default_task_id,
                source_type=tool_id,
                provider="iwencai",
                title=f"深度研究工具证据：{tool_id}",
                content=f"围绕“{query}”调用受治理工具 {tool_id}。",
                metadata={"tool_result_index": index},
            )
        )
    return evidence


def _dependency_results(state: Mapping[str, Any]) -> list[AgentResult]:
    results: list[AgentResult] = []
    for item in list(state.get("dependency_results") or []):
        try:
            results.append(
                item if isinstance(item, AgentResult) else AgentResult.model_validate(item)
            )
        except ValueError:
            continue
    return results


def _dependency_evidence(results: Sequence[AgentResult]) -> list[Evidence]:
    evidence: list[Evidence] = []
    seen: set[str] = set()
    for result in results:
        for item in result.evidence:
            if item.evidence_id not in seen:
                seen.add(item.evidence_id)
                evidence.append(item)
    return evidence


def _dependency_tool(
    dependencies: Sequence[AgentResult],
    collector: list[dict[str, Any]],
) -> StructuredTool:
    """把 Root 依赖以只读 Tool 暴露给 Deep Agent，避免拼接进用户文本。"""

    async def _read_dependencies() -> dict[str, Any]:
        payload = {"results": [item.model_dump(mode="json") for item in dependencies]}
        collector.append(
            {
                "tool_id": "orchestrator.dependencies",
                "ok": True,
                "data": payload,
                "error": "",
            }
        )
        return payload

    return StructuredTool.from_function(
        coroutine=_read_dependencies,
        name="read_orchestrator_dependencies",
        description="读取 Root Orchestrator 已完成来源任务的完整结构化结果和证据。",
    )


def build_deep_research_agent(
    *,
    run_context: RunContext,
    collector: list[dict[str, Any]],
    llm: BaseChatModel | None = None,
    spec: DeepResearchSpec = DEEP_RESEARCH_SPEC,
    binding: SkillBinding | None = None,
    dependencies: Sequence[AgentResult] = (),
):
    """创建只能读取绑定 Skill、只能调用白名单 Tool 的 Deep Agent。"""
    from deepagents import create_deep_agent
    from deepagents.backends.filesystem import FilesystemBackend
    from deepagents.middleware.filesystem import FilesystemPermission

    ensure_financial_deep_agent_profile()
    active_binding = binding or resolve_skill_binding(spec.skills)
    skill_paths = list(active_binding.skill_paths)
    governed_tools = build_governed_tools(
        _required_tools(active_binding),
        run_context=run_context,
        collector=collector,
        max_tool_calls=spec.max_tool_calls,
    )
    if dependencies:
        governed_tools.insert(0, _dependency_tool(dependencies, collector))
    return create_deep_agent(
        model=llm or get_faq_llm(),
        tools=governed_tools,
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


async def run_deep_research_agent(
    state: Mapping[str, Any],
    *,
    query: str,
    config: RunnableConfig | None = None,
    runtime: Runtime[AgentRuntimeContext] | None = None,
    llm: BaseChatModel | None = None,
    spec: DeepResearchSpec = DEEP_RESEARCH_SPEC,
) -> AgentResult:
    """运行多轮研究并将自由输出收敛为统一 AgentResult。"""
    collector: list[dict[str, Any]] = []
    dependencies = _dependency_results(state)
    try:
        binding = resolve_skill_binding(spec.skills)
        agent = build_deep_research_agent(
            run_context=run_context_from_runtime(runtime, agent_name=spec.agent_id),
            collector=collector,
            llm=llm,
            spec=spec,
            binding=binding,
            dependencies=dependencies,
        )
        messages = list(state.get("messages") or []) or [HumanMessage(content=query)]
        invocation_config = dict(config or {})
        invocation_config["recursion_limit"] = min(
            int(invocation_config.get("recursion_limit") or spec.recursion_limit),
            spec.recursion_limit,
        )
        output = await agent.ainvoke({"messages": messages}, config=invocation_config)
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
                "runtime": "deep_agent",
                "skills": list(spec.skills),
                "tool_results": collector,
            },
        )

    evidence = _dependency_evidence(dependencies)
    seen_evidence = {item.evidence_id for item in evidence}
    evidence.extend(
        item
        for item in _evidence_from_collector(query, collector)
        if item.evidence_id not in seen_evidence
    )
    answer = _last_answer(
        list(output.get("messages") or []),
        fallback=spec.busy_answer,
    )
    failed_tools = [
        str(item.get("tool_id") or "unknown")
        for item in collector
        if not item.get("ok")
    ]
    if evidence:
        status = "partial" if failed_tools else "completed"
        error_code = ""
        gaps = (
            [f"以下数据源调用失败：{', '.join(sorted(set(failed_tools)))}"]
            if failed_tools
            else []
        )
    else:
        status = "failed"
        error_code = "deep_research_evidence_missing"
        gaps = ["深度研究未取得可验证的工具证据"]

    source_tools = list(
        dict.fromkeys(
            str(item.get("tool_id") or "")
            for item in collector
            if item.get("ok") and item.get("tool_id") != "orchestrator.dependencies"
        )
    )
    source_tools = list(
        dict.fromkeys(
            [
                *(item.source_type for item in evidence),
                *source_tools,
            ]
        )
    )
    report = DeepResearchReport(
        query=query,
        summary=answer,
        source_tools=source_tools,
        evidence_ids=[item.evidence_id for item in evidence],
        gaps=gaps,
        metadata={"tool_call_count": len(collector)},
    )
    return AgentResult(
        task_id=spec.default_task_id,
        agent_id=spec.agent_id,
        status=status,
        answer=answer,
        structured_data=report.model_dump(),
        evidence=evidence,
        gaps=gaps,
        error_code=error_code,
        metadata={
            "query": query,
            "runtime": "deep_agent",
            "skills": list(spec.skills),
            "tool_call_count": len(collector),
            "tool_results": collector,
        },
    )


__all__ = ["build_deep_research_agent", "run_deep_research_agent"]
