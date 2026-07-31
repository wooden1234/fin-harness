"""Research Workflow 内部受限 Deep Agent 深度研究运行时。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.exceptions import ContextOverflowError
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langgraph.runtime import Runtime

from agents.deep_agent_support import (
    build_governed_tools,
    ensure_financial_deep_agent_profile,
    run_context_from_runtime,
)
from agents.context_space import ContextCounters, ContextMeasurement, ContextSpaceType
from agents.context_space.events import record_context_event
from agents.research_workflow.deep_agent.spec import (
    DEEP_RESEARCH_SPEC,
    DeepResearchSpec,
)
from agents.research_workflow.deep_agent.context_middleware import (
    GovernedResearchSummarizationMiddleware,
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


def _assert_single_summary_middleware(middleware: Sequence[Any]) -> None:
    """启动时确保项目只注入一个摘要 owner。"""
    from deepagents.middleware.summarization import SummarizationMiddleware

    count = sum(isinstance(item, SummarizationMiddleware) for item in middleware)
    if count != 1:
        raise RuntimeError(f"deep_agent_summary_middleware_count={count}")


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
    collector: list[dict[str, Any]],
) -> list[Evidence]:
    """只接收工具返回的真实 Evidence，禁止按调用记录生成占位证据。"""
    evidence: list[Evidence] = []
    seen: set[str] = set()
    for item in collector:
        if not item.get("ok"):
            continue
        data = item.get("data")
        if not isinstance(data, Mapping):
            continue
        for raw in list(data.get("evidence") or []):
            try:
                parsed = raw if isinstance(raw, Evidence) else Evidence.model_validate(raw)
            except ValueError:
                continue
            if parsed.metadata.get("quality_status") in {"quarantined", "rejected"}:
                continue
            if parsed.evidence_id not in seen:
                seen.add(parsed.evidence_id)
                evidence.append(parsed)
    return evidence


def skills_for_data_sources(
    data_sources: Sequence[str],
    *,
    spec: DeepResearchSpec = DEEP_RESEARCH_SPEC,
) -> tuple[str, ...]:
    """将语义来源确定性映射为最小 Skill 集。"""
    sources = set(data_sources)
    mapping = {
        "market": {
            "stock-screening",
            "market-quotes",
            "industry-data",
            "index-data",
        },
        "research": {
            "announcement-search",
            "research-report-search",
            "institution-rating",
        },
        "finance_rag": {"dependency-analysis"},
        "local_documents": {"pdf-knowledge"},
        "stable_rules": {"faq-knowledge"},
    }
    allowed = {"dependency-analysis"}
    for source in sources:
        allowed.update(mapping.get(source, set()))
    return tuple(skill for skill in spec.skills if skill in allowed)


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


def _research_question_ids(results: Sequence[AgentResult]) -> tuple[str, ...]:
    """从可信研究计划投影中提取工具可使用的问题 ID。"""
    values: list[str] = []
    for result in results:
        if result.metadata.get("result_type") != "research_plan":
            continue
        data = result.structured_data or {}
        for question in list(data.get("questions") or []):
            if not isinstance(question, Mapping):
                continue
            question_id = str(question.get("question_id") or "").strip()
            if question_id and question_id not in values:
                values.append(question_id)
    return tuple(values)


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
    allowed_research_question_ids: Sequence[str] | None = None,
    max_compaction_rounds: int = 2,
    return_context_middleware: bool = False,
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
        allowed_research_question_ids=allowed_research_question_ids,
    )
    if dependencies:
        governed_tools.insert(0, _dependency_tool(dependencies, collector))
    active_llm = llm or get_faq_llm()
    context_middleware = GovernedResearchSummarizationMiddleware(
        active_llm,
        max_compaction_rounds=max_compaction_rounds,
    )
    middleware = [context_middleware]
    _assert_single_summary_middleware(middleware)
    agent = create_deep_agent(
        model=active_llm,
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
        middleware=middleware,
        name=spec.agent_id,
    )
    if return_context_middleware:
        return agent, context_middleware
    return agent


async def run_deep_research_agent(
    state: Mapping[str, Any],
    *,
    query: str,
    config: RunnableConfig | None = None,
    runtime: Runtime[AgentRuntimeContext] | None = None,
    llm: BaseChatModel | None = None,
    spec: DeepResearchSpec = DEEP_RESEARCH_SPEC,
    parent_compaction_round_count: int = 0,
    allowed_skills: Sequence[str] | None = None,
) -> AgentResult:
    """运行多轮研究并将自由输出收敛为统一 AgentResult。"""
    collector: list[dict[str, Any]] = []
    dependencies = _dependency_results(state)
    context_middleware: GovernedResearchSummarizationMiddleware | None = None

    async def _record_deep_context_events() -> None:
        if context_middleware is None:
            return
        total_rounds = parent_compaction_round_count + context_middleware.compaction_round_count
        counters = ContextCounters(
            compaction_round_count=total_rounds,
            summary_attempt_count=context_middleware.summary_attempt_count,
            snip_count=context_middleware.snip_count,
            provider_retry_count=context_middleware.provider_retry_count,
        )
        estimated = max(0, int(context_middleware.last_estimated_tokens))
        measurement = ContextMeasurement(
            estimated_tokens=estimated,
            counter_kind="approximate",
            effective_limit=16_000,
            utilization_ratio=estimated / 16_000,
            trigger_exceeded=estimated >= 12_000,
            admission_exceeded=estimated > 13_600,
        )
        parent_run_id = str(
            getattr(runtime.context, "run_id", "")
            if runtime is not None and runtime.context is not None
            else ""
        )
        parent_space_id = f"research_run:{parent_run_id}" if parent_run_id else None
        if context_middleware.compaction_round_count:
            await record_context_event(
                runtime,
                space_type=ContextSpaceType.DEEP_AGENT_LOOP,
                event_type="context.compaction_completed",
                measurement=measurement,
                counters=counters,
                agent_id=spec.agent_id,
                parent_space_id=parent_space_id,
            )
        if context_middleware.summary_attempt_count > context_middleware.compaction_round_count:
            await record_context_event(
                runtime,
                space_type=ContextSpaceType.DEEP_AGENT_LOOP,
                event_type="context.summary_repair",
                measurement=measurement,
                counters=counters,
                agent_id=spec.agent_id,
                parent_space_id=parent_space_id,
            )
        if context_middleware.snip_count:
            await record_context_event(
                runtime,
                space_type=ContextSpaceType.DEEP_AGENT_LOOP,
                event_type="context.snip_applied",
                measurement=measurement,
                counters=counters,
                agent_id=spec.agent_id,
                parent_space_id=parent_space_id,
            )
        if context_middleware.provider_retry_count:
            await record_context_event(
                runtime,
                space_type=ContextSpaceType.DEEP_AGENT_LOOP,
                event_type="context.provider_overflow",
                measurement=measurement,
                counters=counters,
                agent_id=spec.agent_id,
                parent_space_id=parent_space_id,
            )
            await record_context_event(
                runtime,
                space_type=ContextSpaceType.DEEP_AGENT_LOOP,
                event_type="context.provider_retry",
                measurement=measurement,
                counters=counters,
                agent_id=spec.agent_id,
                parent_space_id=parent_space_id,
            )
    try:
        selected_skills = tuple(allowed_skills or spec.skills)
        binding = resolve_skill_binding(selected_skills)
        built_agent = build_deep_research_agent(
            run_context=run_context_from_runtime(runtime, agent_name=spec.agent_id),
            collector=collector,
            llm=llm,
            spec=spec,
            binding=binding,
            dependencies=dependencies,
            allowed_research_question_ids=_research_question_ids(dependencies),
            max_compaction_rounds=max(
                0,
                2 - max(0, int(parent_compaction_round_count)),
            ),
            return_context_middleware=True,
        )
        if (
            isinstance(built_agent, tuple)
            and len(built_agent) == 2
            and isinstance(
                built_agent[1],
                GovernedResearchSummarizationMiddleware,
            )
        ):
            agent, context_middleware = built_agent
        else:
            # 保持第三方构造器和测试替身只返回 Agent 的兼容性。
            agent = built_agent
        messages = list(state.get("messages") or []) or [HumanMessage(content=query)]
        invocation_config = dict(config or {})
        invocation_config["recursion_limit"] = min(
            int(invocation_config.get("recursion_limit") or spec.recursion_limit),
            spec.recursion_limit,
        )
        output = await agent.ainvoke({"messages": messages}, config=invocation_config)
    except Exception as exc:  # noqa: BLE001
        if isinstance(exc, ContextOverflowError):
            await record_context_event(
                runtime,
                space_type=ContextSpaceType.DEEP_AGENT_LOOP,
                event_type="context.provider_overflow",
                counters=ContextCounters(
                    compaction_round_count=parent_compaction_round_count
                    + (
                        context_middleware.compaction_round_count
                        if context_middleware
                        else 0
                    ),
                    snip_count=context_middleware.snip_count if context_middleware else 0,
                    provider_retry_count=(
                        context_middleware.provider_retry_count
                        if context_middleware
                        else 0
                    ),
                ),
                agent_id=spec.agent_id,
                details={"error_code": type(exc).__name__},
            )
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
                "skills": list(allowed_skills or spec.skills),
                "tool_results": collector,
                "context_counters": (
                    {
                        "compaction_round_count": parent_compaction_round_count
                        + context_middleware.compaction_round_count,
                        "summary_attempt_count": context_middleware.summary_attempt_count,
                        "snip_count": context_middleware.snip_count,
                        "provider_retry_count": context_middleware.provider_retry_count,
                    }
                    if context_middleware is not None
                    else {}
                ),
            },
        )

    await _record_deep_context_events()

    evidence = _dependency_evidence(dependencies)
    seen_evidence = {item.evidence_id for item in evidence}
    evidence.extend(
        item
        for item in _evidence_from_collector(collector)
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
            "skills": list(allowed_skills or spec.skills),
            "tool_call_count": len(collector),
            "tool_results": collector,
            "context_counters": {
                "compaction_round_count": parent_compaction_round_count
                + (context_middleware.compaction_round_count if context_middleware else 0),
                "summary_attempt_count": (
                    context_middleware.summary_attempt_count
                    if context_middleware
                    else 0
                ),
                "snip_count": context_middleware.snip_count if context_middleware else 0,
                "provider_retry_count": (
                    context_middleware.provider_retry_count
                    if context_middleware
                    else 0
                ),
            },
        },
    )


__all__ = [
    "_assert_single_summary_middleware",
    "build_deep_research_agent",
    "run_deep_research_agent",
    "skills_for_data_sources",
]
