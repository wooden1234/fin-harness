"""Research Workflow 的研究级计划生成器。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

from agents.llm import get_router_llm
from agents.orchestrator.agent_registry import get_agent_spec
from agents.orchestrator.contracts import (
    EvidencePolicy,
    TaskPlan,
    TaskSpec,
)
from agents.orchestrator.task_identity import validate_task_plan
from agents.structured_output import ainvoke_json_output
from agents.research_workflow.contracts import (
    ResearchPlan,
    ResearchPlanDraft,
    ResearchQuestion,
    ResearchQuestionDraft,
)
from agents.research_workflow.prompts import RESEARCH_PLANNER_SYSTEM_PROMPT
from agents.research_workflow.spec import RESEARCH_WORKFLOW_SPEC
from app.core.logger import get_logger

_ALLOWED_DATA_SOURCES = frozenset(RESEARCH_WORKFLOW_SPEC.allowed_data_sources)
_MAX_RESEARCH_QUESTIONS = 8
logger = get_logger(service="research_workflow_planner")

_SOURCE_TASK_IDS = {
    "market": ("research:stock_screening",),
    "research": ("research:announcement", "research:report"),
    "finance_rag": ("research:finance",),
    "local_documents": ("deep-research",),
    "stable_rules": ("deep-research",),
}
_QUESTION_TASK_IDS = {
    "market_context": ("research:stock_screening",),
    "public_disclosures": ("research:announcement",),
    "institution_views": ("research:report",),
    "financial_facts": ("research:finance",),
    "local_document_facts": ("deep-research",),
    "stable_rule_context": ("deep-research",),
}
_DEEP_AGENT_SOURCES = frozenset({"local_documents", "stable_rules"})


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return list(
        dict.fromkeys(
            str(item).strip()
            for item in value
            if str(item).strip()
        )
    )


def _source_scope(
    task_input: Mapping[str, Any],
) -> tuple[list[str], str, list[str]]:
    """区分字段缺失与显式空范围，并返回非法来源。"""
    if "data_sources" not in task_input:
        return (
            list(RESEARCH_WORKFLOW_SPEC.default_data_sources),
            "missing_default",
            [],
        )
    requested = _string_list(task_input.get("data_sources"))
    valid = [item for item in requested if item in _ALLOWED_DATA_SOURCES]
    invalid = [item for item in requested if item not in _ALLOWED_DATA_SOURCES]
    return valid, ("explicit" if valid else "explicit_empty"), invalid


def _task_validation_scope(
    query: str,
    task_identity: Mapping[str, Any] | None,
) -> str:
    identity = task_identity or {}
    parent = str(
        identity.get("idempotency_key")
        or identity.get("logical_task_id")
        or identity.get("task_id")
        or "direct"
    )
    return f"research:{parent}:{query}"


def _validate_source_tasks(
    query: str,
    tasks: Sequence[TaskSpec],
    *,
    validation_scope: str,
) -> list[TaskSpec]:
    """在领域执行边界重新校验任务能力、身份、幂等键和依赖。"""
    with_policies = [
        task.model_copy(
            update={
                "evidence_policy": get_agent_spec(
                    task.agent_id
                ).evidence_policy
            }
        )
        for task in tasks
    ]
    plan = validate_task_plan(
        TaskPlan(
            plan_id=f"research:sources:{validation_scope}",
            query=query,
            tasks=with_policies,
            max_replans=0,
        ),
        scope=validation_scope,
    )
    return list(plan.tasks)


def _source_tasks(
    query: str,
    data_sources: Sequence[str],
    *,
    validation_scope: str,
) -> list[TaskSpec]:
    """将研究范围映射为经过统一校验的 Workflow 内部来源任务。"""
    sources = set(data_sources)
    tasks: list[TaskSpec] = []
    if "market" in sources:
        tasks.append(
            TaskSpec(
                task_id="research:stock_screening",
                objective=f"围绕“{query}”获取相关标的和市场候选数据",
                agent_id="stock_screening_agent",
                required_capabilities=["iwencai.screen"],
            )
        )
    if "research" in sources:
        tasks.extend(
            [
                TaskSpec(
                    task_id="research:announcement",
                    objective=f"围绕“{query}”检索相关公告和重大披露",
                    agent_id="research_retrieval_workflow",
                    required_capabilities=["iwencai.announcement.search"],
                    input_data={
                        "research_tool_id": "iwencai.announcement.search",
                    },
                ),
                TaskSpec(
                    task_id="research:report",
                    objective=f"围绕“{query}”检索相关研报和机构观点",
                    agent_id="research_retrieval_workflow",
                    required_capabilities=["iwencai.report.search"],
                    input_data={"research_tool_id": "iwencai.report.search"},
                ),
            ]
        )
    if "finance_rag" in sources:
        tasks.append(
            TaskSpec(
                task_id="research:finance",
                objective=f"围绕“{query}”核验财务表现和关键指标",
                agent_id="finance_agent",
                required_capabilities=["financial_query"],
            )
        )
    return _validate_source_tasks(
        query,
        tasks,
        validation_scope=validation_scope,
    )


def _research_questions(
    query: str,
    data_sources: Sequence[str],
) -> list[ResearchQuestionDraft]:
    sources = set(data_sources)
    if not sources:
        return []
    questions: list[ResearchQuestionDraft] = []
    if "market" in sources:
        questions.append(
            ResearchQuestionDraft(
                question_id="market_context",
                objective=f"与“{query}”相关的市场表现和候选范围是什么？",
                data_sources=["market"],
                evidence_requirements=["行情或候选集工具证据"],
            )
        )
    if "research" in sources:
        questions.extend(
            [
                ResearchQuestionDraft(
                    question_id="public_disclosures",
                    objective="公开公告披露了哪些可能影响结论的重大事实？",
                    data_sources=["research"],
                    evidence_requirements=["公告标题、发布时间和摘要"],
                ),
                ResearchQuestionDraft(
                    question_id="institution_views",
                    objective="机构观点有哪些共识、分歧和时效限制？",
                    data_sources=["research"],
                    evidence_requirements=["研报机构、日期、评级或摘要"],
                ),
            ]
        )
    if "finance_rag" in sources:
        questions.append(
            ResearchQuestionDraft(
                question_id="financial_facts",
                objective="财务事实是否支持研究假设和机构观点？",
                data_sources=["finance_rag"],
                evidence_requirements=["财务指标、报告期和数据口径"],
            )
        )
    if "local_documents" in sources:
        questions.append(
            ResearchQuestionDraft(
                question_id="local_document_facts",
                objective="本地已准入年报、宏观报告或研究材料提供了哪些可核验事实和观点？",
                data_sources=["local_documents"],
                evidence_requirements=["文档 ID、页码、章节、发布日期和原文片段"],
            )
        )
    if "stable_rules" in sources:
        questions.append(
            ResearchQuestionDraft(
                question_id="stable_rule_context",
                objective="哪些稳定规则、概念或制度模板构成研究结论的背景约束？",
                data_sources=["stable_rules"],
                evidence_requirements=["FAQ 文档 ID、生效日期和适用域"],
            )
        )
    questions.append(
        ResearchQuestionDraft(
            question_id="critical_review",
            objective="现有证据支持什么结论，反方证据和未解决缺口是什么？",
            data_sources=list(data_sources),
            evidence_requirements=["支持证据、反方证据和数据缺口"],
        )
    )
    return questions


def _question_policy(
    question_id: str,
    data_sources: Sequence[str],
    task_ids: Sequence[str],
) -> EvidencePolicy:
    # 综合判断至少需要两个独立来源组；来源范围不足时应显式降级为 partial。
    min_count = 2 if question_id == "critical_review" else 1
    return EvidencePolicy(
        required=True,
        min_count=max(1, min_count),
        require_provenance=True,
        require_structured_data=set(data_sources) == {"market"},
    )


def _map_questions(
    drafts: Sequence[ResearchQuestionDraft],
    *,
    data_sources: Sequence[str],
    source_tasks: Sequence[TaskSpec],
    issues: list[str],
) -> list[ResearchQuestion]:
    """把研究问题确定性映射到可执行任务和证据策略。"""
    allowed_sources = set(data_sources)
    executable_ids = {task.task_id for task in source_tasks}
    if allowed_sources & _DEEP_AGENT_SOURCES:
        executable_ids.add("deep-research")
    questions: list[ResearchQuestion] = []
    for item in drafts:
        requested_sources = _string_list(item.data_sources)
        invalid_sources = [
            source for source in requested_sources if source not in allowed_sources
        ]
        if invalid_sources:
            issues.append(
                f"question_sources_outside_scope:{item.question_id}:"
                + ",".join(invalid_sources)
            )
        question_sources = [
            source for source in requested_sources if source in allowed_sources
        ]
        if not requested_sources:
            question_sources = list(data_sources)
        eligible_task_ids = {
            task_id
            for source in question_sources
            for task_id in _SOURCE_TASK_IDS.get(source, ())
        }

        if item.question_id == "critical_review":
            task_ids = [
                task.task_id
                for task in source_tasks
                if task.task_id in eligible_task_ids
            ]
            if "deep-research" in eligible_task_ids:
                task_ids.append("deep-research")
        elif item.question_id in _QUESTION_TASK_IDS:
            task_ids = [
                task_id
                for task_id in _QUESTION_TASK_IDS[item.question_id]
                if task_id in executable_ids and task_id in eligible_task_ids
            ]
        else:
            task_ids = [
                task_id
                for source in question_sources
                for task_id in _SOURCE_TASK_IDS.get(source, ())
                if task_id in executable_ids
            ]
            task_ids = list(dict.fromkeys(task_ids))

        if not question_sources or not task_ids:
            issues.append(f"question_unexecutable:{item.question_id}")
            continue
        questions.append(
            ResearchQuestion(
                question_id=item.question_id,
                objective=item.objective,
                data_sources=question_sources,
                evidence_requirements=list(item.evidence_requirements),
                source_task_ids=task_ids,
                evidence_policy=_question_policy(
                    item.question_id,
                    question_sources,
                    task_ids,
                ),
            )
        )
    return questions


def build_research_plan(
    query: str,
    task_input: Mapping[str, Any] | None = None,
    *,
    task_identity: Mapping[str, Any] | None = None,
) -> ResearchPlan:
    """根据 Root 传入的研究范围提示生成内部研究计划。"""
    inputs = task_input if task_input is not None else {}
    data_sources, scope_origin, invalid_sources = _source_scope(inputs)
    entities = _string_list(inputs.get("entities"))
    validation_scope = _task_validation_scope(query, task_identity)
    source_tasks = _source_tasks(
        query,
        data_sources,
        validation_scope=validation_scope,
    )
    issues = (
        ["unsupported_data_sources:" + ",".join(invalid_sources)]
        if invalid_sources
        else []
    )
    questions = _map_questions(
        _research_questions(query, data_sources),
        data_sources=data_sources,
        source_tasks=source_tasks,
        issues=issues,
    )
    return ResearchPlan(
        query=query,
        entities=entities,
        data_sources=data_sources,
        questions=questions,
        source_tasks=source_tasks,
        rationale="根据请求画像提供的研究范围，生成内部问题清单和来源任务。",
        metadata={
            "planner": "deterministic",
            "source_scope_origin": scope_origin,
            "scope_blocked": not data_sources,
            "task_validation_scope": validation_scope,
            "validation_issues": issues,
        },
    )


def _validated_llm_plan(
    query: str,
    task_input: Mapping[str, Any],
    draft: ResearchPlanDraft,
    *,
    task_identity: Mapping[str, Any] | None = None,
) -> ResearchPlan:
    """将 LLM 候选计划收敛到 Root 授权范围和确定性任务白名单。"""
    allowed_scope, scope_origin, invalid_scope_sources = _source_scope(task_input)
    allowed_set = set(allowed_scope)

    issues: list[str] = (
        ["unsupported_scope_sources:" + ",".join(invalid_scope_sources)]
        if invalid_scope_sources
        else []
    )
    draft_sources_explicit = "data_sources" in draft.model_fields_set
    requested_sources = (
        _string_list(draft.data_sources)
        if draft_sources_explicit
        else list(allowed_scope)
    )
    invalid_sources = [
        item
        for item in requested_sources
        if item not in _ALLOWED_DATA_SOURCES or item not in allowed_set
    ]
    if invalid_sources:
        issues.append(
            "unsupported_data_sources:" + ",".join(invalid_sources)
        )
    data_sources = [
        item for item in requested_sources if item in allowed_set
    ]
    if draft_sources_explicit and not requested_sources:
        issues.append("llm_data_sources_explicit_empty")

    question_drafts: list[ResearchQuestionDraft] = []
    seen_question_ids: set[str] = set()
    for item in draft.questions[:_MAX_RESEARCH_QUESTIONS]:
        question_id = item.question_id.strip()
        objective = item.objective.strip()
        if not question_id or not objective or question_id in seen_question_ids:
            issues.append(f"invalid_or_duplicate_question:{question_id}")
            continue
        seen_question_ids.add(question_id)
        question_drafts.append(
            item.model_copy(
                update={
                    "question_id": question_id,
                    "objective": objective,
                    "data_sources": _string_list(item.data_sources),
                    "evidence_requirements": _string_list(
                        item.evidence_requirements
                    )[:8],
                }
            )
        )
    if not question_drafts and data_sources:
        raise ValueError("research_plan_questions_empty")
    if "critical_review" not in seen_question_ids:
        question_drafts = question_drafts[: _MAX_RESEARCH_QUESTIONS - 1]
        question_drafts.append(
            ResearchQuestionDraft(
                question_id="critical_review",
                objective="现有证据支持什么结论，反方证据和未解决缺口是什么？",
                data_sources=list(data_sources),
                evidence_requirements=["支持证据、反方证据和数据缺口"],
            )
        )
        issues.append("critical_review_added")
    question_drafts = question_drafts[:_MAX_RESEARCH_QUESTIONS]
    validation_scope = _task_validation_scope(query, task_identity)
    source_tasks = _source_tasks(
        query,
        data_sources,
        validation_scope=validation_scope,
    )
    questions = _map_questions(
        question_drafts,
        data_sources=data_sources,
        source_tasks=source_tasks,
        issues=issues,
    )
    if data_sources and not questions:
        raise ValueError("research_plan_questions_unexecutable")

    return ResearchPlan(
        query=query,
        entities=_string_list(task_input.get("entities")),
        data_sources=data_sources,
        questions=questions,
        source_tasks=source_tasks,
        rationale=draft.rationale.strip(),
        metadata={
            "planner": "llm_validated",
            "allowed_source_scope": allowed_scope,
            "source_scope_origin": scope_origin,
            "scope_blocked": not data_sources,
            "task_validation_scope": validation_scope,
            "validation_issues": issues,
        },
    )


async def plan_research_adaptive(
    query: str,
    task_input: Mapping[str, Any] | None = None,
    *,
    task_identity: Mapping[str, Any] | None = None,
    config: RunnableConfig | None = None,
    llm: BaseChatModel | None = None,
) -> ResearchPlan:
    """单次 LLM 规划；异常或校验失败时回退确定性研究计划。"""
    inputs = task_input if task_input is not None else {}
    allowed_scope, scope_origin, _ = _source_scope(inputs)
    if scope_origin == "explicit_empty":
        blocked = build_research_plan(
            query,
            inputs,
            task_identity=task_identity,
        )
        return blocked.model_copy(
            update={
                "metadata": {
                    **blocked.metadata,
                    "planner": "scope_blocked",
                }
            }
        )
    entities = _string_list(inputs.get("entities"))
    human_prompt = (
        f"用户研究问题：\n{query}\n\n"
        f"允许的数据范围：{', '.join(allowed_scope)}\n"
        f"已识别实体：{', '.join(entities) if entities else '无'}"
    )
    try:
        model = llm or get_router_llm()
        raw = await ainvoke_json_output(
            model,
            ResearchPlanDraft,
            [
                ("system", RESEARCH_PLANNER_SYSTEM_PROMPT),
                ("human", human_prompt),
            ],
            config=config,
        )
        draft = (
            raw
            if isinstance(raw, ResearchPlanDraft)
            else ResearchPlanDraft.model_validate(raw)
        )
        return _validated_llm_plan(
            query,
            inputs,
            draft,
            task_identity=task_identity,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "research planner llm failed, fallback deterministic: {}",
            type(exc).__name__,
        )
        fallback = build_research_plan(
            query,
            inputs,
            task_identity=task_identity,
        )
        return fallback.model_copy(
            update={
                "metadata": {
                    **fallback.metadata,
                    "planner": "deterministic_fallback",
                    "fallback_reason": type(exc).__name__,
                }
            }
        )


__all__ = [
    "build_research_plan",
    "plan_research_adaptive",
    "_validate_source_tasks",
]
