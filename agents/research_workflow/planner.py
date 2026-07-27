"""Research Workflow 的研究级计划生成器。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

from agents.llm import get_router_llm
from agents.orchestrator.contracts import TaskSpec
from agents.research_workflow.contracts import (
    ResearchPlan,
    ResearchPlanDraft,
    ResearchQuestion,
)
from agents.research_workflow.prompts import RESEARCH_PLANNER_SYSTEM_PROMPT
from agents.research_workflow.spec import RESEARCH_WORKFLOW_SPEC
from app.core.logger import get_logger

_ALLOWED_DATA_SOURCES = frozenset(RESEARCH_WORKFLOW_SPEC.default_data_sources)
_MAX_RESEARCH_QUESTIONS = 8
logger = get_logger(service="research_workflow_planner")


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


def _source_tasks(query: str, data_sources: Sequence[str]) -> list[TaskSpec]:
    """将研究范围映射为 Workflow 内部来源任务。"""
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
    return tasks


def _research_questions(
    query: str,
    data_sources: Sequence[str],
) -> list[ResearchQuestion]:
    sources = set(data_sources)
    questions: list[ResearchQuestion] = []
    if "market" in sources:
        questions.append(
            ResearchQuestion(
                question_id="market_context",
                objective=f"与“{query}”相关的市场表现和候选范围是什么？",
                evidence_requirements=["行情或候选集工具证据"],
            )
        )
    if "research" in sources:
        questions.extend(
            [
                ResearchQuestion(
                    question_id="public_disclosures",
                    objective="公开公告披露了哪些可能影响结论的重大事实？",
                    evidence_requirements=["公告标题、发布时间和摘要"],
                ),
                ResearchQuestion(
                    question_id="institution_views",
                    objective="机构观点有哪些共识、分歧和时效限制？",
                    evidence_requirements=["研报机构、日期、评级或摘要"],
                ),
            ]
        )
    if "finance_rag" in sources:
        questions.append(
            ResearchQuestion(
                question_id="financial_facts",
                objective="财务事实是否支持研究假设和机构观点？",
                evidence_requirements=["财务指标、报告期和数据口径"],
            )
        )
    questions.append(
        ResearchQuestion(
            question_id="critical_review",
            objective="现有证据支持什么结论，反方证据和未解决缺口是什么？",
            evidence_requirements=["支持证据、反方证据和数据缺口"],
        )
    )
    return questions


def build_research_plan(
    query: str,
    task_input: Mapping[str, Any] | None = None,
) -> ResearchPlan:
    """根据 Root 传入的研究范围提示生成内部研究计划。"""
    inputs = task_input or {}
    data_sources = [
        item
        for item in _string_list(inputs.get("data_sources"))
        if item in _ALLOWED_DATA_SOURCES
    ]
    if not data_sources:
        data_sources = list(RESEARCH_WORKFLOW_SPEC.default_data_sources)
    entities = _string_list(inputs.get("entities"))
    return ResearchPlan(
        query=query,
        entities=entities,
        data_sources=data_sources,
        questions=_research_questions(query, data_sources),
        source_tasks=_source_tasks(query, data_sources),
        rationale="根据请求画像提供的研究范围，生成内部问题清单和来源任务。",
        metadata={"planner": "deterministic"},
    )


def _validated_llm_plan(
    query: str,
    task_input: Mapping[str, Any],
    draft: ResearchPlanDraft,
) -> ResearchPlan:
    """将 LLM 候选计划收敛到 Root 授权范围和确定性任务白名单。"""
    hinted_sources = _string_list(task_input.get("data_sources"))
    allowed_scope = [
        item
        for item in (
            hinted_sources or list(RESEARCH_WORKFLOW_SPEC.default_data_sources)
        )
        if item in _ALLOWED_DATA_SOURCES
    ]
    if not allowed_scope:
        allowed_scope = list(RESEARCH_WORKFLOW_SPEC.default_data_sources)
    allowed_set = set(allowed_scope)

    issues: list[str] = []
    requested_sources = _string_list(draft.data_sources)
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
    if not data_sources:
        data_sources = allowed_scope
        issues.append("llm_data_sources_empty_use_allowed_scope")

    questions: list[ResearchQuestion] = []
    seen_question_ids: set[str] = set()
    for item in draft.questions[:_MAX_RESEARCH_QUESTIONS]:
        question_id = item.question_id.strip()
        objective = item.objective.strip()
        if not question_id or not objective or question_id in seen_question_ids:
            issues.append(f"invalid_or_duplicate_question:{question_id}")
            continue
        seen_question_ids.add(question_id)
        questions.append(
            item.model_copy(
                update={
                    "question_id": question_id,
                    "objective": objective,
                    "evidence_requirements": _string_list(
                        item.evidence_requirements
                    )[:8],
                }
            )
        )
    if not questions:
        raise ValueError("research_plan_questions_empty")
    if "critical_review" not in seen_question_ids:
        questions = questions[: _MAX_RESEARCH_QUESTIONS - 1]
        questions.append(
            ResearchQuestion(
                question_id="critical_review",
                objective="现有证据支持什么结论，反方证据和未解决缺口是什么？",
                evidence_requirements=["支持证据、反方证据和数据缺口"],
            )
        )
        issues.append("critical_review_added")
    questions = questions[:_MAX_RESEARCH_QUESTIONS]

    return ResearchPlan(
        query=query,
        entities=_string_list(task_input.get("entities")),
        data_sources=data_sources,
        questions=questions,
        source_tasks=_source_tasks(query, data_sources),
        rationale=draft.rationale.strip(),
        metadata={
            "planner": "llm_validated",
            "allowed_source_scope": allowed_scope,
            "validation_issues": issues,
        },
    )


async def plan_research_adaptive(
    query: str,
    task_input: Mapping[str, Any] | None = None,
    *,
    config: RunnableConfig | None = None,
    llm: BaseChatModel | None = None,
) -> ResearchPlan:
    """单次 LLM 规划；异常或校验失败时回退确定性研究计划。"""
    inputs = task_input or {}
    allowed_scope = [
        item
        for item in _string_list(inputs.get("data_sources"))
        if item in _ALLOWED_DATA_SOURCES
    ]
    if not allowed_scope:
        allowed_scope = list(RESEARCH_WORKFLOW_SPEC.default_data_sources)
    entities = _string_list(inputs.get("entities"))
    human_prompt = (
        f"用户研究问题：\n{query}\n\n"
        f"允许的数据范围：{', '.join(allowed_scope)}\n"
        f"已识别实体：{', '.join(entities) if entities else '无'}"
    )
    try:
        model = llm or get_router_llm()
        raw = await model.with_structured_output(
            ResearchPlanDraft,
            method="json_mode",
        ).ainvoke(
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
        return _validated_llm_plan(query, inputs, draft)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "research planner llm failed, fallback deterministic: {}",
            type(exc).__name__,
        )
        fallback = build_research_plan(query, inputs)
        return fallback.model_copy(
            update={
                "metadata": {
                    **fallback.metadata,
                    "planner": "deterministic_fallback",
                    "fallback_reason": type(exc).__name__,
                }
            }
        )


__all__ = ["build_research_plan", "plan_research_adaptive"]
