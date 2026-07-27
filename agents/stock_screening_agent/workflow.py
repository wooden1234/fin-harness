"""普通 LangGraph 驱动的选股工作流。"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from typing import Any, Literal, TypedDict, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from pydantic import BaseModel

from agents.iwencai_source_runtime import (
    normalize_screening_result,
    run_iwencai_source_tool,
)
from agents.llm import get_router_llm
from agents.orchestrator.contracts import (
    AgentResult,
    CandidateSet,
    MarketQueryPlan,
)
from agents.runtime_context import AgentRuntimeContext
from agents.stock_screening_agent.skill_binding import (
    SkillBinding,
    resolve_skill_binding,
)
from agents.stock_screening_agent.spec import (
    STOCK_SCREENING_SPEC,
    StockScreeningSpec,
)


class ScreeningPlanDraft(BaseModel):
    """LLM 生成的结构化选股计划。"""

    plan: MarketQueryPlan
    adjustment: str = ""


class StockScreeningWorkflowState(TypedDict, total=False):
    """选股工作流的内部状态。"""

    original_query: str
    current_query: str
    adjustment: str
    task_input: dict[str, Any]
    query_history: list[str]
    retry_count: int
    retry_planning_failed: bool
    parse_errors: list[str]
    parsed_plan: MarketQueryPlan
    previous_plan: MarketQueryPlan
    validation_errors: list[str]
    validation_failed: bool
    tool_result: AgentResult
    should_retry: bool
    final_result: AgentResult


def _skill_prompt(binding: SkillBinding) -> str:
    return "\n\n".join(document.instructions for document in binding.documents)


_SCREENING_PLAN_SYSTEM_PROMPT = """你负责把 A 股自然语言选股要求转换为 MarketQueryPlan。

要求：
1. 只输出结构化计划，不生成问财 query，不回答选股结果。
2. universe 只能根据用户要求填写 A股、沪深京或沪深A股。
3. 用户明确给出的行业、概念、指标、阈值、排序和数量必须保留。
4. 不得自行补充用户未给出的阈值、行业、股票或投资结论。
5. 行业和概念分别使用 field=industry、field=concept 和 operator=contains。
6. 常用字段优先使用 pe_ttm、pb、roe、revenue_growth、net_profit_growth、
   market_cap、price_change_pct、volume_ratio。
7. “前 N 只”写入 limit；“最高/最低”同时写入 sort。

示例一：
用户：筛选新能源行业，市盈率低于30，ROE大于15%，取前10只
计划：
{"universe":"A股","filters":[
{"field":"industry","operator":"contains","value":"新能源"},
{"field":"pe_ttm","operator":"lt","value":30},
{"field":"roe","operator":"gt","value":15}
],"sort":[],"limit":10}

示例二：
用户：筛选净利润增长率最高的5只半导体股票
计划：
{"universe":"A股","filters":[
{"field":"industry","operator":"contains","value":"半导体"}
],"sort":[{"field":"net_profit_growth","direction":"desc"}],"limit":5}
"""

_SCREENING_RETRY_SYSTEM_PROMPT = """你负责调整一个未返回候选股票的 A 股选股计划。

要求：
1. 只输出新的 MarketQueryPlan，不生成问财 query。
2. 保留 universe、行业、概念以及用户明确指定的硬条件。
3. 最多删除或放宽一个非核心过滤条件。
4. 不得增加新的过滤条件、行业、股票或阈值。
5. adjustment 必须准确说明删除或放宽了哪个条件。

示例：
原计划包含“新能源、PE<20、ROE>20”，结果为空；
可以调整为“新能源、PE<30、ROE>20”，并说明“将PE上限从20放宽到30”。
"""


async def _invoke_screening_plan(
    *,
    system_prompt: str,
    human_prompt: str,
    binding: SkillBinding,
    config: RunnableConfig | None,
    llm: BaseChatModel | None,
) -> ScreeningPlanDraft:
    """调用 LLM 生成纯结构化计划。"""
    model = llm or get_router_llm()
    return cast(
        ScreeningPlanDraft,
        await model.with_structured_output(
            ScreeningPlanDraft,
            method="json_mode",
        ).ainvoke(
            [
                (
                    "system",
                    f"{system_prompt}\n\n当前 Skill 约束：\n{_skill_prompt(binding)}",
                ),
                ("human", human_prompt),
            ],
            config=config,
        ),
    )


async def parse_screening_plan(
    *,
    original_query: str,
    binding: SkillBinding,
    config: RunnableConfig | None,
    llm: BaseChatModel | None,
) -> ScreeningPlanDraft:
    """首次解析用户选股要求。"""
    return await _invoke_screening_plan(
        system_prompt=_SCREENING_PLAN_SYSTEM_PROMPT,
        human_prompt=f"用户原始选股要求：\n{original_query}",
        binding=binding,
        config=config,
        llm=llm,
    )


async def relax_screening_plan(
    *,
    original_query: str,
    previous_plan: MarketQueryPlan,
    previous_result: AgentResult,
    binding: SkillBinding,
    config: RunnableConfig | None,
    llm: BaseChatModel | None,
) -> ScreeningPlanDraft:
    """空结果时生成一次受限的放宽计划。"""
    return await _invoke_screening_plan(
        system_prompt=_SCREENING_RETRY_SYSTEM_PROMPT,
        human_prompt=(
            f"用户原始要求：\n{original_query}\n\n"
            f"上一次结构化计划：\n{previous_plan.model_dump_json()}\n\n"
            f"上一次结果：status={previous_result.status}, "
            f"error_code={previous_result.error_code}, "
            f"gaps={previous_result.gaps}"
        ),
        binding=binding,
        config=config,
        llm=llm,
    )


_FIELD_NAME_RE = re.compile(
    r"^(?:[A-Za-z][A-Za-z0-9_.]{0,63}|[\u4e00-\u9fff]{1,32})$"
)
_FIELD_LABELS = {
    "industry": "行业",
    "concept": "概念",
    "board": "板块",
    "symbol": "股票代码",
    "name": "股票名称",
    "pe_ttm": "市盈率TTM",
    "pb": "市净率",
    "roe": "ROE",
    "revenue_growth": "营收增长率",
    "net_profit_growth": "净利润增长率",
    "market_cap": "总市值",
    "price_change_pct": "涨跌幅",
    "volume_ratio": "量比",
}
_OPERATOR_LABELS = {
    "eq": "等于",
    "ne": "不等于",
    "gt": "大于",
    "gte": "大于等于",
    "lt": "小于",
    "lte": "小于等于",
    "contains": "包含",
}


def _render_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "、".join(str(item) for item in value)
    return str(value)


def _render_filter(field: str, operator: str, value: Any) -> str:
    label = _FIELD_LABELS.get(field, field)
    if operator == "between":
        return f"{label}介于{value[0]}和{value[1]}之间"
    if operator == "in":
        return f"{label}属于{_render_value(value)}"
    if operator == "not_in":
        return f"{label}不属于{_render_value(value)}"
    return f"{label}{_OPERATOR_LABELS[operator]}{_render_value(value)}"


def build_iwencai_query(plan: MarketQueryPlan) -> str:
    """根据已校验计划确定性生成问财自然语言查询。"""
    parts = [plan.universe]
    parts.extend(
        _render_filter(item.field, item.operator, item.value)
        for item in plan.filters
    )
    if plan.metrics:
        parts.append(f"返回指标：{'、'.join(plan.metrics)}")
    if plan.enrichments:
        parts.append(f"补充字段：{'、'.join(plan.enrichments)}")
    for item in plan.sort:
        direction = "从高到低" if item.direction == "desc" else "从低到高"
        parts.append(f"按{_FIELD_LABELS.get(item.field, item.field)}{direction}排序")
    if plan.as_of:
        parts.append(f"数据日期{plan.as_of}")
    parts.append(f"取前{plan.limit}只")
    return "，".join(parts)


def validate_screening_plan(
    plan: MarketQueryPlan,
) -> list[str]:
    """确定性校验结构化选股计划，避免非法计划进入 Tool。"""
    errors: list[str] = []
    if plan.universe not in {"A股", "沪深京", "沪深A股"}:
        errors.append(f"universe_not_allowed:{plan.universe}")
    if plan.limit > 100:
        errors.append("limit_exceeds_screening_maximum:100")
    if len(plan.filters) > 50:
        errors.append("too_many_filters")
    if len(plan.sort) > 10:
        errors.append("too_many_sort_fields")
    for index, condition in enumerate(plan.filters):
        if not _FIELD_NAME_RE.fullmatch(condition.field):
            errors.append(f"filter_field_invalid:{index}")
        if condition.value is None:
            errors.append(f"filter_value_missing:{index}")
        if condition.operator in {"in", "not_in"} and not isinstance(
            condition.value, (list, tuple)
        ):
            errors.append(f"filter_value_must_be_list:{index}")
        if condition.operator == "between" and (
            not isinstance(condition.value, (list, tuple))
            or len(condition.value) != 2
        ):
            errors.append(f"filter_between_requires_two_values:{index}")
        if condition.operator == "contains" and not isinstance(
            condition.value, str
        ):
            errors.append(f"filter_contains_requires_text:{index}")
    for index, item in enumerate(plan.sort):
        if not _FIELD_NAME_RE.fullmatch(item.field):
            errors.append(f"sort_field_invalid:{index}")
    return errors


def validate_retry_plan(
    previous: MarketQueryPlan,
    current: MarketQueryPlan,
) -> list[str]:
    """校验重试只能放宽一个条件，且不能改变核心范围。"""
    errors: list[str] = []
    if current.universe != previous.universe:
        errors.append("retry_universe_changed")
    if current.sort != previous.sort:
        errors.append("retry_sort_changed")
    if current.limit != previous.limit:
        errors.append("retry_limit_changed")
    if current.metrics != previous.metrics:
        errors.append("retry_metrics_changed")
    if current.enrichments != previous.enrichments:
        errors.append("retry_enrichments_changed")
    if current.group_by != previous.group_by:
        errors.append("retry_group_by_changed")
    if current.as_of != previous.as_of:
        errors.append("retry_as_of_changed")

    previous_fields = Counter(item.field for item in previous.filters)
    current_fields = Counter(item.field for item in current.filters)
    if current_fields - previous_fields:
        errors.append("retry_added_filter_field")

    previous_signatures = Counter(
        item.model_dump_json() for item in previous.filters
    )
    current_signatures = Counter(
        item.model_dump_json() for item in current.filters
    )
    removed = sum((previous_signatures - current_signatures).values())
    added = sum((current_signatures - previous_signatures).values())
    if removed == 0 and added == 0:
        errors.append("retry_plan_unchanged")
    if removed > 1 or added > 1:
        errors.append("retry_changed_more_than_one_filter")

    for hard_field in ("industry", "concept"):
        before = [
            item.model_dump()
            for item in previous.filters
            if item.field == hard_field
        ]
        after = [
            item.model_dump()
            for item in current.filters
            if item.field == hard_field
        ]
        if before != after:
            errors.append(f"retry_hard_filter_changed:{hard_field}")
    return errors


def _skill_for_tool(binding: SkillBinding, tool_id: str) -> str:
    for document in binding.documents:
        if tool_id in document.tool_ids:
            return document.name
    raise ValueError(f"skill_tool_binding_missing:{tool_id}")


def _is_empty_result(result: AgentResult) -> bool:
    return (
        result.status == "uncovered"
        and result.error_code.endswith("_empty_result")
    )


def _final_answer(result: AgentResult) -> str:
    if result.status != "completed":
        return result.answer
    candidates = CandidateSet.model_validate(result.structured_data)
    return f"已完成问财选股，共找到 {len(candidates.rows)} 只候选股票。"


def build_stock_screening_workflow(
    *,
    config: RunnableConfig | None = None,
    runtime: Runtime[AgentRuntimeContext] | None = None,
    llm: BaseChatModel | None = None,
    spec: StockScreeningSpec = STOCK_SCREENING_SPEC,
):
    """构建 parse、validate、execute、evaluate、finalize 工作流。"""
    binding = resolve_skill_binding(spec.skills)
    required_tool = binding.primary_required_tool
    skill_name = _skill_for_tool(binding, required_tool)

    async def parse_node(
        state: StockScreeningWorkflowState,
    ) -> StockScreeningWorkflowState:
        original_query = state["original_query"]
        retry_count = int(state.get("retry_count") or 0)
        parse_errors = list(state.get("parse_errors") or [])
        try:
            previous_plan = state.get("parsed_plan")
            previous_result = state.get("tool_result")
            if retry_count:
                if previous_plan is None or previous_result is None:
                    raise ValueError("retry_context_missing")
                draft = await relax_screening_plan(
                    original_query=original_query,
                    previous_plan=previous_plan,
                    previous_result=previous_result,
                    binding=binding,
                    config=config,
                    llm=llm,
                )
            else:
                draft = await parse_screening_plan(
                    original_query=original_query,
                    binding=binding,
                    config=config,
                    llm=llm,
                )
            updates: StockScreeningWorkflowState = {
                "adjustment": draft.adjustment.strip(),
                "parsed_plan": draft.plan,
                "retry_planning_failed": False,
                "parse_errors": parse_errors,
            }
            if retry_count and previous_plan is not None:
                updates["previous_plan"] = previous_plan
            return updates
        except Exception as exc:  # noqa: BLE001
            parse_errors.append(f"{type(exc).__name__}:{exc}")
            return {
                "adjustment": "",
                "retry_planning_failed": True,
                "parse_errors": parse_errors,
                "validation_failed": True,
                "validation_errors": ["screening_plan_parse_failed"],
            }

    async def validate_node(
        state: StockScreeningWorkflowState,
    ) -> StockScreeningWorkflowState:
        if state.get("retry_planning_failed"):
            return {
                "validation_failed": True,
                "validation_errors": ["screening_plan_parse_failed"],
            }
        plan = state.get("parsed_plan")
        if plan is None:
            return {
                "validation_failed": True,
                "validation_errors": ["screening_plan_missing"],
            }
        errors = validate_screening_plan(plan)
        previous_plan = state.get("previous_plan")
        if int(state.get("retry_count") or 0) and previous_plan is not None:
            errors.extend(validate_retry_plan(previous_plan, plan))
        return {
            "current_query": build_iwencai_query(plan) if not errors else "",
            "validation_failed": bool(errors),
            "validation_errors": errors,
        }

    async def execute_node(
        state: StockScreeningWorkflowState,
    ) -> StockScreeningWorkflowState:
        if state.get("validation_failed") or state.get("retry_planning_failed"):
            return {}
        query = state.get("current_query") or state["original_query"]
        task_input = dict(state.get("task_input") or {})
        task_input["call_type"] = (
            "retry" if int(state.get("retry_count") or 0) else "normal"
        )
        result = await run_iwencai_source_tool(
            tool_id=required_tool,
            skill_name=skill_name,
            query=query,
            task_input=task_input,
            runtime=runtime,
            agent_id=spec.agent_id,
            task_id=spec.default_task_id,
        )
        return {
            "tool_result": result,
            "query_history": [*state.get("query_history", []), query],
        }

    async def evaluate_node(
        state: StockScreeningWorkflowState,
    ) -> StockScreeningWorkflowState:
        result = state.get("tool_result")
        retry_count = int(state.get("retry_count") or 0)
        should_retry = bool(
            result is not None
            and _is_empty_result(result)
            and retry_count < spec.max_retries
            and not state.get("retry_planning_failed")
        )
        return {
            "should_retry": should_retry,
            "retry_count": retry_count + 1 if should_retry else retry_count,
        }

    async def finalize_node(
        state: StockScreeningWorkflowState,
    ) -> StockScreeningWorkflowState:
        result = state.get("tool_result")
        if result is None:
            result = AgentResult(
                task_id=spec.default_task_id,
                agent_id=spec.agent_id,
                status="clarify" if state.get("validation_failed") else "failed",
                answer=(
                    "选股条件无法通过结构化校验，请补充明确的 A 股范围、指标或排序条件。"
                    if state.get("validation_failed")
                    else spec.busy_answer
                ),
                error_code=(
                    "screening_plan_invalid"
                    if state.get("validation_failed")
                    else "screening_result_missing"
                ),
                gaps=(
                    list(state.get("validation_errors") or [])
                    if state.get("validation_failed")
                    else ["选股工作流未产生工具结果"]
                ),
            )
        normalized = normalize_screening_result(
            result,
            query=state.get("current_query") or state["original_query"],
            agent_id=spec.agent_id,
            task_id=spec.default_task_id,
            query_plan=state.get("parsed_plan"),
        )
        metadata = {
            **normalized.metadata,
            "runtime": "workflow",
            "original_query": state["original_query"],
            "query_history": list(state.get("query_history") or []),
            "retry_count": int(state.get("retry_count") or 0),
            "adjustment": state.get("adjustment") or "",
            "parse_errors": list(state.get("parse_errors") or []),
            "validation_errors": list(state.get("validation_errors") or []),
            "parsed_plan": (
                state["parsed_plan"].model_dump()
                if state.get("parsed_plan") is not None
                else None
            ),
        }
        return {
            "final_result": normalized.model_copy(
                update={
                    "answer": _final_answer(normalized),
                    "metadata": metadata,
                }
            )
        }

    async def route_after_evaluate(
        state: StockScreeningWorkflowState,
    ) -> Literal["retry", "finalize"]:
        return "retry" if state.get("should_retry") else "finalize"

    builder = StateGraph(StockScreeningWorkflowState)
    builder.add_node("parse", parse_node)
    builder.add_node("validate", validate_node)
    builder.add_node("execute", execute_node)
    builder.add_node("evaluate", evaluate_node)
    builder.add_node("finalize", finalize_node)
    builder.add_edge(START, "parse")
    builder.add_edge("parse", "validate")
    builder.add_edge("validate", "execute")
    builder.add_edge("execute", "evaluate")
    builder.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {"retry": "parse", "finalize": "finalize"},
    )
    builder.add_edge("finalize", END)
    return builder.compile()


async def run_stock_screening_workflow(
    state: Mapping[str, Any],
    *,
    query: str,
    config: RunnableConfig | None = None,
    runtime: Runtime[AgentRuntimeContext] | None = None,
    llm: BaseChatModel | None = None,
    spec: StockScreeningSpec = STOCK_SCREENING_SPEC,
) -> AgentResult:
    """执行普通选股工作流并返回统一结果。"""
    task_input = state.get("task_input")
    workflow = build_stock_screening_workflow(
        config=config,
        runtime=runtime,
        llm=llm,
        spec=spec,
    )
    output = await workflow.ainvoke(
        {
            "original_query": query,
            "current_query": "",
            "task_input": (
                dict(task_input) if isinstance(task_input, Mapping) else {}
            ),
            "query_history": [],
            "retry_count": 0,
            "retry_planning_failed": False,
            "parse_errors": [],
        },
        config=config,
    )
    result = output.get("final_result")
    if isinstance(result, AgentResult):
        return result
    return AgentResult.model_validate(result)


__all__ = [
    "ScreeningPlanDraft",
    "StockScreeningWorkflowState",
    "build_iwencai_query",
    "build_stock_screening_workflow",
    "parse_screening_plan",
    "relax_screening_plan",
    "run_stock_screening_workflow",
    "validate_retry_plan",
    "validate_screening_plan",
]
