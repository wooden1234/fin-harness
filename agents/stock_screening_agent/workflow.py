"""普通 LangGraph 驱动的选股工作流。"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, TypedDict, cast

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
from agents.structured_output import ainvoke_json_output
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


class StockScreeningWorkflowState(TypedDict, total=False):
    """选股工作流的内部状态。"""

    original_query: str
    current_query: str
    task_input: dict[str, Any]
    query_history: list[str]
    parse_errors: list[str]
    parsed_plan: MarketQueryPlan
    validation_errors: list[str]
    validation_failed: bool
    tool_result: AgentResult
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
8. 不支持分组聚合，禁止填写 group_by；如用户要求分组统计，请改用 sort/limit
   或在 rationale 之外的字段留空，不得编造。

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
        await ainvoke_json_output(
            model,
            ScreeningPlanDraft,
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
_MAX_METRICS = 20
_MAX_ENRICHMENTS = 20
_MAX_FREEFORM_FIELD_LENGTH = 64


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


def _validate_freeform_fields(
    values: list[str],
    *,
    label: str,
    max_length: int,
    errors: list[str],
) -> None:
    """校验 metrics/enrichments 等自由文本字段，避免非法或超长内容流入 Tool 查询。"""
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{label}_invalid:{index}")
        elif len(value) > max_length:
            errors.append(f"{label}_too_long:{index}")


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
    if plan.group_by:
        # 选股工作流和下游 market.compute 都不支持分组聚合，
        # 与其静默丢弃 group_by 不如直接判定计划不可执行。
        errors.append("group_by_not_supported_by_screening")
    if len(plan.metrics) > _MAX_METRICS:
        errors.append("too_many_metrics")
    if len(plan.enrichments) > _MAX_ENRICHMENTS:
        errors.append("too_many_enrichments")
    _validate_freeform_fields(
        plan.metrics,
        label="metric",
        max_length=_MAX_FREEFORM_FIELD_LENGTH,
        errors=errors,
    )
    _validate_freeform_fields(
        plan.enrichments,
        label="enrichment",
        max_length=_MAX_FREEFORM_FIELD_LENGTH,
        errors=errors,
    )
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


def _skill_for_tool(binding: SkillBinding, tool_id: str) -> str:
    for document in binding.documents:
        if tool_id in document.tool_ids:
            return document.name
    raise ValueError(f"skill_tool_binding_missing:{tool_id}")


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
    """构建 parse、validate、execute、finalize 工作流。"""
    binding = resolve_skill_binding(spec.skills)
    required_tool = binding.primary_required_tool
    skill_name = _skill_for_tool(binding, required_tool)

    async def parse_node(
        state: StockScreeningWorkflowState,
    ) -> StockScreeningWorkflowState:
        original_query = state["original_query"]
        parse_errors = list(state.get("parse_errors") or [])
        try:
            draft = await parse_screening_plan(
                original_query=original_query,
                binding=binding,
                config=config,
                llm=llm,
            )
            return {
                "parsed_plan": draft.plan,
                "parse_errors": parse_errors,
            }
        except Exception as exc:  # noqa: BLE001
            parse_errors.append(f"{type(exc).__name__}:{exc}")
            return {
                "parse_errors": parse_errors,
                "validation_failed": True,
                "validation_errors": ["screening_plan_parse_failed"],
            }

    async def validate_node(
        state: StockScreeningWorkflowState,
    ) -> StockScreeningWorkflowState:
        if state.get("validation_failed"):
            # parse_node 已经记录了更具体的失败原因（如 LLM 调用异常），
            # 这里直接透传，避免被下面通用的 screening_plan_missing 覆盖掉。
            return {}
        plan = state.get("parsed_plan")
        if plan is None:
            return {
                "validation_failed": True,
                "validation_errors": ["screening_plan_missing"],
            }
        errors = validate_screening_plan(plan)
        return {
            "current_query": build_iwencai_query(plan) if not errors else "",
            "validation_failed": bool(errors),
            "validation_errors": errors,
        }

    async def execute_node(
        state: StockScreeningWorkflowState,
    ) -> StockScreeningWorkflowState:
        if state.get("validation_failed"):
            return {}
        query = state.get("current_query") or state["original_query"]
        task_input = dict(state.get("task_input") or {})
        task_input["call_type"] = "normal"
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

    async def finalize_node(
        state: StockScreeningWorkflowState,
    ) -> StockScreeningWorkflowState:
        result = state.get("tool_result")
        if result is None:
            if state.get("validation_failed"):
                # parse_node/validate_node 现在都不会覆盖彼此的失败原因，
                # 这里把两边记录的具体原因合并进 gaps，方便排查到底是
                # LLM 解析异常（screening_plan_parse_failed）还是结构化
                # 计划本身不合规（如 universe_not_allowed 等），同时对外
                # 仍统一保持 clarify 语义，不因原因不同而改变响应契约。
                gaps = list(
                    dict.fromkeys(
                        [
                            *(state.get("validation_errors") or []),
                            *(state.get("parse_errors") or []),
                        ]
                    )
                )
                result = AgentResult(
                    task_id=spec.default_task_id,
                    agent_id=spec.agent_id,
                    status="clarify",
                    answer="选股条件无法通过结构化校验，请补充明确的 A 股范围、指标或排序条件。",
                    error_code="screening_plan_invalid",
                    gaps=gaps,
                )
            else:
                result = AgentResult(
                    task_id=spec.default_task_id,
                    agent_id=spec.agent_id,
                    status="failed",
                    answer=spec.busy_answer,
                    error_code="screening_result_missing",
                    gaps=["选股工作流未产生工具结果"],
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
            "retry_count": 0,
            "adjustment": "",
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

    builder = StateGraph(StockScreeningWorkflowState)
    builder.add_node("parse", parse_node)
    builder.add_node("validate", validate_node)
    builder.add_node("execute", execute_node)
    builder.add_node("finalize", finalize_node)
    builder.add_edge(START, "parse")
    builder.add_edge("parse", "validate")
    builder.add_edge("validate", "execute")
    builder.add_edge("execute", "finalize")
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
    "run_stock_screening_workflow",
    "validate_screening_plan",
]
