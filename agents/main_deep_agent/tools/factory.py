"""Main DeepAgent 的受治理工具组装。"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from collections.abc import Mapping
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from langgraph.config import get_stream_writer

from agents.main_deep_agent.middleware.authorization import (
    permissions_allow as _permissions_allow,
)
from agents.main_deep_agent.middleware.budget import (
    TOOL_SOURCE_FAMILY,
    MainAgentBudgetController,
)
from agents.main_deep_agent.state import MainAgentProgressJournal
from agents.main_deep_agent.tools.catalog import (
    MAIN_TOOL_ARGS_SCHEMAS,
    MAIN_TOOL_IDS,
)
from agents.main_deep_agent.tools.evidence_adapter import (
    _compact_evidence_for_model,
    _evidence_from_payload,
    _sanitize_tool_payload,
    web_evidence_from_payload,
)
from agents.main_deep_agent.tools.step_detail import build_public_tool_step_detail
from agents.main_deep_agent.tools.executor import execute_registered_tool
from agents.runtime_context import AgentRuntimeContext
from app.core.config import settings
from harness.context import build_run_context
from tools import get_langchain_tool, get_tool_spec, load_all_tools


_FALLBACK_TOOL_IDS = {
    "financial": (
        "iwencai.finance.query",
        "iwencai.query",
        "iwencai.market.query",
        "web.search",
        "knowledge.pdf.catalog",
    ),
    "knowledge": (
        "iwencai.finance.query",
        "iwencai.query",
        "iwencai.report.search",
        "iwencai.announcement.search",
        "web.search",
    ),
    "market": (
        "iwencai.announcement.search",
        "iwencai.report.search",
        "web.search",
        "knowledge.pdf.catalog",
    ),
}

_TIME_COMPARISON_OPERATIONS = frozenset({
    "change_rate",
    "drawdown",
    "year_over_year",
    "quarter_over_quarter",
    "cagr",
})


def _as_mapping(value: object) -> Mapping[str, Any] | None:
    """兼容 StructuredTool 将嵌套参数解析成 Pydantic 对象的情况。"""
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="python")
        return dumped if isinstance(dumped, Mapping) else None
    return None


def _fact_from_ref(
    journal: MainAgentProgressJournal,
    raw_ref: object,
) -> tuple[dict[str, Any] | None, str]:
    """解析 Evidence 事实引用；不从文本或相同数值中猜测来源。"""
    ref = _as_mapping(raw_ref)
    if ref is None:
        return None, "calculation_fact_reference_invalid"
    evidence_id = str(ref.get("evidence_id") or "").strip()
    try:
        fact_index = int(ref.get("fact_index"))
    except (TypeError, ValueError):
        return None, "calculation_fact_reference_invalid"
    evidence = journal.evidence.get(evidence_id)
    if evidence is None:
        return None, "calculation_evidence_not_found"
    facts = list(evidence.metadata.get("facts") or [])
    if fact_index < 0 or fact_index >= len(facts):
        return None, "calculation_fact_not_found"
    fact = facts[fact_index]
    if not isinstance(fact, Mapping):
        return None, "calculation_fact_not_found"
    try:
        value = float(fact.get("value"))
    except (TypeError, ValueError):
        return None, "calculation_fact_value_invalid"
    if not math.isfinite(value):
        return None, "calculation_fact_value_invalid"
    return {
        **dict(fact),
        "value": value,
        "evidence_id": evidence_id,
        "fact_index": fact_index,
    }, ""


def _calculation_scope_error(
    operation: str,
    current: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> str:
    """校验运算两端的实体和口径，避免跨对象误算。"""
    current_entity = str(current.get("entity") or "").strip()
    reference_entity = str(reference.get("entity") or "").strip()
    current_currency = str(current.get("currency") or "").strip()
    reference_currency = str(reference.get("currency") or "").strip()
    if current_currency and reference_currency and current_currency != reference_currency:
        return "calculation_currency_mismatch"
    if operation in _TIME_COMPARISON_OPERATIONS:
        if current_entity and reference_entity and current_entity != reference_entity:
            return "calculation_entity_mismatch"
        current_metric = str(current.get("metric") or "").strip()
        reference_metric = str(reference.get("metric") or "").strip()
        if current_metric and reference_metric and current_metric != reference_metric:
            return "calculation_metric_mismatch"
        current_unit = str(current.get("unit") or "").strip()
        reference_unit = str(reference.get("unit") or "").strip()
        if current_unit and reference_unit and current_unit != reference_unit:
            return "calculation_unit_mismatch"
    if operation == "difference":
        current_metric = str(current.get("metric") or "").strip()
        reference_metric = str(reference.get("metric") or "").strip()
        if current_metric and reference_metric and current_metric != reference_metric:
            return "calculation_metric_mismatch"
        current_unit = str(current.get("unit") or "").strip()
        reference_unit = str(reference.get("unit") or "").strip()
        if current_unit and reference_unit and current_unit != reference_unit:
            return "calculation_unit_mismatch"
        current_period = str(current.get("fiscal_period") or "").strip()
        reference_period = str(reference.get("fiscal_period") or "").strip()
        if (
            current_entity != reference_entity
            and current_period
            and reference_period
            and current_period != reference_period
        ):
            return "calculation_period_mismatch"
    if operation == "ratio":
        if current_entity and reference_entity and current_entity != reference_entity:
            return "calculation_entity_mismatch"
        current_period = str(current.get("fiscal_period") or "").strip()
        reference_period = str(reference.get("fiscal_period") or "").strip()
        if current_period and reference_period and current_period != reference_period:
            return "calculation_period_mismatch"
    return ""


def _resolve_calculation_arguments(
    raw_arguments: Mapping[str, Any],
    journal: MainAgentProgressJournal,
) -> tuple[dict[str, Any] | None, str]:
    """把批量事实引用解析成底层计算工具所需的受信数值。"""
    calculations = list(raw_arguments.get("calculations") or [])
    if not calculations or len(calculations) > 8:
        return None, "invalid_calculation_batch_size"
    resolved: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw_value in calculations:
        raw = _as_mapping(raw_value)
        if raw is None:
            return None, "invalid_calculation_arguments"
        calculation_id = str(raw.get("calculation_id") or "").strip()
        operation = str(raw.get("operation") or "").strip()
        if not calculation_id or calculation_id in seen_ids:
            return None, "duplicate_calculation_id"
        seen_ids.add(calculation_id)
        current, error = _fact_from_ref(journal, raw.get("current"))
        if error:
            return None, error
        reference, error = _fact_from_ref(journal, raw.get("reference"))
        if error:
            return None, error
        assert current is not None and reference is not None
        error = _calculation_scope_error(operation, current, reference)
        if error:
            return None, error
        resolved.append(
            {
                "calculation_id": calculation_id,
                "operation": operation,
                "current_value": current["value"],
                "reference_value": reference["value"],
                "periods": raw.get("periods", 1.0),
                "input_evidence_ids": list(dict.fromkeys([
                    current["evidence_id"],
                    reference["evidence_id"],
                ])),
                "operand_refs": [
                    {
                        "evidence_id": current["evidence_id"],
                        "fact_index": current["fact_index"],
                    },
                    {
                        "evidence_id": reference["evidence_id"],
                        "fact_index": reference["fact_index"],
                    },
                ],
            }
        )
    return {"calculations": resolved}, ""


def _failure_payload(
    *,
    tool_id: str,
    error: str,
    allowed_tool_ids: frozenset[str],
    stop_new_tools: bool = False,
    finalization_reason: str = "",
) -> dict[str, Any]:
    """把失败转成模型可执行的停止与降级信号。"""
    family = TOOL_SOURCE_FAMILY.get(tool_id, "unknown")
    terminal_same_tool = {
        "fact_not_in_local_store",
        "fact_query_requires_clarification",
        "fact_query_scope_unsupported",
        "local_document_not_found",
        "local_document_evidence_not_found",
        "calculation_fact_reference_invalid",
        "calculation_evidence_not_found",
        "calculation_fact_not_found",
        "calculation_fact_value_invalid",
        "calculation_entity_mismatch",
        "calculation_currency_mismatch",
        "calculation_metric_mismatch",
        "calculation_unit_mismatch",
        "calculation_period_mismatch",
        "invalid_calculation_batch_size",
        "invalid_calculation_arguments",
        "duplicate_calculation_id",
        "calculation_batch_failed",
        "duplicate_tool_call",
        "narrow_finance_call_exhausted",
    }
    prerequisite_errors = {"pdf_doc_ids_required", "pdf_doc_ids_not_cataloged"}
    payload: dict[str, Any] = {
        "ok": False,
        "error": error,
        "retryable": error not in terminal_same_tool
        and error not in prerequisite_errors
        and not error.startswith("tool_family_budget_exhausted:"),
    }
    if error in terminal_same_tool:
        payload["stop_same_tool"] = True
    if error.startswith("tool_family_budget_exhausted:"):
        payload["stop_same_family"] = True
    if error in prerequisite_errors:
        payload["requires_tool_id"] = "knowledge.pdf.catalog"
        payload["required_action"] = "先查询本地 PDF 目录，再使用本轮返回的 doc_ids。"
    fallback_ids = [
        candidate
        for candidate in _FALLBACK_TOOL_IDS.get(family, ())
        if candidate in allowed_tool_ids and candidate != tool_id
    ]
    if fallback_ids:
        payload["fallback_tool_ids"] = fallback_ids
    if stop_new_tools:
        payload.update(
            {
                "stop_new_tools": True,
                "finalization_reason": finalization_reason or error,
                "notice": (
                    "stop_new_tools=true，请基于已有 Evidence 输出 MainAgentResponse，"
                    "禁止继续调用工具。"
                ),
            }
        )
    return payload


def _emit_tool_progress(
    *,
    step_id: str,
    tool_id: str,
    family: str,
    status: str,
    detail: dict[str, Any] | None = None,
) -> None:
    """发送可观察状态；detail 仅含面向用户的展示子集。"""
    try:
        writer = get_stream_writer()
        payload: dict[str, Any] = {
            "kind": "main_tool_progress",
            "step_id": step_id,
            "tool_id": tool_id,
            "source_family": family,
            "status": status,
        }
        if detail:
            payload["detail"] = detail
        writer(payload)
    except (KeyError, RuntimeError):
        return


def resolve_main_tool_ids(context: AgentRuntimeContext) -> tuple[str, ...]:
    """根据运行时主体权限生成稳定工具目录，后续可直接接入租户能力表。"""
    return tuple(
        tool_id
        for tool_id in MAIN_TOOL_IDS
        if _permissions_allow(context, tool_id)
    )


def _governed_tools(
    *,
    context: AgentRuntimeContext,
    budget: MainAgentBudgetController,
    journal: MainAgentProgressJournal,
) -> list[BaseTool]:
    load_all_tools()
    run_context = build_run_context(
        user_id=context.user_id,
        tenant_id=context.tenant_id,
        conversation_id=context.conversation_id,
        permissions=context.permissions,
        metadata={"agent": "main_deep_agent", "run_id": context.run_id},
    )
    resolved_tool_ids = resolve_main_tool_ids(context)
    allowed = frozenset(resolved_tool_ids)
    cataloged_pdf_doc_ids: set[str] = set()
    cataloged_pdf_by_category: dict[str, set[str]] = {}
    seen_call_signatures: set[str] = set()
    failure_counts: Counter[tuple[str, str]] = Counter()
    wrapped: list[BaseTool] = []
    for tool_id in resolved_tool_ids:
        base = get_langchain_tool(tool_id)

        async def _ainvoke(_tool_id: str = tool_id, **kwargs: Any) -> dict[str, Any]:
            family = TOOL_SOURCE_FAMILY.get(_tool_id, "unknown")
            started = time.perf_counter()
            query = str(kwargs.get("query") or "").strip()
            entity = str(kwargs.get("entity") or "").strip()
            target_entities = [
                str(item).strip()
                for item in list(kwargs.get("entities") or [])
                if str(item).strip()
            ]
            if not entity and target_entities:
                entity = ",".join(target_entities)
            purpose = str(kwargs.get("purpose") or "").strip()
            expected_fields = [
                str(item) for item in list(kwargs.get("expected_fields") or [])
            ]
            call_signature = json.dumps(
                {"tool_id": _tool_id, "arguments": kwargs},
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            if call_signature in seen_call_signatures:
                error = "duplicate_tool_call"
                journal.record(
                    tool_id=_tool_id,
                    source_family=family,
                    status="rejected",
                    duration_ms=0.0,
                    evidence=[],
                    error=error,
                    entity=entity,
                    purpose=purpose,
                    expected_fields=expected_fields,
                )
                return _failure_payload(
                    tool_id=_tool_id,
                    error=error,
                    allowed_tool_ids=allowed,
                )
            seen_call_signatures.add(call_signature)
            if not _permissions_allow(context, _tool_id):
                error = "tool_not_authorized"
                budget.request_finalization(error)
                journal.record(
                    tool_id=_tool_id,
                    source_family=family,
                    status="rejected",
                    duration_ms=0.0,
                    evidence=[],
                    error=error,
                    entity=entity,
                    purpose=purpose,
                    expected_fields=expected_fields,
                )
                return {"ok": False, "error": error}
            authorized, error = budget.authorize(_tool_id, entity=entity)
            if not authorized:
                journal.record(
                    tool_id=_tool_id,
                    source_family=family,
                    status="rejected",
                    duration_ms=0.0,
                    evidence=[],
                    error=error,
                    entity=entity,
                    purpose=purpose,
                    expected_fields=expected_fields,
                )
                return _failure_payload(
                    tool_id=_tool_id,
                    error=error,
                    allowed_tool_ids=allowed,
                    stop_new_tools=budget.stop_new_tools,
                    finalization_reason=budget.finalization_reason,
                )
            if _tool_id == "knowledge.pdf.search":
                requested_doc_ids = {
                    str(item).strip()
                    for item in list(kwargs.get("doc_ids") or [])
                    if str(item).strip()
                }
                # 模型常漏传 doc_ids；本轮已 catalog 时按 categories 自动补齐。
                if not requested_doc_ids and cataloged_pdf_doc_ids:
                    categories = [
                        str(item).strip()
                        for item in list(kwargs.get("categories") or [])
                        if str(item).strip()
                    ]
                    selected: set[str] = set()
                    for category in categories:
                        selected.update(cataloged_pdf_by_category.get(category, set()))
                    if not selected:
                        selected = set(cataloged_pdf_doc_ids)
                    kwargs["doc_ids"] = sorted(selected)
                    requested_doc_ids = set(kwargs["doc_ids"])
                error = ""
                if not requested_doc_ids:
                    error = "pdf_doc_ids_required"
                elif not requested_doc_ids.issubset(cataloged_pdf_doc_ids):
                    error = "pdf_doc_ids_not_cataloged"
            else:
                error = ""
            if error:
                journal.record(
                    tool_id=_tool_id,
                    source_family=family,
                    status="rejected",
                    duration_ms=0.0,
                    evidence=[],
                    error=error,
                    entity=entity,
                    purpose=purpose,
                    expected_fields=expected_fields,
                )
                return _failure_payload(
                    tool_id=_tool_id,
                    error=error,
                    allowed_tool_ids=allowed,
                )

            if _tool_id == "calculation.run":
                resolved_arguments, error = _resolve_calculation_arguments(
                    kwargs,
                    journal,
                )
                if error:
                    journal.record(
                        tool_id=_tool_id,
                        source_family=family,
                        status="rejected",
                        duration_ms=0.0,
                        evidence=[],
                        error=error,
                        entity=entity,
                        purpose=purpose,
                        expected_fields=expected_fields,
                    )
                    return _failure_payload(
                        tool_id=_tool_id,
                        error=error,
                        allowed_tool_ids=allowed,
                    )
                assert resolved_arguments is not None
                kwargs = resolved_arguments

            budget.register(_tool_id, entity=entity)
            step_id = f"main-tool-{budget.tool_calls}-{_tool_id}"
            _emit_tool_progress(
                step_id=step_id,
                tool_id=_tool_id,
                family=family,
                status="running",
                detail=build_public_tool_step_detail(
                    tool_id=_tool_id,
                    source_family=family,
                    query=query,
                    entity=entity,
                    status="running",
                ),
            )
            if _tool_id in {"knowledge.faq.search", "knowledge.pdf.search"}:
                kwargs["research_question_id"] = "main-question"
            if _tool_id == "web.search":
                # 域名策略在 tools.web_search 内执行；主路径只传问句。
                kwargs = {"query": query}
            timeout_seconds = float(get_tool_spec(_tool_id).timeout_seconds)
            if budget.budget_tier == "deep_research":
                inflight_deadline = (
                    float(settings.MAIN_AGENT_RESEARCH_TOOL_CUTOFF_SEC)
                    + float(settings.MAIN_AGENT_INFLIGHT_GRACE_SEC)
                )
                timeout_seconds = min(
                    timeout_seconds,
                    max(0.05, inflight_deadline - budget.elapsed()),
                )
            try:
                result = await execute_registered_tool(
                    tool_id=_tool_id,
                    run_context=run_context,
                    arguments=dict(kwargs),
                    allowed_tool_ids=allowed,
                    timeout_seconds=timeout_seconds,
                )
                data = result.data
                payload_failed = isinstance(data, Mapping) and data.get("ok") is False
                ok = bool(result.ok) and not payload_failed
                error = str(result.error or (data.get("error") if payload_failed else "") or "")
            except TimeoutError:
                data = None
                ok = False
                error = "tool_timeout"
            evidence = (
                web_evidence_from_payload(
                    data,
                    entity=entity,
                    entities=target_entities,
                )
                if ok and _tool_id == "web.search"
                else _evidence_from_payload(_tool_id, data)
                if ok
                else []
            )
            if ok and _tool_id == "web.search" and not any(
                bool(item.metadata.get("displayable")) for item in evidence
            ):
                ok = False
                error = "web_no_relevant_results"
            if (
                ok
                and _tool_id in {"iwencai.query", "iwencai.finance.query"}
                and not evidence
            ):
                ok = False
                error = "iwencai_no_structured_data"
            if ok and _tool_id == "knowledge.pdf.catalog" and isinstance(data, Mapping):
                documents = list(data.get("documents") or [])
                discovered_doc_ids: set[str] = set()
                for item in documents:
                    if not isinstance(item, Mapping):
                        continue
                    doc_id = str(item.get("doc_id") or "").strip()
                    if not doc_id:
                        continue
                    discovered_doc_ids.add(doc_id)
                    category = str(item.get("category") or "").strip()
                    if category:
                        cataloged_pdf_by_category.setdefault(category, set()).add(doc_id)
                if discovered_doc_ids:
                    cataloged_pdf_doc_ids.update(discovered_doc_ids)
                else:
                    ok = False
                    error = "local_document_not_found"
            if ok and _tool_id == "knowledge.pdf.search" and not evidence:
                ok = False
                error = "local_document_evidence_not_found"
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            journal.record(
                tool_id=_tool_id,
                source_family=family,
                status="completed" if ok else "failed",
                duration_ms=duration_ms,
                evidence=evidence,
                error=error,
                entity=entity,
                purpose=purpose,
                expected_fields=expected_fields,
            )
            done_status = "done" if ok else "error"
            _emit_tool_progress(
                step_id=step_id,
                tool_id=_tool_id,
                family=family,
                status=done_status,
                detail=build_public_tool_step_detail(
                    tool_id=_tool_id,
                    source_family=family,
                    query=query,
                    entity=entity,
                    evidence=evidence,
                    status=done_status,
                    error=error,
                ),
            )
            if not ok:
                failure_counts[(_tool_id, error or "tool_execution_failed")] += 1
                failure = _failure_payload(
                    tool_id=_tool_id,
                    error=error or "tool_execution_failed",
                    allowed_tool_ids=allowed,
                )
                if failure_counts[(_tool_id, error or "tool_execution_failed")] >= 2:
                    failure.update(
                        {
                            "retryable": False,
                            "stop_same_tool": True,
                            "repeated_failure": error or "tool_execution_failed",
                        }
                    )
                if isinstance(data, Mapping):
                    for key in ("message", "route"):
                        if data.get(key):
                            failure[key] = str(data[key])
                return failure
            return {
                "ok": True,
                "data": _sanitize_tool_payload(
                    _tool_id,
                    data,
                    evidence=evidence,
                ),
                "evidence": _compact_evidence_for_model(
                    evidence,
                    tool_id=_tool_id,
                ),
            }

        wrapped.append(
            StructuredTool.from_function(
                coroutine=_ainvoke,
                name=base.name,
                description=base.description,
                args_schema=MAIN_TOOL_ARGS_SCHEMAS.get(
                    tool_id,
                    getattr(base, "args_schema", None),
                ),
            )
        )
    return wrapped


build_main_tools = _governed_tools

__all__ = ["build_main_tools", "resolve_main_tool_ids"]
