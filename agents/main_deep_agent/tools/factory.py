"""Main DeepAgent 的受治理工具组装。"""

from __future__ import annotations

import time
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
from agents.main_deep_agent.tools.executor import execute_registered_tool
from agents.runtime_context import AgentRuntimeContext
from app.core.config import settings
from harness.context import build_run_context
from tools import get_langchain_tool, get_tool_spec, load_all_tools


_FALLBACK_TOOL_IDS = {
    "financial": (
        "iwencai.query",
        "iwencai.market.query",
        "web.search",
        "knowledge.pdf.catalog",
    ),
    "knowledge": (
        "iwencai.query",
        "iwencai.report.search",
        "iwencai.announcement.search",
        "web.search",
    ),
}


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
) -> None:
    """仅发送可观察状态，不发送参数、结果或模型推理。"""
    try:
        writer = get_stream_writer()
        writer(
            {
                "kind": "main_tool_progress",
                "step_id": step_id,
                "tool_id": tool_id,
                "source_family": family,
                "status": status,
            }
        )
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
    wrapped: list[BaseTool] = []
    for tool_id in resolved_tool_ids:
        base = get_langchain_tool(tool_id)

        async def _ainvoke(_tool_id: str = tool_id, **kwargs: Any) -> dict[str, Any]:
            family = TOOL_SOURCE_FAMILY.get(_tool_id, "unknown")
            started = time.perf_counter()
            query = str(kwargs.get("query") or "").strip()
            entity = str(kwargs.get("entity") or "").strip()
            if not entity and isinstance(kwargs.get("entities"), list):
                entity = ",".join(str(item).strip() for item in kwargs["entities"] if str(item).strip())
            purpose = str(kwargs.get("purpose") or "").strip()
            expected_fields = [
                str(item) for item in list(kwargs.get("expected_fields") or [])
            ]
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

            budget.register(_tool_id, entity=entity)
            step_id = f"main-tool-{budget.tool_calls}-{_tool_id}"
            _emit_tool_progress(
                step_id=step_id,
                tool_id=_tool_id,
                family=family,
                status="running",
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
                web_evidence_from_payload(data, entity=entity)
                if ok and _tool_id == "web.search"
                else _evidence_from_payload(_tool_id, data)
                if ok
                else []
            )
            if ok and _tool_id == "web.search" and not evidence:
                ok = False
                error = "insufficient_tool_result"
            if ok and _tool_id == "knowledge.pdf.catalog" and isinstance(data, Mapping):
                documents = list(data.get("documents") or [])
                discovered_doc_ids = {
                    str(item.get("doc_id") or "").strip()
                    for item in documents
                    if isinstance(item, Mapping) and str(item.get("doc_id") or "").strip()
                }
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
            _emit_tool_progress(
                step_id=step_id,
                tool_id=_tool_id,
                family=family,
                status="done" if ok else "error",
            )
            if not ok:
                failure = _failure_payload(
                    tool_id=_tool_id,
                    error=error or "tool_execution_failed",
                    allowed_tool_ids=allowed,
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
