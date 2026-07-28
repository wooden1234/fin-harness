"""问财来源 Agent 共用的受治理调用与结果标准化。"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from langgraph.runtime import Runtime

from agents.orchestrator.contracts import (
    AgentResult,
    CandidateSet,
    DocumentHit,
    DocumentHitSet,
    Evidence,
    MarketQueryPlan,
)
from agents.runtime_context import (
    AgentRuntimeContext,
    RunHardDeadlineExceeded,
    RunSoftDeadlineExceeded,
)
from app.core.config import settings
from harness.context import build_run_context
from skills.loader import load_skill
from tools import execute_tool, load_all_tools, validate_tool_ids


def _rows_from_payload(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Mapping):
        return []
    for key in ("rows", "items", "results", "candidates"):
        rows = value.get(key)
        if isinstance(rows, list):
            return [dict(row) for row in rows if isinstance(row, Mapping)]
    for key in ("data", "result", "payload"):
        rows = _rows_from_payload(value.get(key))
        if rows:
            return rows
    return []


def normalize_screening_result(
    result: AgentResult,
    *,
    query: str,
    agent_id: str,
    task_id: str,
    query_plan: MarketQueryPlan | None = None,
) -> AgentResult:
    """把选股结果收敛为 CandidateSet。"""
    if result.status != "completed":
        return result.model_copy(update={"task_id": task_id, "agent_id": agent_id})

    rows = _rows_from_payload(result.structured_data)
    digest = hashlib.sha256(
        f"{query}:{result.structured_data}".encode("utf-8")
    ).hexdigest()[:12]
    as_of = str(
        result.structured_data.get("as_of")
        or result.structured_data.get("date")
        or datetime.now(timezone.utc).date().isoformat()
    )
    active_plan = (
        query_plan.model_copy(deep=True)
        if query_plan is not None
        else MarketQueryPlan(universe="A股")
    )
    active_plan = active_plan.model_copy(
        update={
            "metadata": {
                **active_plan.metadata,
                "natural_language_query": query,
            }
        }
    )
    candidates = CandidateSet(
        dataset_id=f"iwencai:{digest}",
        universe=active_plan.universe,
        provider="iwencai",
        as_of=as_of,
        rows=rows,
        query_plan=active_plan,
        evidence_ids=[item.evidence_id for item in result.evidence],
        metadata={"producer_id": result.agent_id},
    )
    return result.model_copy(
        update={
            "task_id": task_id,
            "agent_id": agent_id,
            "structured_data": candidates.model_dump(),
            "metadata": {**result.metadata, "mode": "acquire"},
        }
    )


def _record_list(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, Mapping)]
    if not isinstance(value, Mapping):
        return []
    if any(key in value for key in ("title", "标题", "report_title", "公告标题")):
        return [value]
    for key in ("documents", "data", "results", "items", "datas", "articles"):
        records = _record_list(value.get(key))
        if records:
            return records
    return []


def _has_records(value: Any) -> bool:
    """判断问财研究响应是否包含至少一条业务记录。"""
    if isinstance(value, (list, tuple)):
        return bool(value)
    if not isinstance(value, Mapping):
        return False
    for key in (
        "rows",
        "items",
        "results",
        "records",
        "documents",
        "articles",
        "data",
        "result",
        "payload",
    ):
        if key in value and _has_records(value[key]):
            return True
    return False


def _first_value(record: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None


def _normalize_document_set(
    tool_id: str,
    query: str,
    payload: Mapping[str, Any],
    evidence_id: str,
) -> DocumentHitSet:
    channel = "report" if tool_id == "iwencai.report.search" else "announcement"
    raw_data = payload.get("data", payload)
    records = _record_list(raw_data)
    documents: list[DocumentHit] = []
    for index, record in enumerate(records):
        title = str(_first_value(record, "title", "report_title", "公告标题", "标题") or "")
        url = _first_value(record, "url", "link", "链接")
        pdf_url = _first_value(record, "pdf_url", "pdfURL", "PDF链接", "研报链接")
        published_at = _first_value(
            record,
            "published_at",
            "publish_date",
            "公告日期",
            "发布时间",
            "日期",
        )
        documents.append(
            DocumentHit(
                document_id=str(
                    _first_value(
                        record,
                        "document_id",
                        "uid",
                        "id",
                        "report_id",
                        "seq",
                    )
                    or f"{channel}:{index}:{title}"
                ),
                title=title,
                summary=str(
                    _first_value(record, "summary", "摘要", "snippet", "content")
                    or ""
                ),
                published_at=str(published_at) if published_at is not None else None,
                organization=str(
                    _first_value(
                        record,
                        "organization",
                        "org",
                        "机构名称",
                        "券商",
                        "研究机构",
                    )
                    or ""
                ),
                rating=str(
                    _first_value(record, "rating", "评级", "投资评级", "机构评级")
                    or ""
                ),
                target_price=_first_value(record, "target_price", "目标价", "目标价格"),
                url=str(url) if url is not None else None,
                pdf_url=str(pdf_url) if pdf_url is not None else None,
                metadata={"source_fields": dict(record)},
            )
        )
    return DocumentHitSet(
        query=query,
        channel=channel,
        documents=documents,
        evidence_ids=[evidence_id],
        metadata={"provider": "iwencai", "raw_count": len(records)},
    )


def _run_context(
    runtime: Runtime[AgentRuntimeContext] | None,
    *,
    agent_id: str,
):
    context = runtime.context if runtime is not None else None
    return build_run_context(
        user_id=getattr(context, "user_id", None),
        tenant_id=getattr(context, "tenant_id", None),
        conversation_id=getattr(context, "conversation_id", None),
        permissions=tuple(getattr(context, "permissions", ()) or ()),
        metadata={"agent": agent_id},
    )


async def run_iwencai_source_tool(
    *,
    tool_id: str,
    skill_name: str,
    query: str,
    task_input: Mapping[str, Any],
    runtime: Runtime[AgentRuntimeContext] | None,
    agent_id: str,
    task_id: str,
) -> AgentResult:
    """调用 Skill 声明的问财 Tool，并生成统一 AgentResult。"""
    try:
        skill = load_skill(skill_name)
    except (FileNotFoundError, ValueError) as exc:
        return AgentResult(
            task_id=task_id,
            agent_id=agent_id,
            status="failed",
            error_code="source_skill_invalid",
            gaps=[str(exc)],
        )
    if tool_id not in skill.tool_ids:
        return AgentResult(
            task_id=task_id,
            agent_id=agent_id,
            status="failed",
            error_code="source_skill_tool_mismatch",
            gaps=[f"Skill {skill_name} 未声明 Tool：{tool_id}"],
        )

    try:
        load_all_tools()
        validate_tool_ids([tool_id])
        arguments: dict[str, Any] = {"query": query}
        if tool_id.endswith(".search"):
            arguments["limit"] = int(task_input.get("limit") or 10)
        else:
            arguments.update(
                {
                    "page": int(task_input.get("page") or 1),
                    "limit": int(task_input.get("limit") or 10),
                    "call_type": str(task_input.get("call_type") or "normal"),
                }
            )
        context = runtime.context if runtime is not None else None
        configured_timeout = float(settings.AGENT_V2_TOOL_SKILL_TIMEOUT_SEC)
        if context is not None and context.unit_timeouts:
            timeout_seconds, timeout_limit = context.execution_timeout_for(
                "tool_skill",
                default_seconds=configured_timeout,
            )
        else:
            timeout_seconds, timeout_limit = 0.0, "disabled"
        if timeout_limit == "disabled":
            tool_result = await execute_tool(
                tool_id,
                _run_context(runtime, agent_id=agent_id),
                arguments=arguments,
                allowed_tool_ids={tool_id},
            )
        elif timeout_seconds <= 0:
            if timeout_limit == "hard":
                raise RunHardDeadlineExceeded("run_hard_deadline_exceeded")
            if timeout_limit == "soft":
                raise RunSoftDeadlineExceeded("run_soft_deadline_exceeded")
            raise TimeoutError("tool_skill_timeout")
        else:
            try:
                async with asyncio.timeout(timeout_seconds):
                    tool_result = await execute_tool(
                        tool_id,
                        _run_context(runtime, agent_id=agent_id),
                        arguments=arguments,
                        allowed_tool_ids={tool_id},
                    )
            except TimeoutError as exc:
                if timeout_limit == "hard":
                    raise RunHardDeadlineExceeded(
                        "run_hard_deadline_exceeded"
                    ) from exc
                if timeout_limit == "soft":
                    raise RunSoftDeadlineExceeded(
                        "run_soft_deadline_exceeded"
                    ) from exc
                return AgentResult(
                    task_id=task_id,
                    agent_id=agent_id,
                    status="failed",
                    error_code="tool_skill_timeout",
                    gaps=[f"Tool/Skill 未在 {timeout_seconds:.1f} 秒内完成"],
                )
    except (TypeError, ValueError, KeyError) as exc:
        return AgentResult(
            task_id=task_id,
            agent_id=agent_id,
            status="failed",
            error_code="source_tool_invalid",
            gaps=[str(exc)],
        )

    if not tool_result.ok:
        return AgentResult(
            task_id=task_id,
            agent_id=agent_id,
            status="failed",
            error_code=tool_result.error or "source_tool_failed",
            gaps=[f"受控 Tool 调用失败：{tool_id}"],
        )

    data = (
        tool_result.data
        if isinstance(tool_result.data, dict)
        else {"data": tool_result.data}
    )
    evidence = Evidence(
        evidence_id=f"{tool_id}:{query}",
        task_id=task_id,
        source_type=tool_id,
        provider="iwencai",
        title=tool_id,
        content=f"问财 {tool_id} 查询：{query}",
    )
    structured_data: dict[str, Any] = data
    answer = f"已完成问财 {tool_id} 查询。"
    if tool_id.endswith(".search"):
        document_set = _normalize_document_set(
            tool_id,
            query,
            data,
            evidence.evidence_id,
        )
        structured_data = document_set.model_dump()
        answer = f"已完成问财 {tool_id} 查询，找到 {len(document_set.documents)} 条结果。"

    if not (
        bool(structured_data.get("documents"))
        if tool_id.endswith(".search")
        else _has_records(data)
    ):
        return AgentResult(
            task_id=task_id,
            agent_id=agent_id,
            status="uncovered",
            answer="已完成查询，但未检索到符合条件的结果。",
            structured_data=structured_data,
            evidence=[evidence],
            gaps=[f"{tool_id} 未返回业务记录"],
            error_code=f"{tool_id}_empty_result",
            metadata={
                "mode": "tool",
                "tool_id": tool_id,
                "skill_name": skill_name,
                "query": query,
            },
        )

    return AgentResult(
        task_id=task_id,
        agent_id=agent_id,
        status="completed",
        answer=answer,
        structured_data=structured_data,
        evidence=[evidence],
        metadata={
            "mode": "tool",
            "tool_id": tool_id,
            "skill_name": skill_name,
            "query": query,
        },
    )


__all__ = ["normalize_screening_result", "run_iwencai_source_tool"]
