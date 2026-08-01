"""Orchestrator analyze_request 节点：LLM 画像 + 规则校验 + 启发式回退。"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from langchain_core.runnables import RunnableConfig

from agents.orchestrator.analyzer.heuristic import heuristic_profile, latest_query
from agents.orchestrator.analyzer.llm import (
    analyze_once,
    is_transient_api_error,
    repair_profile,
)
from agents.orchestrator.analyzer.validate import validate_and_normalize
from agents.orchestrator.analyzer.schema import (
    ActiveTopicProjection,
    AnalyzerInputEnvelope,
    ArtifactDescriptor,
)
from agents.context_compressor.structured import parse_summary_v2
from agents.general_agent.weather_direct import parse_weather_request
from agents.orchestrator.contracts import AgentResult
from agents.orchestrator.state import OrchestratorState
from app.core.logger import get_logger

logger = get_logger(service="orchestrator_analyzer")

# 单次请求 Analyzer LLM 上限：第 2 次只能是瞬时重试或 repair，二者互斥。
_MAX_ANALYZER_LLM_CALLS = 2
_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b", re.IGNORECASE),
    re.compile(r"\b(Bearer\s+)[A-Za-z0-9._~-]+", re.IGNORECASE),
)


def _safe_log_text(value: Any, *, max_length: int = 1200) -> str:
    """清理供应商错误字段，避免凭据和控制字符进入日志。"""
    text = " ".join(str(value or "").split())
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(
            lambda match: f"{match.group(1)}[REDACTED]"
            if match.lastindex
            else "[REDACTED]",
            text,
        )
    return text[:max_length] or "unknown"


def _analyzer_error_details(exc: BaseException) -> dict[str, str]:
    """仅提取供应商错误的白名单字段，不记录请求体和 Prompt。"""
    details = {
        "exception_type": type(exc).__name__,
        "status_code": "unknown",
        "provider_type": "unknown",
        "provider_code": "unknown",
        "param": "unknown",
        "request_id": "unknown",
        "message": "unknown",
        "parser_reason": "unknown",
        "output_length": "unknown",
        "output_sha256": "unknown",
    }
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        status_code = getattr(current, "status_code", None)
        response = getattr(current, "response", None)
        if status_code is None and response is not None:
            status_code = getattr(response, "status_code", None)
        if status_code is not None:
            details["status_code"] = _safe_log_text(status_code, max_length=16)

        if response is not None:
            headers = getattr(response, "headers", None)
            if headers is not None:
                request_id = headers.get("x-request-id") or headers.get("request-id")
                if request_id:
                    details["request_id"] = _safe_log_text(request_id, max_length=128)

        body = getattr(current, "body", None)
        if isinstance(body, dict):
            provider_error = body.get("error", body)
            if isinstance(provider_error, dict):
                field_map = {
                    "type": "provider_type",
                    "code": "provider_code",
                    "param": "param",
                    "message": "message",
                }
                for source, target in field_map.items():
                    value = provider_error.get(source)
                    if value not in (None, ""):
                        details[target] = _safe_log_text(value)
                break

        if "outputparser" in type(current).__name__.lower():
            raw_output = getattr(current, "llm_output", None)
            if raw_output is not None:
                raw_text = str(raw_output)
                details["output_length"] = str(len(raw_text))
                details["output_sha256"] = hashlib.sha256(
                    raw_text.encode("utf-8", errors="replace")
                ).hexdigest()[:16]
            observation = str(getattr(current, "observation", "") or "").lower()
            error_text = str(current).lower()
            parser_text = f"{observation} {error_text}"
            if "pydantic" in parser_text or "validation" in parser_text:
                details["parser_reason"] = "schema_validation_failed"
            elif "json" in parser_text:
                details["parser_reason"] = "invalid_json_output"
            elif not raw_output:
                details["parser_reason"] = "empty_model_output"
            else:
                details["parser_reason"] = "structured_output_parse_failed"
            # 不记录解析器原文，避免把用户问题或模型回答写入日志。
            details["message"] = details["parser_reason"]

        current = current.__cause__ or current.__context__
    return details


def _log_analyzer_error(stage: str, exc: BaseException) -> None:
    details = _analyzer_error_details(exc)
    logger.warning(
        "analyzer {}: exception_type={} status_code={} provider_type={} "
        "provider_code={} param={} request_id={} message={}",
        stage,
        details["exception_type"],
        details["status_code"],
        details["provider_type"],
        details["provider_code"],
        details["param"],
        details["request_id"],
        details["message"],
    )
    if details["parser_reason"] != "unknown":
        logger.warning(
            "analyzer parser diagnostics reason={} output_length={} output_sha256={}",
            details["parser_reason"],
            details["output_length"],
            details["output_sha256"],
        )


def _active_topic_projection(state: OrchestratorState) -> ActiveTopicProjection | None:
    summary = parse_summary_v2(state.get("conversation_summary_v2"))
    if summary is None or summary.active_topic_id is None:
        return None
    topic = next(
        (item for item in summary.topics if item.topic_id == summary.active_topic_id),
        None,
    )
    if topic is None:
        return None
    finance = topic.finance_context
    return ActiveTopicProjection(
        topic_id=topic.topic_id,
        domain=topic.domain,
        entities=[item.name for item in topic.entities],
        securities=list(finance.securities) if finance else [],
        metrics=list(finance.metrics) if finance else [],
        time_ranges=list(finance.time_ranges) if finance else [],
    )


def _artifact_descriptors(state: OrchestratorState) -> list[ArtifactDescriptor]:
    descriptors: list[ArtifactDescriptor] = []
    for raw_result in reversed(list(state.get("agent_results") or [])):
        try:
            result = (
                raw_result
                if isinstance(raw_result, AgentResult)
                else AgentResult.model_validate(raw_result)
            )
        except ValueError:
            continue
        data = result.structured_data or {}
        dataset_id = str(data.get("dataset_id") or "")
        rows = data.get("rows")
        if not dataset_id or not isinstance(rows, list):
            continue
        fields = list(
            dict.fromkeys(
                str(key)
                for row in rows[:20]
                if isinstance(row, dict)
                for key in row.keys()
            )
        )[:64]
        descriptors.append(
            ArtifactDescriptor(
                artifact_type="CandidateSet",
                artifact_id=dataset_id,
                universe=str(data.get("universe") or ""),
                as_of=str(data.get("as_of") or ""),
                row_count=len(rows),
                available_fields=fields,
                source_task_id=result.task_id,
            )
        )
        if len(descriptors) >= 3:
            break
    return descriptors


def build_analyzer_envelope(state: OrchestratorState) -> AnalyzerInputEnvelope:
    query = latest_query(state)
    resolution = state.get("rewrite_resolution") or {}
    resolved_entities = (
        list(resolution.get("resolved_entities") or [])
        if isinstance(resolution, dict)
        else []
    )
    return AnalyzerInputEnvelope(
        original_query=query,
        rewritten_query=str(state.get("rewritten_query") or ""),
        rewrite_status=str(state.get("rewrite_status") or "passthrough"),
        resolved_entities=[str(item) for item in resolved_entities if str(item)],
        active_topic=_active_topic_projection(state),
        artifacts=_artifact_descriptors(state),
    )


def _heuristic_from_envelope(envelope: AnalyzerInputEnvelope):
    """启发式回退也只消费 Analyzer 信封中的安全投影。"""
    candidate_set_id = next(
        (
            item.artifact_id
            for item in envelope.artifacts
            if item.artifact_type == "CandidateSet" and item.artifact_id
        ),
        None,
    )
    return heuristic_profile(
        envelope.effective_query,
        candidate_set_id=candidate_set_id,
    )


async def analyze_request(
    state: OrchestratorState,
    config: RunnableConfig = None,
) -> dict[str, Any]:
    """分析用户请求，写入 request_profile；失败时回退启发式。"""
    envelope = build_analyzer_envelope(state)
    query = envelope.effective_query
    if not query:
        profile = heuristic_profile("")
        return {
            "request_profile": profile,
            "steps": ["orchestrator:analyze_request:empty"],
        }

    if parse_weather_request(query) is not None:
        return {
            "request_profile": _heuristic_from_envelope(envelope),
            "steps": ["orchestrator:analyze_request:deterministic_weather"],
        }

    llm_calls = 0

    async def _analyze() -> Any:
        nonlocal llm_calls
        llm_calls += 1
        return await analyze_once(envelope, config=config)

    async def _repair(raw: Any, issues: list[str]) -> Any:
        nonlocal llm_calls
        llm_calls += 1
        return await repair_profile(envelope, raw, issues, config=config)

    try:
        try:
            raw = await _analyze()
        except Exception as exc:
            if not is_transient_api_error(exc) or llm_calls >= _MAX_ANALYZER_LLM_CALLS:
                raise
            _log_analyzer_error("transient api error, retrying once", exc)
            raw = await _analyze()

        validation = validate_and_normalize(raw, original_query=query)
        if validation.needs_repair:
            if llm_calls >= _MAX_ANALYZER_LLM_CALLS:
                logger.warning(
                    "analyzer needs repair but llm budget exhausted issues={}, "
                    "fallback heuristic",
                    validation.issues,
                )
                return {
                    "request_profile": _heuristic_from_envelope(envelope),
                    "steps": ["orchestrator:analyze_request:heuristic_fallback"],
                }
            logger.warning(
                "analyzer validation issues={}, attempting repair",
                validation.issues,
            )
            repaired = await _repair(raw, validation.issues)
            validation = validate_and_normalize(repaired, original_query=query)

        if validation.needs_repair:
            logger.warning(
                "analyzer repair still invalid issues={}, fallback heuristic",
                validation.issues,
            )
            return {
                "request_profile": _heuristic_from_envelope(envelope),
                "steps": ["orchestrator:analyze_request:heuristic_fallback"],
            }
        return {
            "request_profile": validation.profile,
            "steps": ["orchestrator:analyze_request:llm"],
        }
    except Exception as exc:  # noqa: BLE001
        _log_analyzer_error("llm failed, fallback heuristic", exc)
        fallback_step = (
            "orchestrator:analyze_request:parser_fallback"
            if "outputparser" in type(exc).__name__.lower()
            else "orchestrator:analyze_request:heuristic_error_fallback"
        )
        return {
            "request_profile": _heuristic_from_envelope(envelope),
            "steps": [fallback_step],
        }


__all__ = ["analyze_request", "build_analyzer_envelope"]
