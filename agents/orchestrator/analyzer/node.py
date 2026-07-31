"""Orchestrator analyze_request 节点：LLM 画像 + 规则校验 + 启发式回退。"""

from __future__ import annotations

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
from agents.orchestrator.contracts import AgentResult
from agents.orchestrator.state import OrchestratorState
from app.core.logger import get_logger

logger = get_logger(service="orchestrator_analyzer")

# 单次请求 Analyzer LLM 上限：第 2 次只能是瞬时重试或 repair，二者互斥。
_MAX_ANALYZER_LLM_CALLS = 2


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
            logger.warning(
                "analyzer transient api error, retrying once: {}",
                type(exc).__name__,
            )
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
        logger.warning(
            "analyzer llm failed, fallback heuristic: {}",
            type(exc).__name__,
        )
        return {
            "request_profile": _heuristic_from_envelope(envelope),
            "steps": ["orchestrator:analyze_request:heuristic_error_fallback"],
        }


__all__ = ["analyze_request", "build_analyzer_envelope"]
