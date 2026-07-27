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
from agents.orchestrator.state import OrchestratorState
from app.core.logger import get_logger

logger = get_logger(service="orchestrator_analyzer")

# 单次请求 Analyzer LLM 上限：第 2 次只能是瞬时重试或 repair，二者互斥。
_MAX_ANALYZER_LLM_CALLS = 2


async def analyze_request(
    state: OrchestratorState,
    config: RunnableConfig = None,
) -> dict[str, Any]:
    """分析用户请求，写入 request_profile；失败时回退启发式。"""
    query = latest_query(state)
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
        return await analyze_once(query, config=config)

    async def _repair(raw: Any, issues: list[str]) -> Any:
        nonlocal llm_calls
        llm_calls += 1
        return await repair_profile(query, raw, issues, config=config)

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
                    "request_profile": heuristic_profile(query),
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
                "request_profile": heuristic_profile(query),
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
            "request_profile": heuristic_profile(query),
            "steps": ["orchestrator:analyze_request:heuristic_error_fallback"],
        }


__all__ = ["analyze_request"]
