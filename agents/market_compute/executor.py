"""Root Orchestrator 调用确定性 market.compute 的适配入口。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from agents.market_compute.compute import MarketComputationError, compute_candidate_set
from agents.orchestrator.contracts import (
    AgentResult,
    CandidateSet,
    Evidence,
    MarketQueryPlan,
)


def _dependencies(state: Mapping[str, Any]) -> list[AgentResult]:
    results: list[AgentResult] = []
    for item in list(state.get("dependency_results") or []):
        try:
            results.append(
                item if isinstance(item, AgentResult) else AgentResult.model_validate(item)
            )
        except ValidationError:
            continue
    return results


def _candidate_sets(results: list[AgentResult]) -> list[CandidateSet]:
    candidates: list[CandidateSet] = []
    for result in results:
        try:
            candidates.append(CandidateSet.model_validate(result.structured_data))
        except ValidationError:
            continue
    return candidates


def _evidence(results: list[AgentResult]) -> list[Evidence]:
    merged: list[Evidence] = []
    seen: set[str] = set()
    for result in results:
        for item in result.evidence:
            if item.evidence_id not in seen:
                seen.add(item.evidence_id)
                merged.append(item)
    return merged


async def run_market_compute(
    state: Mapping[str, Any],
    *,
    query: str,
) -> AgentResult:
    """只计算上游 CandidateSet；禁止在缺少数据时自动采集。"""
    results = _dependencies(state)
    candidates = _candidate_sets(results)
    evidence = _evidence(results)
    if len(candidates) != 1:
        return AgentResult(
            task_id="market-compute",
            agent_id="market.compute",
            status="clarify",
            answer="market.compute 需要且只接受一个上游候选数据集。",
            error_code="candidate_set_missing" if not candidates else "candidate_set_ambiguous",
            gaps=["请明确指定一个 CandidateSet"],
            evidence=evidence,
        )

    task_input = state.get("task_input")
    raw_plan = (
        task_input.get("market_query_plan") or task_input.get("query_plan")
        if isinstance(task_input, Mapping)
        else None
    )
    if raw_plan is None:
        return AgentResult(
            task_id="market-compute",
            agent_id="market.compute",
            status="clarify",
            answer="已收到候选数据，但缺少确定性的市场计算计划。",
            error_code="market_query_plan_missing",
            gaps=["task_input.market_query_plan 未提供"],
            evidence=evidence,
        )
    try:
        plan = MarketQueryPlan.model_validate(raw_plan)
        computed = compute_candidate_set(candidates[0], plan)
    except (ValidationError, MarketComputationError) as exc:
        return AgentResult(
            task_id="market-compute",
            agent_id="market.compute",
            status="failed",
            error_code="market_computation_failed",
            gaps=[str(exc)],
            evidence=evidence,
        )

    return AgentResult(
        task_id="market-compute",
        agent_id="market.compute",
        status="completed",
        answer=f"市场数据计算完成，得到 {len(computed.rows)} 条候选记录。",
        structured_data=computed.model_dump(),
        evidence=evidence,
        metadata={"mode": "deterministic", "operation": "market.compute", "query": query},
    )


__all__ = ["run_market_compute"]
