"""每轮临时工作区：统一重置清单。

跨轮保留（禁止出现在重置清单）：
- messages
- conversation_summary
- conversation_summary_until
- conversation_summary_v2
- pending_query_clarification
- pending_memory_action

带 reducer（``add``）的字段必须用 ``Overwrite`` 清空，否则会跨轮无限增长；
``steps`` 即为此类字段。
"""

from __future__ import annotations

from langgraph.types import Overwrite


def begin_turn_workspace() -> dict:
    """每轮入口（init_turn）统一重置临时工作区。"""
    return {
        # ── reducer 字段：必须 Overwrite ──
        "steps": Overwrite([]),
        "task_results": Overwrite([]),
        "citations": Overwrite([]),
        "agent_results": Overwrite([]),
        "evidence": Overwrite([]),

        # ── 本轮派生 / 路由 / 护栏 / 合规 ──
        "rewritten_query": "",
        "rewrite_status": "",
        "rewrite_reason_codes": [],
        "rewrite_failure_kind": "",
        "rewrite_resolution": {},
        "rewrite_clarification_message": "",
        "context_admission_rejected": False,
        "summary": "",
        "rag_trace": {},
        "route": "",
        "execution_lane": "",
        "guardrail_decision": {},
        "guardrails_reason": "",
        "compliance_action": "",
        "compliance_reason_code": "",
        "compliance_reason": "",
        "turn_preferences": {},
        "memory_action_handled": False,
        "memory_cache_bypass": False,
        "request_profile": None,
        "task_plan": None,
        "claims": [],
        "claim_evidence_links": [],
        "evidence_assessments": [],
        "evidence_conflicts": [],
        "constrained_answer": None,
        "quality_report": None,
        "main_agent_response": None,
        "main_agent_journal": {},
        "main_quality_metrics": {},
        "answer_follow_ups": [],
        "answer_charts": [],
        "investment_action_sensitive": False,
        "execution_mode": "",
        "execution_status": "",
        "memory_requirements": {},
        "task_memory_context": {},
        "current_task_memory_context": {},

        # ── Planner / 派发工作区 ──
        "planner_query": "",
        "planner_raw_tasks": [],
        "planner_validation_issues": [],
        "planner_needs_repair": False,
        "planner_repair_attempted": False,
        "planner_error_reason": "",
        "sub_tasks": [],
        "sub_question": "",
        "sub_task_id": "",
        "evidence_chain": [],

        # ── financial_query 子图临时字段 ──
        "financial_query_text": "",
        "financial_query_intent": None,
        "financial_query_plan_route": "",
        "financial_query_plan_reason": "",
        "financial_query_missing_fields": [],
        "financial_query_sql": "",
        "financial_query_validated_sql": "",
        "financial_query_validation_error": "",
        "financial_query_validation_error_type": "",
        "financial_query_validation_errors": [],
        "financial_query_validation_error_types": [],
        "financial_query_sql_attempts": 0,
        "financial_query_next_action_sql": "",
        "financial_query_sql_params": {},
        "financial_query_template_id": None,
        "financial_query_schema_prompt": "",
        "financial_query_fewshot_examples": "",
        "financial_query_failure_category": None,
        "financial_query_failure_code": "",
        "financial_query_failure_retryable": False,
    }


def reset_worker_workspace() -> dict:
    """进入 plan/worker 前再次清空并行输出（不碰 steps，保留本轮已记录步骤）。"""
    return {
        "task_results": Overwrite([]),
        "citations": Overwrite([]),
        "summary": "",
        "rag_trace": {},
    }


__all__ = ["begin_turn_workspace", "reset_worker_workspace"]
