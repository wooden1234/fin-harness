"""finance_agent fan-in / join 测试。"""

from __future__ import annotations

import pytest
from langgraph.graph import END
from langgraph.types import Send

from app.agents.finance_agent.join import fan_in_ready, sub_task_satisfied
from app.agents.finance_agent.join.node import route_after_join
from app.agents.finance_agent.planner.dispatch_workers import (
    route_after_dispatch_workers,
    route_after_retrieval_worker,
)
from app.agents.states import FinAgentState, SubTask


def test_sub_task_satisfied_waits_for_web_fallback():
    assert not sub_task_satisfied(
        "t1",
        [{"sub_task_id": "t1", "type": "faq", "fallback_to_web": True}],
    )
    assert sub_task_satisfied(
        "t1",
        [
            {"sub_task_id": "t1", "type": "faq", "fallback_to_web": True},
            {"sub_task_id": "t1", "type": "web_search"},
        ],
    )


def test_fan_in_ready_requires_all_sub_tasks():
    tasks = [
        SubTask(id="a", question="q1", type="faq"),
        SubTask(id="b", question="q2", type="web_search"),
    ]
    assert not fan_in_ready(
        sub_tasks=tasks,
        task_results=[{"sub_task_id": "b", "type": "web_search"}],
    )
    assert fan_in_ready(
        sub_tasks=tasks,
        task_results=[
            {"sub_task_id": "a", "type": "faq"},
            {"sub_task_id": "b", "type": "web_search"},
        ],
    )


def test_route_after_join_waits_until_all_sub_tasks_ready():
    tasks = [
        SubTask(id="a", question="q1", type="faq"),
        SubTask(id="b", question="q2", type="web_search"),
    ]
    waiting: FinAgentState = {
        "sub_tasks": tasks,
        "task_results": [{"sub_task_id": "b", "type": "web_search"}],
    }
    ready: FinAgentState = {
        "sub_tasks": tasks,
        "task_results": [
            {"sub_task_id": "a", "type": "faq"},
            {"sub_task_id": "b", "type": "web_search"},
        ],
    }
    assert route_after_join(waiting) == END
    assert route_after_join(ready) == "summarize"


@pytest.mark.asyncio
async def test_graph_builds_and_routes_asymmetric_web_fallback():
    """显式图可构建；faq→web 兜底与直达 web 使用同一 join 收齐规则。"""
    from app.agents.finance_agent.graph import build_finance_agent_subgraph

    build_finance_agent_subgraph().compile()

    tasks = [
        SubTask(
            id="f1",
            question="交易规则是什么",
            intent="concept_explain",
            type="faq",
            evidence_chain=["faq", "web_search"],
        ),
        SubTask(
            id="w1",
            question="最近有什么新规",
            intent="market_event",
            type="web_search",
            evidence_chain=["web_search"],
        ),
    ]
    sends = route_after_dispatch_workers({"sub_tasks": tasks})
    assert [send.node for send in sends] == ["faq_agent", "web_search_agent"]
    assert all(isinstance(send, Send) for send in sends)

    fallback_route = route_after_retrieval_worker(
        {
            "sub_task_id": "f1",
            "sub_question": "交易规则是什么",
            "evidence_chain": ["faq", "web_search"],
            "task_results": [
                {
                    "sub_task_id": "f1",
                    "type": "faq",
                    "coverage": "uncovered",
                    "fallback_to_web": True,
                }
            ],
        }
    )
    assert isinstance(fallback_route, Send)
    assert fallback_route.node == "web_search_agent"

    assert not fan_in_ready(
        sub_tasks=tasks,
        task_results=[
            {"sub_task_id": "w1", "type": "web_search"},
            {"sub_task_id": "f1", "type": "faq", "fallback_to_web": True},
        ],
    )
    assert fan_in_ready(
        sub_tasks=tasks,
        task_results=[
            {"sub_task_id": "w1", "type": "web_search"},
            {"sub_task_id": "f1", "type": "faq", "fallback_to_web": True},
            {"sub_task_id": "f1", "type": "web_search"},
        ],
    )
