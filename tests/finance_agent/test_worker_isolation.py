"""worker_isolation 单元测试。"""

import pytest

from app.agents.finance_agent.workers import (
    isolate_worker_node,
    project_worker_updates_to_parent,
)


def test_project_worker_updates_to_parent_keeps_only_safe_keys():
    projected = project_worker_updates_to_parent(
        {
            "task_results": [{"sub_task_id": "t1"}],
            "citations": [{"source": "a.pdf"}],
            "messages": ["m"],
            "steps": ["s1"],
            "financial_query_sql": "SELECT 1",
            "financial_query_text": "营收",
            "sub_question": "should-not-bubble",
        }
    )

    assert projected == {
        "task_results": [{"sub_task_id": "t1"}],
        "citations": [{"source": "a.pdf"}],
        "messages": ["m"],
        "steps": ["s1"],
    }


@pytest.mark.asyncio
async def test_isolate_worker_node_strips_private_fields_from_async_fn():
    async def worker(state, config=None):
        return {
            "task_results": [{"sub_task_id": state["sub_task_id"]}],
            "financial_query_sql": "SELECT leaked",
        }

    isolated = isolate_worker_node(worker)
    out = await isolated({"sub_task_id": "x1"}, {})

    assert out == {"task_results": [{"sub_task_id": "x1"}]}
