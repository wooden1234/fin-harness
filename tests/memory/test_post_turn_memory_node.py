from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage
import pytest

from agents.memory_recall import node


def _runtime():
    return SimpleNamespace(
        context=SimpleNamespace(
            tenant_id="tenant-1",
            user_id=7,
            conversation_id=12,
            run_id="run-1",
        )
    )


@pytest.mark.asyncio
async def test_post_turn_memory_enqueues_episodic_extraction(monkeypatch):
    calls: list[dict] = []

    monkeypatch.setattr(node, "is_substantive_progress", lambda **_kwargs: True)

    async def fake_enqueue(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        node.OutboxService,
        "enqueue_episodic_extraction",
        fake_enqueue,
    )

    result = await node.post_turn_memory_node(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "请分析这家公司最近几个季度的业绩变化、盈利能力、现金流状况，"
                        "并说明当前最需要关注的经营风险。"
                    )
                ),
                AIMessage(
                    content=(
                        "已完成分析。结果包括收入和利润变化、盈利能力、现金流状况，"
                        "以及经营风险、证据缺口和后续需要持续跟踪的指标。"
                    )
                ),
            ],
            "execution_status": "completed",
            "agent_results": [object()],
        },
        _runtime(),
    )

    assert result["post_turn_memory_status"] == "enqueued"
    assert result["post_turn_memory_enqueued"] == ["episodic"]
    assert calls[0]["run_id"] == "run-1"
    assert calls[0]["query"].startswith("请分析")


@pytest.mark.asyncio
async def test_post_turn_memory_enqueues_implicit_preference(monkeypatch):
    calls: list[str] = []

    async def fake_enqueue(**kwargs):
        calls.append("preference")

    monkeypatch.setattr(
        node.OutboxService,
        "enqueue_memory_extraction",
        fake_enqueue,
    )

    result = await node.post_turn_memory_node(
        {
            "messages": [
                HumanMessage(content="以后回答尽量简洁一些"),
                AIMessage(content="好的，之后我会尽量简洁回答。"),
            ],
            "execution_status": "completed",
        },
        _runtime(),
    )

    assert result["post_turn_memory_status"] == "enqueued"
    assert calls == ["preference"]


@pytest.mark.asyncio
async def test_post_turn_memory_does_not_process_blocked_answer(monkeypatch):
    async def fail_if_called(**_kwargs):
        raise AssertionError("blocked answer must not enqueue memory extraction")

    monkeypatch.setattr(node.OutboxService, "enqueue_memory_extraction", fail_if_called)
    monkeypatch.setattr(node.OutboxService, "enqueue_episodic_extraction", fail_if_called)

    result = await node.post_turn_memory_node(
        {
            "messages": [HumanMessage(content="敏感请求"), AIMessage(content="已拦截")],
            "guardrails_pass": False,
        },
        _runtime(),
    )

    assert result["post_turn_memory_status"] == "skipped_guardrail"
