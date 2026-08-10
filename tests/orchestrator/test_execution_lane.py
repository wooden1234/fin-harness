"""执行档位定档与普通档透传测试。"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import add_messages

from agents.context_compressor.node import _summary_prompts_for_lane
from agents.context_compressor.prompts import (
    GENERAL_SUMMARY_PROMPT,
    SUMMARY_PROMPT,
)
from agents.final_answer.node import final_answer_node
from agents.image_query_protocol import IMAGE_CLUE_HEADER, USER_INTENT_HEADER
from agents.orchestrator.execution_lane import (
    classify_execution_lane,
    classify_execution_lane_node,
    route_after_execution_lane,
    route_after_rule_lane,
)
from agents.orchestrator.execution_lane_resolver import (
    ExecutionLaneResolution,
    resolve_execution_lane_node,
)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("上海今天天气怎么样", "general"),
        ("帮我查一下上海未来3天天气", "general"),
        ("你好", "general"),
        ("谢谢", "general"),
        ("聊聊天", "general"),
        ("今天天气怎么样", "general"),  # 有天气词但无城市 → 仍普通（绑 weather 工具）
        ("苹果和亚马逊最新财报对比", "deep"),
        ("贵州茅台股价多少", "deep"),
        ("茅台怎么样", "uncertain"),
        ("体感温度是多少", "uncertain"),
        ("湿度高吗", "uncertain"),
        ("出门需要带伞吗", "uncertain"),
        ("你好，帮我看看茅台股价", "deep"),
        ("什么是市盈率", "general"),
        ("如何理解 ROE", "general"),
        ("什么是市盈率，比较腾讯和阿里", "deep"),
        ("查询当前市盈率", "deep"),
        (
            "2026-08-04 01:59:48.335 | WARNING | agents.main_deep_agent.assembly:"
            "run_main_deep_agent:190 - main agent failed: exception_type=ValueError "
            "message=deep_summary_unknown_evidence_id tool_calls=5 journal_entries=7",
            "general",
        ),
        ("Traceback (most recent call last):\n  File \"a.py\", line 1", "general"),
        (
            f"{USER_INTENT_HEADER}\n帮我看看图里写了什么\n\n{IMAGE_CLUE_HEADER}\n"
            "- 类型：股票行情截图\n- 摘要：贵州茅台股价1328.36，跌2.25%\n"
            "- 可见数值：涨跌幅: -2.25% (未核验)；成交量: 661 (未核验)",
            "general",
        ),
        (
            f"{USER_INTENT_HEADER}\n总结截图要点\n\n{IMAGE_CLUE_HEADER}\n"
            "- 类型：财报截图\n- 摘要：营业总收入834亿，净利润416亿",
            "general",
        ),
        (
            f"{USER_INTENT_HEADER}\n贵州茅台最新股价多少\n\n{IMAGE_CLUE_HEADER}\n"
            "- 类型：截图\n- 摘要：个股页面",
            "deep",
        ),
    ],
)
def test_classify_execution_lane_rules(query: str, expected: str) -> None:
    assert classify_execution_lane(query) == expected


def test_routing_query_strips_image_clues() -> None:
    from agents.image_query_protocol import IMAGE_CLUE_HEADER, USER_INTENT_HEADER
    from agents.orchestrator.execution_lane import routing_query_from_message

    full = (
        f"{USER_INTENT_HEADER}\n总结截图要点\n\n{IMAGE_CLUE_HEADER}\n"
        "- 摘要：贵州茅台股价下跌"
    )
    assert routing_query_from_message(full) == "总结截图要点"


@pytest.mark.asyncio
async def test_classify_execution_lane_node_sets_route() -> None:
    update = await classify_execution_lane_node(
        {"messages": [HumanMessage(content="上海天气怎么样")]}
    )
    assert update["execution_lane"] == "general"
    assert update["route"] == "general"

    deep_update = await classify_execution_lane_node(
        {"messages": [HumanMessage(content="查询最新财报")]}
    )
    assert deep_update["execution_lane"] == "deep"
    assert deep_update["route"] == "deep"

    uncertain_update = await classify_execution_lane_node(
        {"messages": [HumanMessage(content="体感温度是多少？")]}
    )
    assert uncertain_update["execution_lane"] == "uncertain"
    assert "route" not in uncertain_update


def test_route_after_rule_lane_only_resolves_uncertain_with_llm() -> None:
    assert route_after_rule_lane({"execution_lane": "general"}) == "resolved"
    assert route_after_rule_lane({"execution_lane": "deep"}) == "resolved"
    assert route_after_rule_lane({"execution_lane": "uncertain"}) == "uncertain"


def test_execution_lane_resolution_accepts_tier_alias() -> None:
    """模型偶发输出 tier 时仍应解析为 lane，避免灰区错误降级到 general。"""
    parsed = ExecutionLaneResolution.model_validate({"tier": "deep"})
    assert parsed.lane == "deep"


@pytest.mark.asyncio
async def test_llm_resolver_routes_weather_followup_to_general(monkeypatch) -> None:
    async def fake_resolve(*args, **kwargs):
        return ExecutionLaneResolution(
            lane="general",
            intent="weather_followup",
            confidence=0.98,
            reason="上一轮是成都天气查询",
        )

    monkeypatch.setattr(
        "agents.orchestrator.execution_lane_resolver._resolve_with_llm",
        fake_resolve,
    )
    update = await resolve_execution_lane_node(
        {
            "messages": [
                HumanMessage(content="成都今天天气怎么样？"),
                AIMessage(content="成都当前多云，气温28℃，体感31℃，湿度62%。"),
                HumanMessage(content="体感温度是多少？"),
            ]
        }
    )

    assert update["execution_lane"] == "general"
    assert update["route"] == "general"


@pytest.mark.asyncio
async def test_llm_resolver_failure_falls_back_to_general(monkeypatch) -> None:
    async def fail_resolve(*args, **kwargs):
        raise TimeoutError("router timeout")

    monkeypatch.setattr(
        "agents.orchestrator.execution_lane_resolver._resolve_with_llm",
        fail_resolve,
    )
    update = await resolve_execution_lane_node(
        {"messages": [HumanMessage(content="帮我解释一下这个概念")]}
    )

    assert update["execution_lane"] == "general"
    assert update["route"] == "general"


def test_route_after_execution_lane_defaults_to_deep() -> None:
    assert route_after_execution_lane({"execution_lane": "general"}) == "general"
    assert route_after_execution_lane({"execution_lane": "deep"}) == "deep"
    assert route_after_execution_lane({}) == "deep"
    # compress 节点若收窄通道丢了 execution_lane，仍可用 route 兜底
    assert route_after_execution_lane({"route": "general"}) == "general"


def test_summary_prompts_follow_execution_lane() -> None:
    general_prompt, _ = _summary_prompts_for_lane("general")
    finance_prompt, _ = _summary_prompts_for_lane("deep")
    assert general_prompt is GENERAL_SUMMARY_PROMPT
    assert finance_prompt is SUMMARY_PROMPT
    assert "日常对话摘要" in general_prompt
    assert "金融对话摘要" in finance_prompt


@pytest.mark.asyncio
async def test_final_answer_passthrough_for_general_lane() -> None:
    candidate = AIMessage(
        content="上海当前晴，气温 29℃。",
        id="general-answer-1",
    )
    update = await final_answer_node(
        {
            "execution_lane": "general",
            "route": "general",
            "messages": [candidate],
            "citations": [],
        }
    )
    assert update["messages"][0].content == "上海当前晴，气温 29℃。"
    assert update["messages"][0].id == candidate.id
    merged = add_messages([candidate], update["messages"])
    assert len(merged) == 1
    assert update["compliance_reason_code"] == "general_lane_passthrough"
    assert update["compliance_action"] == "pass"
    assert update["pending_query_clarification"] == {}
