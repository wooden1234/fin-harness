"""天气高置信快路径测试。"""

import importlib

from langchain_core.messages import HumanMessage

from agents.general_agent.node import general_agent
from agents.general_agent.weather_direct import parse_weather_request
from agents.orchestrator.analyzer.node import analyze_request


def test_parse_weather_request_requires_single_explicit_city() -> None:
    request = parse_weather_request("帮我查一下上海未来3天天气")

    assert request is not None
    assert request.city == "上海"
    assert request.days == 3
    assert parse_weather_request("今天天气怎么样") is None
    assert parse_weather_request("北京和上海天气对比") is None


async def test_analyzer_uses_deterministic_weather_profile(monkeypatch) -> None:
    async def should_not_call_llm(*args, **kwargs):
        raise AssertionError("明确天气请求不应调用 Analyzer LLM")

    monkeypatch.setattr(
        "agents.orchestrator.analyzer.node.analyze_once",
        should_not_call_llm,
    )
    update = await analyze_request(
        {
            "messages": [HumanMessage(content="上海今天天气怎么样")],
            "rewrite_status": "passthrough",
            "rewritten_query": "上海今天天气怎么样",
        }
    )

    assert update["request_profile"].execution.mode == "general_answer"
    assert update["steps"] == ["orchestrator:analyze_request:deterministic_weather"]


async def test_general_agent_calls_weather_without_llm(monkeypatch) -> None:
    calls = []

    async def fake_fetch(city: str, days: int = 1):
        calls.append((city, days))
        return {
            "ok": True,
            "query_city": city,
            "location": {"local_name": city},
            "current": {
                "condition": "晴",
                "temperature_c": 28,
                "apparent_temperature_c": 29,
                "humidity_pct": 60,
            },
            "daily": [],
        }

    node_module = importlib.import_module("agents.general_agent.node")
    monkeypatch.setattr(node_module, "fetch_weather", fake_fetch)
    output = await general_agent(
        {"messages": [HumanMessage(content="上海今天天气怎么样")]}
    )

    assert calls == [("上海", 1)]
    assert "28℃" in output["messages"][0].content
