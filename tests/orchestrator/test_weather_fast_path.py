"""天气解析与普通档路由相关测试。"""

from langchain_core.messages import HumanMessage

from agents.general_agent.weather_direct import parse_weather_request
from agents.orchestrator.analyzer.node import analyze_request
from agents.orchestrator.execution_lane import classify_execution_lane


def test_parse_weather_request_requires_single_explicit_city() -> None:
    request = parse_weather_request("帮我查一下上海未来3天天气")

    assert request is not None
    assert request.city == "上海"
    assert request.days == 3
    assert parse_weather_request("今天天气怎么样") is None
    assert parse_weather_request("北京和上海天气对比") is None


def test_weather_queries_route_to_general_lane() -> None:
    assert classify_execution_lane("上海今天天气怎么样") == "general"
    assert classify_execution_lane("今天天气怎么样") == "general"


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


async def test_general_agent_uses_shared_tool_runtime_for_weather(monkeypatch) -> None:
    """天气不再走零 LLM 快路径，统一走 general 的 run_with_tools。"""
    import importlib

    calls: list[str] = []

    async def fake_run_with_tools(state, **kwargs):
        del state
        calls.append(kwargs.get("agent_name") or "")
        assert kwargs.get("tool_ids") == ("weather.get",)
        assert "小财" in str(kwargs.get("system_prompt") or "")
        from langchain_core.messages import AIMessage

        return {"messages": [AIMessage(content="上海市当前晴，气温 28℃。")]}

    node_module = importlib.import_module("agents.general_agent.node")
    monkeypatch.setattr(node_module, "run_with_tools", fake_run_with_tools)
    output = await node_module.general_agent(
        {"messages": [HumanMessage(content="上海今天天气怎么样")]}
    )

    assert calls == ["general_agent"]
    assert "28℃" in output["messages"][0].content
