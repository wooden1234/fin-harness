import pytest

from harness.tools.arguments import coerce_tool_arguments, parse_json_objects
from harness.tools.definition import ToolDefinition, function_schema
from harness.tools.pipeline import ToolPipeline
from harness.tools.scheduler import _parse_arguments


def test_parse_arguments_keeps_first_object_when_concatenated():
    parsed = _parse_arguments(
        '{"query": "永鼎股份2026年半年度业绩预告 净利润"}{"query": "永鼎股份2026年半年度净利润预计"}'
    )
    assert parsed == {"query": "永鼎股份2026年半年度业绩预告 净利润"}


def test_parse_json_objects_splits_concatenated_payloads():
    raw = (
        '{"query": "永鼎股份、中天科技、亨通光电 2025年报 净利润 毛利率 营业收入", '
        '"entities": ["永鼎股份", "中天科技", "亨通光电"]}'
        '{"query": "永鼎股份、中天科技、亨通光电 市盈率 总市值 市净率", '
        '"entities": ["永鼎股份", "中天科技", "亨通光电"]}'
        '{"query": "永鼎股份 2026年半年报业绩预告 净利润"}'
    )
    objects = parse_json_objects({"_raw": raw})
    assert len(objects) == 3
    assert objects[0]["query"].startswith("永鼎股份、中天科技、亨通光电 2025年报")
    assert objects[2] == {"query": "永鼎股份 2026年半年报业绩预告 净利润"}


def test_coerce_strips_entities_for_announcement_schema():
    schema = {
        "type": "function",
        "function": {
            "name": "search_iwencai_announcement",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
            },
        },
    }
    payloads = coerce_tool_arguments(
        {
            "_raw": (
                '{"query": "永鼎股份 2025年报 净利润", "entities": ["永鼎股份"]}'
                '{"query": "永鼎股份 市盈率", "entities": ["永鼎股份"]}'
            )
        },
        openai_schema=schema,
    )
    assert payloads == [
        {"query": "永鼎股份 2025年报 净利润"},
        {"query": "永鼎股份 市盈率"},
    ]


@pytest.mark.asyncio
async def test_pipeline_runs_concatenated_objects_in_parallel():
    seen: list[str] = []

    async def handler(arguments: dict) -> dict:
        seen.append(str(arguments.get("query") or ""))
        return {"ok": True, "data": {"datas": [{"query": arguments["query"]}]}}

    definition = ToolDefinition(
        tool_id="iwencai.announcement.search",
        name="search_iwencai_announcement",
        description="search",
        handler=handler,
        openai_schema=function_schema(
            "search_iwencai_announcement",
            "search",
            {
                "type": "object",
                "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
            },
        ),
    )
    raw = (
        '{"query": "永鼎股份 2025年报 净利润", "entities": ["永鼎股份"]}'
        '{"query": "永鼎股份 市盈率", "entities": ["永鼎股份"]}'
        '{"query": "永鼎股份 2026年半年报业绩预告 净利润"}'
    )
    result = await ToolPipeline().run(definition, raw)
    assert result["ok"] is True
    assert result["batched"] is True
    assert seen == [
        "永鼎股份 2025年报 净利润",
        "永鼎股份 市盈率",
        "永鼎股份 2026年半年报业绩预告 净利润",
    ]
    assert len(result["data"]["datas"]) == 3
