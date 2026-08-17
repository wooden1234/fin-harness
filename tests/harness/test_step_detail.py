from datetime import datetime, timezone
import json

from harness.projection.sse import project_session_event
from harness.projection.step_detail import detail_from_tool_call, detail_from_tool_result
from harness.session.types import SessionEvent


def _event(event_type: str, data: dict, seq: int = 1) -> SessionEvent:
    return SessionEvent(
        seq=seq,
        event_id="e",
        event_type=event_type,
        schema_version=1,
        run_id="r1",
        turn=1,
        step=1,
        causation_seq=None,
        correlation_id=None,
        surface_op="none",
        source_event_seqs=(),
        visibility="public",
        data=data,
        created_at=datetime.now(timezone.utc),
    )


def test_detail_from_tool_call_keeps_query():
    detail = detail_from_tool_call('{"query": "永鼎股份2026年半年度净利润预告"}')
    assert detail == {"query": "永鼎股份2026年半年度净利润预告"}


def test_iwencai_facts_become_table():
    payload = {
        "ok": True,
        "query": "永鼎股份净利润",
        "data": {
            "facts": [
                {
                    "entity": "永鼎股份",
                    "metric": "预告净利润下限",
                    "value": 5,
                    "unit": "亿",
                },
                {
                    "entity": "永鼎股份",
                    "metric": "预告净利润上限",
                    "value": 7.0,
                    "unit": "亿",
                },
            ]
        },
    }
    detail = detail_from_tool_result(
        name="query_iwencai",
        content=json.dumps(payload, ensure_ascii=False),
        ok=True,
    )
    assert detail is not None
    assert detail["query"] == "永鼎股份净利润"
    assert detail["columns"] == ["主体", "指标", "数值"]
    assert detail["rows"][0] == ["永鼎股份", "预告净利润下限", "5亿"]


def test_weather_result_becomes_display_text():
    payload = {
        "ok": True,
        "query_city": "西安",
        "location": {"name": "西安"},
        "current": {"condition": "晴", "temperature_c": 26.4},
    }
    detail = detail_from_tool_result(
        name="get_weather",
        content=json.dumps(payload, ensure_ascii=False),
        ok=True,
    )
    assert detail == {"query": "西安", "display_text": "西安 晴 26.4°C"}


def test_sse_tool_result_includes_sanitized_detail():
    event = _event(
        "tool/result",
        {
            "call_id": "c1",
            "name": "query_iwencai",
            "ok": True,
            "content": json.dumps(
                {
                    "ok": True,
                    "query": "贵州茅台最新价",
                    "data": {
                        "facts": [
                            {
                                "entity": "贵州茅台",
                                "metric": "最新价",
                                "value": 1341.99,
                                "unit": "",
                            }
                        ]
                    },
                },
                ensure_ascii=False,
            ),
        },
    )
    payloads = project_session_event(event)
    assert payloads[0]["type"] == "step"
    assert payloads[0]["label"] == "金融查询"
    assert payloads[0]["detail"]["query"] == "贵州茅台最新价"
    assert payloads[0]["detail"]["rows"][0][0] == "贵州茅台"
