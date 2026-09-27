"""iwencai 统一只读缓存入口测试。"""

from __future__ import annotations

import asyncio

import pytest

from app.core.cache import reset_cache_metrics
from tools import iwencai as iwencai_tools


@pytest.mark.asyncio
async def test_screen_iwencai_uses_cached_skill_not_runner(monkeypatch) -> None:
    reset_cache_metrics()
    calls: list[str] = []

    async def fake_cached(*args, **kwargs):
        calls.append("cached")
        return {"ok": True, "cache_status": "miss", "data": {}}

    async def boom(*args, **kwargs):
        calls.append("runner")
        raise AssertionError("screen_iwencai must not call runner directly")

    monkeypatch.setattr(iwencai_tools, "_run_cached_skill", fake_cached)
    monkeypatch.setattr(iwencai_tools, "run_installed_skill", boom)
    result = await iwencai_tools.screen_iwencai.ainvoke(
        {"query": "新能源", "page": 1, "limit": 5, "call_type": "normal"}
    )
    assert result["ok"] is True
    assert calls == ["cached"]


@pytest.mark.asyncio
async def test_finance_and_usstock_tools_use_cached_skill(monkeypatch) -> None:
    reset_cache_metrics()
    seen: list[str] = []

    async def fake_cached(skill_id, *args, **kwargs):
        seen.append(skill_id)
        return {
            "ok": True,
            "cache_status": "miss",
            "data": {
                "query": "测试",
                "datas": [{"股票简称": "示例", "营业收入[20251231]": "1亿"}],
            },
        }

    monkeypatch.setattr(iwencai_tools, "_run_cached_skill", fake_cached)
    finance = await iwencai_tools.query_iwencai_finance.ainvoke(
        {"query": "示例营业收入", "page": 1, "limit": 5, "call_type": "normal"}
    )
    usstock = await iwencai_tools.screen_iwencai_usstock.ainvoke(
        {"query": "美股市值前十", "page": 1, "limit": 5, "call_type": "normal"}
    )
    assert seen == ["hithink-finance-query", "hithink-usstock-selector"]
    assert finance["ok"] is True
    assert finance["data"]["facts"]
    assert usstock["ok"] is True


@pytest.mark.asyncio
async def test_partial_screen_refetches_up_to_configured_maximum(monkeypatch) -> None:
    limits: list[int] = []

    async def fake_cached(_skill_id, *, limit, **_kwargs):
        limits.append(limit)
        if limit < 100:
            return {
                "ok": True,
                "data": {
                    "code_count": 646,
                    "returned_count": 20,
                    "page": 1,
                    "limit": limit,
                    "pagination_tip": "如需更多数据，请使用 --page 参数翻页。",
                    "datas": [{}] * 20,
                },
            }
        return {
            "ok": True,
            "data": {
                "code_count": 646,
                "returned_count": 100,
                "page": 1,
                "limit": limit,
                "datas": [{}] * 100,
            },
        }

    monkeypatch.setattr(iwencai_tools, "_run_cached_skill", fake_cached)
    result = await iwencai_tools.screen_iwencai.ainvoke(
        {"query": "市值超过300亿", "page": 1, "limit": 20, "call_type": "normal"}
    )
    assert limits == [20, 100]
    assert result["data"]["coverage"] == {
        "matched": 646,
        "returned": 100,
        "complete": False,
    }
    assert "pagination_tip" not in result["data"]
    assert "model_guidance" not in result


@pytest.mark.asyncio
async def test_news_search_fans_out_one_query_per_entity(monkeypatch) -> None:
    seen: list[str] = []
    current = 0
    peak = 0

    async def fake_search(_skill_id, query, *, limit):
        nonlocal current, peak
        assert limit == 20
        current += 1
        peak = max(peak, current)
        seen.append(query)
        await asyncio.sleep(0.01)
        current -= 1
        name = query.split()[0]
        if name == "贵州茅台":
            return {"ok": True, "data": {"data": []}}
        return {
            "ok": True,
            "data": {"data": [{"title": name, "summary": "提到比亚迪"}]},
        }

    monkeypatch.setattr(iwencai_tools, "_run_search_skill", fake_search)
    entities = ["贵州茅台", "宁德时代", "比亚迪"]
    result = await iwencai_tools.search_iwencai_news.ainvoke(
        {
            "query": "贵州茅台、宁德时代、比亚迪 最新新闻",
            "limit": 20,
            "entities": entities,
        }
    )
    data = result["data"]
    assert peak == len(entities)
    assert seen == [
        "贵州茅台 最新新闻",
        "宁德时代 最新新闻",
        "比亚迪 最新新闻",
    ]
    assert set(data["groups"]) == {"宁德时代", "比亚迪"}
    assert data["groups"]["宁德时代"][0]["summary"] == "提到比亚迪"
    assert data["missing"] == ["贵州茅台"]
    assert len(data["groups"]) + len(data["missing"]) == len(entities)


def test_shared_search_condition_strips_longer_entity_first() -> None:
    entities = ["宁德时代新能源", "宁德时代"]
    condition = iwencai_tools._shared_search_condition(
        "宁德时代新能源、宁德时代 最新公告",
        entities,
    )
    assert condition == "最新公告"
    assert iwencai_tools._entity_search_query(condition, "宁德时代新能源") == (
        "宁德时代新能源 最新公告"
    )


@pytest.mark.asyncio
async def test_news_without_entities_stays_one_query(monkeypatch) -> None:
    seen: list[str] = []

    async def fake_search(_skill_id, query, *, limit):
        seen.append(query)
        return {"ok": True, "data": {"data": []}}

    monkeypatch.setattr(iwencai_tools, "_run_search_skill", fake_search)
    result = await iwencai_tools.search_iwencai_news.ainvoke(
        {"query": "新能源政策", "limit": 10}
    )
    assert seen == ["新能源政策"]
    assert "groups" not in result.get("data", {})


def test_news_tool_schema_accepts_entities() -> None:
    properties = iwencai_tools.search_iwencai_news.args_schema.model_json_schema()[
        "properties"
    ]
    assert "entities" in properties


def test_compose_iwencai_query_batches_entities_like_sql_in():
    assert iwencai_tools.compose_iwencai_query(
        "2026半年报净利润 毛利率 光纤光缆收入",
        ["永鼎股份", "中天科技", "亨通光电"],
    ) == "永鼎股份、中天科技、亨通光电 2026半年报净利润 毛利率 光纤光缆收入"
    assert iwencai_tools.compose_iwencai_query(
        "永鼎股份、中天科技、亨通光电 净利润",
        ["永鼎股份", "中天科技", "亨通光电"],
    ) == "永鼎股份、中天科技、亨通光电 净利润"


@pytest.mark.asyncio
async def test_finance_tool_sends_one_batched_query(monkeypatch) -> None:
    seen: list[str] = []

    async def fake_cached(skill_id, *, query, **kwargs):
        seen.append(query)
        return {"ok": True, "data": {"datas": []}}

    monkeypatch.setattr(iwencai_tools, "_run_cached_skill", fake_cached)
    await iwencai_tools.query_iwencai_finance.ainvoke(
        {
            "query": "2026半年报净利润 毛利率",
            "entities": ["永鼎股份", "中天科技", "亨通光电"],
            "page": 1,
            "limit": 10,
            "call_type": "normal",
        }
    )
    assert seen == ["永鼎股份、中天科技、亨通光电 2026半年报净利润 毛利率"]


@pytest.mark.asyncio
async def test_retry_bypasses_cache(monkeypatch) -> None:
    reset_cache_metrics()
    monkeypatch.setattr(iwencai_tools.settings, "IWENCAI_CACHE_ENABLED", True)
    executed = {"count": 0}

    async def fake_exec(**kwargs):
        executed["count"] += 1
        return {"ok": True, "data": {"n": executed["count"]}}

    result = await iwencai_tools._run_cached_skill(
        "hithink-market-query",
        version="1.0.0",
        query="沪深300",
        page=1,
        limit=10,
        call_type="retry",
        executor=fake_exec,
    )
    assert result["cache_status"] == "bypass"
    assert executed["count"] == 1


@pytest.mark.asyncio
async def test_only_ok_true_is_cached(monkeypatch) -> None:
    reset_cache_metrics()
    monkeypatch.setattr(iwencai_tools.settings, "IWENCAI_CACHE_ENABLED", True)
    writes: list[object] = []

    async def fake_get(*args, **kwargs):
        return None

    async def fake_set(*args, **kwargs):
        writes.append(args[1] if args else kwargs.get("result"))
        return True

    async def fake_exec(**kwargs):
        return {"ok": False, "error": "upstream"}

    monkeypatch.setattr(iwencai_tools, "cache_get", fake_get)
    monkeypatch.setattr(iwencai_tools, "cache_set", fake_set)
    result = await iwencai_tools._run_cached_skill(
        "hithink-market-query",
        version="1.0.0",
        query="茅台",
        page=1,
        limit=10,
        call_type="normal",
        executor=fake_exec,
    )
    assert result["ok"] is False
    assert writes == []
