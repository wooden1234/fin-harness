"""iwencai 统一只读缓存入口测试。"""

from __future__ import annotations

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
