"""Skill Runner 的边界行为测试。"""

import pytest

from app.core.config import settings
from skills.runners.iwencai import run_installed_skill


@pytest.mark.asyncio
async def test_skill_runner_can_be_disabled(monkeypatch):
    monkeypatch.setattr(settings, "IWENCAI_SKILL_RUNNER_ENABLED", False)

    result = await run_installed_skill(
        "hithink-astock-selector",
        query="新能源股票",
    )

    assert result == {"ok": False, "error": "skill_runner_disabled"}


@pytest.mark.asyncio
async def test_skill_runner_does_not_spawn_without_api_key(monkeypatch):
    monkeypatch.setattr(settings, "IWENCAI_SKILL_RUNNER_ENABLED", True)
    monkeypatch.setattr(settings, "IWENCAI_API_KEY", "")

    result = await run_installed_skill(
        "hithink-astock-selector",
        query="新能源股票",
    )

    assert result["error"] == "iwencai_not_configured"


@pytest.mark.asyncio
async def test_skill_runner_rejects_unknown_skill(monkeypatch):
    monkeypatch.setattr(settings, "IWENCAI_SKILL_RUNNER_ENABLED", True)
    monkeypatch.setattr(settings, "IWENCAI_API_KEY", "configured-for-test")

    result = await run_installed_skill("unknown", query="新能源股票")

    assert result["error"] == "skill_not_allowed:unknown"
