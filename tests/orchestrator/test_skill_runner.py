"""Skill Runner 的边界行为测试。"""

from pathlib import Path

import pytest

from app.core.config import settings
from skills.runners.iwencai import _command_args, run_installed_skill


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


def test_skill_runner_allows_finance_and_usstock_skills():
    from skills.runners.iwencai import installed_skill_version

    assert installed_skill_version("hithink-finance-query") == "1.0.0"
    assert installed_skill_version("hithink-usstock-selector") == "1.0.0"


def test_skill_runner_builds_query_skill_arguments():
    args = _command_args(
        "hithink-industry-query",
        Path("/skills/industry/scripts/cli.py"),
        query="行业估值排名",
        page=2,
        limit=20,
        call_type="retry",
    )

    assert args[1:] == [
        "--query",
        "行业估值排名",
        "--page",
        "2",
        "--limit",
        "20",
        "--call-type",
        "retry",
        "--timeout",
        "30",
    ]


def test_skill_runner_builds_search_skill_arguments():
    args = _command_args(
        "report-search",
        Path("/skills/report/scripts/report_search.py"),
        query="新能源研报",
        page=1,
        limit=8,
        call_type="normal",
    )

    assert args[1:] == [
        "新能源研报",
        "--size",
        "8",
        "--timeout",
        "30",
    ]
