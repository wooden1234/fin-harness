"""天气工具调用边界日志测试。"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from tools.weather import get_weather


def test_get_weather_logs_start_and_finish() -> None:
    result = {"ok": True, "provider": "openweathermap"}
    with (
        patch("tools.weather.fetch_weather", new=AsyncMock(return_value=result)),
        patch("tools.weather.logger.info") as log_info,
    ):
        actual = asyncio.run(get_weather.ainvoke({"city": "上海"}))

    assert actual == result
    assert log_info.call_count == 2
    assert "get_weather started" in log_info.call_args_list[0].args[0]
    assert "get_weather finished" in log_info.call_args_list[1].args[0]


def test_get_weather_logs_cancellation() -> None:
    with (
        patch(
            "tools.weather.fetch_weather",
            new=AsyncMock(side_effect=asyncio.CancelledError),
        ),
        patch("tools.weather.logger.warning") as log_warning,
        patch("tools.weather.logger.info"),
    ):
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(get_weather.ainvoke({"city": "上海"}))

    assert log_warning.call_count == 1
    assert "get_weather cancelled" in log_warning.call_args.args[0]
