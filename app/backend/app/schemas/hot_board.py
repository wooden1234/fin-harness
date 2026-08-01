"""首页今日热榜响应契约。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


HotBoardPanelId = Literal["hot_discuss", "finance_lookup", "market_view"]


class HotBoardPanel(BaseModel):
    id: HotBoardPanelId
    title: str
    items: list[str] = Field(default_factory=list, min_length=1, max_length=5)


class HotBoardResponse(BaseModel):
    as_of: str
    source: Literal["web", "fallback"]
    panels: list[HotBoardPanel]


__all__ = ["HotBoardPanel", "HotBoardPanelId", "HotBoardResponse"]
