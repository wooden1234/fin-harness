"""首页今日热榜 API。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.core.security import get_current_user
from app.schemas.hot_board import HotBoardResponse
from app.schemas.user import AuthUser
from app.services.hot_board import get_hot_board

router = APIRouter(prefix="/hot-board", tags=["hot-board"])


@router.get("", response_model=HotBoardResponse)
async def fetch_hot_board(
    refresh: bool = Query(False, description="跳过缓存，强制刷新今日热榜"),
    current_user: AuthUser = Depends(get_current_user),
) -> HotBoardResponse:
    """返回今日热榜三区问题；类似微博热搜，打开页面时按需拉取。"""
    del current_user
    return await get_hot_board(force_refresh=refresh)


__all__ = ["router"]
