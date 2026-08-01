"""今日热榜：按需拉取财经热点并归入首页三区。"""

from app.services.hot_board.hot_board_service import get_hot_board

__all__ = ["get_hot_board"]
