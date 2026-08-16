"""LangGraph checkpoint 重建已废弃。"""


class CheckpointRebuildService:
    @staticmethod
    async def rebuild_if_missing(**_kwargs) -> bool:
        return False
