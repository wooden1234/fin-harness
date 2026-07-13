"""初始化 LangGraph Postgres checkpoint 表（首次部署执行一次）。

用法:
    source .venv/bin/activate
    python scripts/setup_langgraph_checkpoint.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from app.agents.checkpoint import checkpoint_dsn, close_checkpoint, init_checkpoint  # noqa: E402


async def main() -> None:
    dsn = checkpoint_dsn()
    print(f"checkpoint DSN = {dsn.split('@')[-1]}")  # 不打印密码
    await init_checkpoint(backend="postgres")
    print("PostgresSaver.setup() 完成")
    await close_checkpoint()


if __name__ == "__main__":
    asyncio.run(main())
