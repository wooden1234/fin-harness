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
BACKEND_DIR = ROOT / "app" / "backend"
for path in (str(BACKEND_DIR), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)
load_dotenv(ROOT / ".env")

from agents.checkpoint import checkpoint_dsn, close_checkpoint, init_checkpoint  # noqa: E402


async def main() -> None:
    dsn = checkpoint_dsn()
    print(f"checkpoint DSN = {dsn.split('@')[-1]}")  # 不打印密码
    await init_checkpoint(backend="postgres")
    print("PostgresSaver.setup() 完成")
    await close_checkpoint()


if __name__ == "__main__":
    asyncio.run(main())
