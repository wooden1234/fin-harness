"""启动 Agent 消息 outbox 重试 worker。"""

import asyncio
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.services.outbox_service import OutboxService  # noqa: E402


if __name__ == "__main__":
    asyncio.run(OutboxService.run_forever())

