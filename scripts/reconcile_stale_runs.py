"""清理长期无心跳的 Agent 运行记录。"""

import asyncio
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.services.agent_run_service import AgentRunService


async def main() -> None:
    count = await AgentRunService.reconcile_stale_runs()
    print(f"reconciled stale runs: {count}")


if __name__ == "__main__":
    asyncio.run(main())
