"""归档过期 checkpoint。"""

import asyncio
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from agents.checkpoint import close_checkpoint, init_checkpoint  # noqa: E402
from app.services.agent.checkpoint_registry_service import CheckpointRegistryService  # noqa: E402


async def main() -> None:
    await init_checkpoint()
    count = await CheckpointRegistryService.archive_expired()
    print(f"archived checkpoints: {count}")
    await close_checkpoint()


if __name__ == "__main__":
    asyncio.run(main())
