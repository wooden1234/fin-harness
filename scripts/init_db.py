"""初始化 PostgreSQL 表结构（W1）。

用法:
    python scripts/init_db.py          # 仅创建缺失的表
    python scripts/init_db.py --reset  # 删表重建（开发环境，会清空数据）
"""

import argparse
import asyncio
import sys
from pathlib import Path

from sqlalchemy import text

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.core.database import Base, engine
from app.core.logger import get_logger
from app.models import (  # noqa: F401 — 注册 ORM 到 metadata
    AgentRun,
    AgentRunStatus,
    AnnualFinancialFact,
    AnnualFinancialTable,
    AnnualReportDocument,
    CanonicalMetric,
    CanonicalMetricAlias,
    CompanyMetricMapping,
    Conversation,
    ConversationLock,
    CheckpointRegistry,
    AuditLog,
    FinancialCompany,
    FinancialMetric,
    Message,
    RawTableCell,
    OutboxEvent,
    User,
    MemoryRecord,
    MemoryEvent,
)

logger = get_logger(service="init_db")


async def init_db(reset: bool = False) -> None:
    try:
        logger.info("Connecting to database...")
        async with engine.begin() as conn:
            for schema in ("app", "fin_core", "rag", "runtime"):
                await conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
            if reset:
                logger.warning("Dropping all tables...")
                await conn.run_sync(Base.metadata.drop_all)
            logger.info("Creating application and normalized financial fact tables")
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database initialization completed.")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize fin-agent-platform database tables")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop all tables before create (destructive)",
    )
    args = parser.parse_args()
    asyncio.run(init_db(reset=args.reset))


if __name__ == "__main__":
    main()
