"""执行仓库内 SQL migration。用法：python scripts/migrate.py scripts/migrations/001_tenant_and_memory.sql"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "app" / "backend"))

from app.core.database import engine  # noqa: E402


def split_sql_statements(sql: str) -> list[str]:
    """按分号拆分 SQL，同时忽略字符串和行注释中的分号。"""
    statements: list[str] = []
    current: list[str] = []
    quote: str | None = None
    line_comment = False
    index = 0

    def append_statement(buffer: list[str]) -> None:
        statement = "".join(buffer).strip()
        if not statement:
            return
        # 忽略纯注释，但保留“注释 + SQL”组成的语句。
        has_sql = any(
            line.strip() and not line.lstrip().startswith("--")
            for line in statement.splitlines()
        )
        if has_sql:
            statements.append(statement)

    while index < len(sql):
        char = sql[index]
        next_char = sql[index + 1] if index + 1 < len(sql) else ""

        if line_comment:
            current.append(char)
            if char == "\n":
                line_comment = False
            index += 1
            continue

        if quote is None and char == "-" and next_char == "-":
            current.extend((char, next_char))
            line_comment = True
            index += 2
            continue

        if char in {"'", '"'}:
            if quote == char:
                # SQL 使用两个连续引号表示转义引号。
                if next_char == char:
                    current.extend((char, next_char))
                    index += 2
                    continue
                quote = None
            elif quote is None:
                quote = char
            current.append(char)
            index += 1
            continue

        if char == ";" and quote is None:
            append_statement(current)
            current = []
        else:
            current.append(char)
        index += 1

    append_statement(current)
    return statements


async def apply_migration(path: Path) -> None:
    sql = path.read_text(encoding="utf-8")
    async with engine.begin() as connection:
        for statement in split_sql_statements(sql):
            await connection.exec_driver_sql(statement)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("用法：python scripts/migrate.py scripts/migrations/<file>.sql")
    asyncio.run(apply_migration(Path(sys.argv[1]).resolve()))


if __name__ == "__main__":
    main()
