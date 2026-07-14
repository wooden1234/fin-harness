"""从 PGVector 全量重建 Elasticsearch BM25 索引（运维入口）。

日常 ingest / --rebuild-index 会双写 ES；本脚本用于：
- 仅重建 ES（PG 已就绪）
- --check-es / --list-tables / --dry-run 排查
"""

from __future__ import annotations

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if sys.path and os.path.abspath(sys.path[0]) == _SCRIPT_DIR:
    sys.path.pop(0)

import argparse
from collections.abc import Iterator
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "app" / "backend"
for path in (ROOT_DIR, BACKEND_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import psycopg  # noqa: E402
from psycopg import sql  # noqa: E402

from retrieval.es_client import create_es_client, index_name  # noqa: E402
from retrieval.es_index import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_SEARCH_ANALYZER,
    DEFAULT_TEXT_ANALYZER,
    bulk_index_documents,
    build_document,
    coerce_metadata,
    selected_collections,
)
from retrieval.index import VECTOR_SCHEMA, _pg_connection_strings  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild Elasticsearch BM25 indexes from PGVector text tables",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        help="只重建指定 category；默认全部 collection",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="ES bulk 单批写入文档数",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="删除并重建目标 ES 索引（全量重建建议打开）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只统计 PG 文本表行数，不写入 ES",
    )
    parser.add_argument(
        "--list-tables",
        action="store_true",
        help=f"列出当前 PG 连接可见的 {VECTOR_SCHEMA}.data_* 表",
    )
    parser.add_argument(
        "--check-es",
        action="store_true",
        help="检查 Elasticsearch 连接和 analyzer 是否可用",
    )
    parser.add_argument(
        "--text-analyzer",
        default=DEFAULT_TEXT_ANALYZER,
        help="text/title/section 字段的索引 analyzer",
    )
    parser.add_argument(
        "--search-analyzer",
        default=DEFAULT_SEARCH_ANALYZER,
        help="text/title/section 字段的查询 analyzer",
    )
    return parser.parse_args()


def check_es(es: Any, *, text_analyzer: str) -> None:
    info = es.info()
    version = info.get("version", {}).get("number", "unknown")
    print(f"Elasticsearch: {info.get('name', 'unknown')} version={version}")
    analyzed = es.indices.analyze(
        body={
            "analyzer": text_analyzer,
            "text": "宁德时代2024年营业收入是多少",
        }
    )
    tokens = [token["token"] for token in analyzed.get("tokens", [])[:12]]
    print(f"Analyzer {text_analyzer}: tokens={tokens}")


def load_rows(category: str, table_name: str) -> Iterator[dict[str, Any]]:
    sync_url, _ = _pg_connection_strings()
    physical_table = f"data_{table_name}"
    with psycopg.connect(sync_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT node_id, text, metadata_ FROM {}.{}").format(
                    sql.Identifier(VECTOR_SCHEMA),
                    sql.Identifier(physical_table),
                )
            )
            for node_id, text, metadata in cur:
                meta = coerce_metadata(metadata)
                yield build_document(
                    category=category,
                    collection=table_name,
                    node_id=str(node_id),
                    text=text or "",
                    metadata=meta,
                )


def count_rows(category: str, table_name: str) -> int:
    sync_url, _ = _pg_connection_strings()
    physical_table = f"data_{table_name}"
    with psycopg.connect(sync_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT to_regclass(%s)",
                (f"{VECTOR_SCHEMA}.{physical_table}",),
            )
            if cur.fetchone()[0] is None:
                raise psycopg.errors.UndefinedTable(physical_table)
            cur.execute(
                sql.SQL("SELECT count(*) FROM {}.{}").format(
                    sql.Identifier(VECTOR_SCHEMA),
                    sql.Identifier(physical_table),
                )
            )
            return int(cur.fetchone()[0])


def table_label(table_name: str) -> str:
    return f"{VECTOR_SCHEMA}.data_{table_name}"


def visible_data_tables() -> list[tuple[str, int | None]]:
    sync_url, _ = _pg_connection_strings()
    with psycopg.connect(sync_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = %s
                  AND table_name LIKE %s
                ORDER BY table_name
                """,
                (VECTOR_SCHEMA, "data_%"),
            )
            table_names = [row[0] for row in cur.fetchall()]

            results: list[tuple[str, int | None]] = []
            for name in table_names:
                cur.execute(
                    sql.SQL("SELECT count(*) FROM {}.{}").format(
                        sql.Identifier(VECTOR_SCHEMA),
                        sql.Identifier(name),
                    )
                )
                results.append((f"{VECTOR_SCHEMA}.{name}", int(cur.fetchone()[0])))
            return results


def pg_connection_label() -> str:
    sync_url, _ = _pg_connection_strings()
    parsed = psycopg.conninfo.conninfo_to_dict(sync_url)
    parts = []
    for key in ("host", "port", "dbname", "user"):
        value = parsed.get(key)
        if value:
            parts.append(f"{key}={value}")
    return ", ".join(parts) or "unknown"


def main() -> None:
    args = parse_args()
    if args.check_es:
        check_es(create_es_client(), text_analyzer=args.text_analyzer)
        return

    if args.list_tables:
        print(f"PG connection: {pg_connection_label()}")
        tables = visible_data_tables()
        if not tables:
            print(f"未发现 {VECTOR_SCHEMA}.data_* 表")
            return
        for table_name, rows in tables:
            print(f"{table_name}: rows={rows}")
        return

    collections = selected_collections(args.categories)
    es = None if args.dry_run else create_es_client()

    for category, table_name in collections.items():
        name = index_name(category)
        try:
            total = count_rows(category, table_name)
        except psycopg.errors.UndefinedTable:
            print(f"{category}: table={table_label(table_name)}, missing, skipped")
            continue
        except psycopg.OperationalError as exc:
            raise SystemExit(
                "无法连接 PGVector 数据库，请检查 PGVECTOR_DATABASE_URL / "
                "DATABASE_URL，并确认 postgres 服务已启动"
            ) from exc
        if args.dry_run:
            print(f"{category}: table={table_label(table_name)}, rows={total}")
            continue

        assert es is not None
        success, error_count = bulk_index_documents(
            load_rows(category, table_name),
            category=category,
            rebuild=args.rebuild,
            batch_size=args.batch_size,
            text_analyzer=args.text_analyzer,
            search_analyzer=args.search_analyzer,
            es=es,
        )
        if error_count:
            print(
                f"{category}: indexed={success}, errors={error_count}, index={name}"
            )
        else:
            print(f"{category}: indexed={success}, rows={total}, index={name}")


if __name__ == "__main__":
    main()
