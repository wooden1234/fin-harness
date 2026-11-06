"""账房规则知识库：正文来自 Markdown，筛选只用元数据列。"""

from __future__ import annotations

from pathlib import Path

from retrieval.services.ledger_rules import LedgerRule, load_rules, normalize_enterprise, normalize_month


def sync_rule_knowledge(root: Path | None = None) -> int:
    """用规则 Markdown 重建 ledger.rules。停用版本一并入库，默认查询再按状态滤掉。"""
    from retrieval.services.ledger_store import _connection

    rules = load_rules(root)
    with _connection(None) as conn:
        conn.execute("DELETE FROM rules")
        conn.executemany(
            """
            INSERT INTO rules (
                rule_id, version, name, enterprises, effective_from, status, topic,
                purpose_required, purpose_threshold, body, source_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    rule.rule_id,
                    rule.version,
                    rule.name,
                    list(rule.enterprises),
                    rule.effective_from,
                    rule.status,
                    rule.topic,
                    rule.purpose_required,
                    rule.purpose_threshold,
                    rule.body,
                    str(rule.path),
                )
                for rule in rules
            ],
        )
    return len(rules)


def search_rule_kb(
    *,
    enterprise: str,
    month: str,
    topic: str | None = None,
    include_disabled: bool = False,
) -> list[LedgerRule]:
    """按企业、生效月份、状态和主题过滤。同一编号只保留生效日最新的启用版。"""
    from retrieval.services.ledger_store import _connection

    code = normalize_enterprise(enterprise)
    month_key = normalize_month(month)
    if code is None or month_key is None:
        return []
    topic_key = str(topic or "").strip()
    if include_disabled:
        sql = """
            SELECT rule_id, version, name, enterprises, effective_from, status, topic,
                   purpose_required, purpose_threshold, body, source_path
            FROM rules
            WHERE ? = ANY(enterprises)
              AND left(effective_from, 7) <= ?
              AND (? = '' OR topic = ?)
            ORDER BY rule_id, effective_from, version
        """
    else:
        sql = """
            SELECT DISTINCT ON (rule_id)
                   rule_id, version, name, enterprises, effective_from, status, topic,
                   purpose_required, purpose_threshold, body, source_path
            FROM rules
            WHERE status = 'active'
              AND ? = ANY(enterprises)
              AND left(effective_from, 7) <= ?
              AND (? = '' OR topic = ?)
            ORDER BY rule_id, effective_from DESC
        """
    with _connection(None) as conn:
        rows = conn.execute(sql, (code, month_key, topic_key, topic_key)).fetchall()
    return [_rule_from_row(row) for row in rows]


def active_rules(
    *,
    enterprise: str,
    month: str,
    topic: str | None = None,
    include_disabled: bool = False,
    root: Path | None = None,
    db_path: Path | None = None,
) -> list[LedgerRule]:
    """PostgreSQL 知识库优先。测试指定了 SQLite 或临时目录时仍读文件。"""
    from retrieval.services.ledger_rules import search_rules
    from retrieval.services.ledger_store import _sqlite_file

    if root is None and _sqlite_file(db_path) is None:
        return search_rule_kb(
            enterprise=enterprise,
            month=month,
            topic=topic,
            include_disabled=include_disabled,
        )
    return search_rules(
        enterprise=enterprise,
        month=month,
        topic=topic,
        include_disabled=include_disabled,
        root=root,
    )


def _rule_from_row(row) -> LedgerRule:
    enterprises = row["enterprises"] or []
    return LedgerRule(
        rule_id=str(row["rule_id"]),
        name=str(row["name"]),
        version=str(row["version"]),
        enterprises=tuple(str(item) for item in enterprises),
        effective_from=str(row["effective_from"]),
        status=str(row["status"]),
        topic=str(row["topic"]),
        body=str(row["body"]),
        path=Path(str(row["source_path"])),
        purpose_required=row["purpose_required"] or None,
        purpose_threshold=row["purpose_threshold"] or None,
    )
