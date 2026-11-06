"""每个登录用户在每家企业保留一条未完成的发票用途任务。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from retrieval.services.ledger_store import _connection

_ENTERPRISE_NAMES = {"C001": "甲企业", "C002": "乙企业"}


def list_open_tasks(
    app_user_id: int,
    *,
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    with _connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT enterprise_code, month, missing_nos
            FROM open_tasks
            WHERE app_user_id = ?
            ORDER BY enterprise_code
            """,
            (int(app_user_id),),
        ).fetchall()
    tasks = []
    for row in rows:
        code = str(row["enterprise_code"])
        tasks.append(
            {
                "enterprise": code,
                "enterprise_name": _ENTERPRISE_NAMES.get(code, code),
                "month": str(row["month"]),
                "missing": _load_missing(row["missing_nos"]),
            }
        )
    return tasks


def open_task_prompt(app_user_id: int, *, db_path: Path | None = None) -> str:
    tasks = list_open_tasks(app_user_id, db_path=db_path)
    if not tasks:
        return ""
    lines = [
        "[未完成的账房任务]",
        "这是当前登录用户留下的发票用途任务，每家企业最多一条。",
        "只有用户要接着补这些用途时，才使用这里的企业和月份，并先说明它们。",
        "用户新问金额，但只说「这个月」或「本月」、前文也没有写明哪一月时，先追问月份。不要用这里的月份去查询。",
        "回答缺哪些用途之前，必须重新调用 ledger_records 的 check_purpose，以当前发票为准。",
    ]
    for task in tasks:
        missing = "、".join(task["missing"])
        lines.append(
            f"- {task['enterprise_name']}（{task['enterprise']}）{task['month']} 仍缺用途：{missing}"
        )
    return "\n".join(lines)


def record_open_tasks_from_events(
    app_user_id: int,
    events: Sequence[Any],
    *,
    db_path: Path | None = None,
) -> None:
    turns = [event.turn for event in events if getattr(event, "turn", None)]
    if not turns:
        return
    latest = max(turns)
    payloads: list[dict[str, Any]] = []
    for event in events:
        if getattr(event, "turn", None) != latest or getattr(event, "event_type", "") != "tool/result":
            continue
        data = getattr(event, "data", None) or {}
        if str(data.get("name") or "") != "ledger_records" or data.get("ok") is not True:
            continue
        payloads.extend(_payloads_from_content(data.get("content")))
    if payloads:
        apply_open_task_payloads(app_user_id, payloads, db_path=db_path)


def apply_open_task_payloads(
    app_user_id: int,
    payloads: Sequence[Mapping[str, Any]],
    *,
    db_path: Path | None = None,
) -> None:
    for item in payloads:
        action = str(item.get("action") or "")
        enterprise = str(item.get("enterprise") or "").strip()
        if not enterprise:
            continue
        if action == "check_purpose":
            month = str(item.get("month") or "").strip()
            if not month:
                continue
            missing = [
                str(row.get("invoice_no") or "").strip()
                for row in item.get("missing") or []
                if isinstance(row, dict) and str(row.get("invoice_no") or "").strip()
            ]
            if missing:
                _save_task(app_user_id, enterprise, month, missing, db_path=db_path)
            else:
                _delete_task(app_user_id, enterprise, db_path=db_path)
        elif action == "update_invoice":
            invoice = item.get("invoice") if isinstance(item.get("invoice"), dict) else {}
            purpose = str(invoice.get("purpose") or "").strip()
            number = str(invoice.get("invoice_no") or "").strip()
            month = str(invoice.get("issued_on") or "")[:7]
            if purpose and number:
                _drop_invoice(
                    app_user_id,
                    enterprise,
                    number,
                    month=month,
                    db_path=db_path,
                )


def _payloads_from_content(content: object) -> list[dict[str, Any]]:
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return []
    elif isinstance(content, dict):
        parsed = content
    else:
        return []
    if not isinstance(parsed, dict) or parsed.get("ok") is False:
        return []
    nested = parsed.get("results") if parsed.get("batched") else None
    if isinstance(nested, list):
        return [item for item in nested if isinstance(item, dict) and item.get("ok") is not False]
    return [parsed]


def _save_task(
    app_user_id: int,
    enterprise: str,
    month: str,
    missing: list[str],
    *,
    db_path: Path | None,
) -> None:
    with _connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO open_tasks (app_user_id, enterprise_code, month, missing_nos, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(app_user_id, enterprise_code) DO UPDATE SET
                month = excluded.month,
                missing_nos = excluded.missing_nos,
                updated_at = excluded.updated_at
            """,
            (
                int(app_user_id),
                enterprise,
                month,
                json.dumps(missing, ensure_ascii=False),
                _now(),
            ),
        )


def _delete_task(app_user_id: int, enterprise: str, *, db_path: Path | None) -> None:
    with _connection(db_path) as conn:
        conn.execute(
            "DELETE FROM open_tasks WHERE app_user_id = ? AND enterprise_code = ?",
            (int(app_user_id), enterprise),
        )


def _drop_invoice(
    app_user_id: int,
    enterprise: str,
    invoice_no: str,
    *,
    month: str,
    db_path: Path | None,
) -> None:
    with _connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT month, missing_nos
            FROM open_tasks
            WHERE app_user_id = ? AND enterprise_code = ?
            """,
            (int(app_user_id), enterprise),
        ).fetchone()
        if row is None:
            return
        if month and str(row["month"]) != month:
            return
        remaining = [item for item in _load_missing(row["missing_nos"]) if item != invoice_no]
        if not remaining:
            conn.execute(
                "DELETE FROM open_tasks WHERE app_user_id = ? AND enterprise_code = ?",
                (int(app_user_id), enterprise),
            )
            return
        conn.execute(
            """
            UPDATE open_tasks
            SET missing_nos = ?, updated_at = ?
            WHERE app_user_id = ? AND enterprise_code = ?
            """,
            (json.dumps(remaining, ensure_ascii=False), _now(), int(app_user_id), enterprise),
        )


def _load_missing(raw: object) -> list[str]:
    try:
        value = json.loads(str(raw or "[]"))
    except json.JSONDecodeError:
        return []
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
