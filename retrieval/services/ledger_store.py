"""账房业务数据：发票与银行流水存在独立 SQLite，和原始 Excel 分开。"""

from __future__ import annotations

from contextlib import contextmanager
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import os
import re
import sqlite3
import threading
from typing import Iterator

from retrieval.services.ledger_rules import (
    normalize_enterprise,
    normalize_month,
)

_USER_CODE = re.compile(r"^U\d+$")
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DB = _PROJECT_ROOT / "data" / "ledger" / "finance.sqlite"
_LOCKS: dict[str, threading.Lock] = {}
_SCHEMA = """
CREATE TABLE IF NOT EXISTS companies (
    code TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    short_name TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS users (
    code TEXT PRIMARY KEY,
    name TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS permissions (
    user_code TEXT NOT NULL,
    enterprise_code TEXT NOT NULL,
    can_read INTEGER NOT NULL,
    can_write INTEGER NOT NULL,
    PRIMARY KEY (user_code, enterprise_code)
);
CREATE TABLE IF NOT EXISTS invoices (
    enterprise_code TEXT NOT NULL,
    invoice_no TEXT NOT NULL,
    issued_on TEXT NOT NULL,
    seller TEXT NOT NULL,
    item_name TEXT NOT NULL,
    amount_cents INTEGER NOT NULL,
    purpose TEXT NOT NULL DEFAULT '',
    remark TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (enterprise_code, invoice_no)
);
CREATE TABLE IF NOT EXISTS bank_txns (
    enterprise_code TEXT NOT NULL,
    txn_no TEXT NOT NULL,
    txn_date TEXT NOT NULL,
    direction TEXT NOT NULL,
    amount_cents INTEGER NOT NULL,
    counterparty TEXT NOT NULL,
    remark TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (enterprise_code, txn_no)
);
CREATE TABLE IF NOT EXISTS session_scope (
    session_id TEXT PRIMARY KEY,
    user_code TEXT NOT NULL,
    enterprise_code TEXT
);
CREATE TABLE IF NOT EXISTS open_tasks (
    app_user_id INTEGER NOT NULL,
    enterprise_code TEXT NOT NULL,
    month TEXT NOT NULL,
    missing_nos TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (app_user_id, enterprise_code)
);
"""


class LedgerFailure(Exception):
    def __init__(self, code: str, **extra: object) -> None:
        super().__init__(code)
        self.code = code
        self.extra = extra


def ledger_db_path() -> Path:
    override = os.environ.get("LEDGER_DB_PATH", "").strip()
    if override:
        return Path(override)
    return _DEFAULT_DB


def to_cents(value: object) -> int:
    amount = Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return int(amount * 100)


def from_cents(cents: int) -> str:
    return f"{Decimal(int(cents)) / Decimal(100):.2f}"


def is_blank(value: object) -> bool:
    return not str(value or "").strip()


def bind_session_ledger_code(
    session_id: str,
    ledger_code: str | None,
    *,
    db_path: Path | None = None,
) -> str | None:
    """按登录账户上的权限身份绑定会话。空值或未知身份不绑定。"""
    code = str(ledger_code or "").strip().upper()
    if code not in {"U001", "U002"}:
        return None
    current = get_actor(session_id, db_path=db_path)
    if current is not None and current[0] == code:
        return code
    enterprise = current[1] if current is not None else None
    bind_actor(session_id, code, enterprise_code=enterprise, db_path=db_path)
    return code


def bind_account_ledger_session(
    session_id: str,
    user_id: int,
    *,
    db_path: Path | None = None,
) -> str | None:
    """用 app.users.ledger_code 绑定当前会话。"""
    return bind_session_ledger_code(
        session_id,
        _app_user_ledger_code(user_id, db_path=db_path),
        db_path=db_path,
    )


def _app_user_ledger_code(user_id: int, *, db_path: Path | None = None) -> str | None:
    if _sqlite_file(db_path) is not None:
        return None
    # 不在这里 ALTER app.users。聊天请求的登录事务会一直占着这张表的共享锁，
    # ADD COLUMN 需要排他锁，会等到整段回复结束，形成互相等待。列由 init_db 补齐。
    with _connection(db_path) as conn:
        row = conn.execute(
            "SELECT ledger_code FROM app.users WHERE id = ?",
            (int(user_id),),
        ).fetchone()
    if row is None or not row["ledger_code"]:
        return None
    return str(row["ledger_code"])


def _ensure_app_ledger_code_column() -> None:
    import psycopg

    with psycopg.connect(_postgres_dsn()) as conn:
        conn.execute(
            "ALTER TABLE app.users ADD COLUMN IF NOT EXISTS ledger_code VARCHAR(16)"
        )
        conn.commit()


def ensure_session_actor(
    session_id: str,
    *,
    user_code: str = "U001",
    db_path: Path | None = None,
) -> tuple[str, str | None]:
    """对话开始时补上演示身份。已经绑定的会话保持原用户。"""
    current = get_actor(session_id, db_path=db_path)
    if current is not None:
        return current
    bind_actor(session_id, user_code, db_path=db_path)
    bound = get_actor(session_id, db_path=db_path)
    if bound is None:
        raise LedgerFailure("ledger_actor_unbound")
    return bound


def bind_actor(
    session_id: str,
    user_code: str,
    *,
    enterprise_code: str | None = None,
    db_path: Path | None = None,
) -> None:
    user = str(user_code or "").strip().upper()
    if user not in {"U001", "U002"}:
        raise ValueError("unknown_user")
    enterprise = normalize_enterprise(enterprise_code) if enterprise_code else None
    if enterprise_code and enterprise is None:
        raise ValueError("unknown_enterprise")
    with _connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO session_scope (session_id, user_code, enterprise_code)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                user_code = excluded.user_code,
                enterprise_code = excluded.enterprise_code
            """,
            (session_id, user, enterprise),
        )


def read_scope(
    session_id: str,
    enterprise: str | None,
    month: str | None,
    *,
    db_path: Path | None = None,
) -> tuple[str, str, str]:
    """确认会话用户对某企业有查看权限，并返回用户、企业编号和月份。"""
    user, code, month_key = _scope(
        session_id,
        enterprise,
        month,
        db_path=db_path,
        need_month=True,
    )
    _require_read(db_path, user, code)
    assert month_key is not None
    return user, code, month_key


def get_actor(
    session_id: str,
    *,
    db_path: Path | None = None,
) -> tuple[str, str | None] | None:
    sqlite_file = _sqlite_file(db_path)
    if sqlite_file is not None and not sqlite_file.exists():
        return None
    with _connection(db_path) as conn:
        row = conn.execute(
            "SELECT user_code, enterprise_code FROM session_scope WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    if row is None:
        return None
    return str(row["user_code"]), (str(row["enterprise_code"]) if row["enterprise_code"] else None)


def query_invoices(
    *,
    session_id: str,
    enterprise: str | None,
    month: str | None,
    db_path: Path | None = None,
) -> dict[str, object]:
    user, code, month_key = _scope(session_id, enterprise, month, db_path=db_path, need_month=True)
    _require_read(db_path, user, code)
    with _connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT invoice_no, issued_on, amount_cents, purpose, remark
            FROM invoices
            WHERE enterprise_code = ? AND substr(issued_on, 1, 7) = ?
            ORDER BY issued_on, invoice_no
            """,
            (code, month_key),
        ).fetchall()
    total = sum(int(row["amount_cents"]) for row in rows)
    invoices = [_invoice_dict(row) for row in rows]
    return {
        "ok": True,
        "action": "query_invoices",
        "user": user,
        "enterprise": code,
        "month": month_key,
        "count": len(invoices),
        "total": from_cents(total),
        "empty": not invoices,
        "invoices": invoices,
    }


def query_bank(
    *,
    session_id: str,
    enterprise: str | None,
    month: str | None,
    db_path: Path | None = None,
) -> dict[str, object]:
    user, code, month_key = _scope(session_id, enterprise, month, db_path=db_path, need_month=True)
    _require_read(db_path, user, code)
    income = 0
    expense = 0
    count = 0
    with _connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT direction, amount_cents
            FROM bank_txns
            WHERE enterprise_code = ? AND substr(txn_date, 1, 7) = ?
            """,
            (code, month_key),
        ).fetchall()
    for row in rows:
        count += 1
        cents = int(row["amount_cents"])
        if row["direction"] == "收入":
            income += cents
        elif row["direction"] == "支出":
            expense += cents
    return {
        "ok": True,
        "action": "query_bank",
        "user": user,
        "enterprise": code,
        "month": month_key,
        "count": count,
        "income": from_cents(income),
        "expense": from_cents(expense),
        "empty": count == 0,
    }


def check_purpose(
    *,
    session_id: str,
    enterprise: str | None,
    month: str | None,
    db_path: Path | None = None,
    rules_root: Path | None = None,
) -> dict[str, object]:
    user, code, month_key = _scope(session_id, enterprise, month, db_path=db_path, need_month=True)
    _require_read(db_path, user, code)
    from retrieval.services.ledger_rule_kb import active_rules

    rules = active_rules(
        enterprise=code,
        month=month_key,
        topic="invoice_purpose",
        root=rules_root,
        db_path=db_path,
    )
    if len(rules) != 1 or not rules[0].purpose_required:
        raise LedgerFailure("ledger_rule_missing", enterprise=code, month=month_key)
    rule = rules[0]
    with _connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT invoice_no, issued_on, amount_cents, purpose, remark
            FROM invoices
            WHERE enterprise_code = ? AND substr(issued_on, 1, 7) = ?
            ORDER BY issued_on, invoice_no
            """,
            (code, month_key),
        ).fetchall()
    missing = [
        _invoice_dict(row)
        for row in rows
        if _purpose_missing(row, rule.purpose_required, rule.purpose_threshold)
    ]
    return {
        "ok": True,
        "action": "check_purpose",
        "user": user,
        "enterprise": code,
        "month": month_key,
        "rule": {
            "rule_id": rule.rule_id,
            "name": rule.name,
            "version": rule.version,
        },
        "missing": missing,
    }


def update_invoice(
    *,
    session_id: str,
    enterprise: str | None,
    invoice_no: str | None,
    purpose: object = None,
    remark: object = None,
    purpose_supplied: bool,
    remark_supplied: bool,
    db_path: Path | None = None,
) -> dict[str, object]:
    user, code, _month = _scope(session_id, enterprise, None, db_path=db_path, need_month=False)
    _require_write(db_path, user, code)
    number = str(invoice_no or "").strip().upper()
    if not number:
        raise LedgerFailure("ledger_not_found", enterprise=code)
    if not purpose_supplied and not remark_supplied:
        raise LedgerFailure("ledger_update_required", enterprise=code, invoice_no=number)
    assignments: list[str] = []
    params: list[object] = []
    if purpose_supplied:
        text = str(purpose or "").strip()
        if not text:
            raise LedgerFailure("ledger_empty_purpose", enterprise=code, invoice_no=number)
        assignments.append("purpose = ?")
        params.append(text)
    if remark_supplied:
        assignments.append("remark = ?")
        params.append(str(remark or "").strip())
    with _connection(db_path) as conn:
        existing = conn.execute(
            """
            SELECT invoice_no, issued_on, amount_cents, purpose, remark
            FROM invoices
            WHERE enterprise_code = ? AND invoice_no = ?
            """,
            (code, number),
        ).fetchone()
        if existing is None:
            raise LedgerFailure("ledger_not_found", enterprise=code, invoice_no=number)
        conn.execute(
            f"UPDATE invoices SET {', '.join(assignments)} WHERE enterprise_code = ? AND invoice_no = ?",
            (*params, code, number),
        )
        saved = conn.execute(
            """
            SELECT invoice_no, issued_on, amount_cents, purpose, remark
            FROM invoices
            WHERE enterprise_code = ? AND invoice_no = ?
            """,
            (code, number),
        ).fetchone()
    assert saved is not None
    return {
        "ok": True,
        "action": "update_invoice",
        "user": user,
        "enterprise": code,
        "invoice": _invoice_dict(saved),
    }


def import_excel(
    xlsx_path: Path,
    *,
    db_path: Path | None = None,
    reset: bool = False,
) -> dict[str, int]:
    from openpyxl import load_workbook

    sqlite_file = _sqlite_file(db_path)
    if reset and sqlite_file is not None and sqlite_file.exists():
        sqlite_file.unlink()
    preserved: dict[tuple[str, str], tuple[str, str]] = {}
    source_ready = sqlite_file is None or sqlite_file.exists()
    if source_ready and not reset:
        with _connection(db_path) as conn:
            for row in conn.execute(
                "SELECT enterprise_code, invoice_no, purpose, remark FROM invoices"
            ):
                preserved[(row["enterprise_code"], row["invoice_no"])] = (
                    row["purpose"],
                    row["remark"],
                )
    workbook = load_workbook(xlsx_path, data_only=True)
    companies, users, permissions = _read_parties(workbook["企业与用户"])
    invoices = _read_invoices(workbook["发票"])
    banks = _read_bank(workbook["银行流水"])
    with _connection(db_path) as conn:
        conn.execute("DELETE FROM companies")
        conn.execute("DELETE FROM users")
        conn.execute("DELETE FROM permissions")
        conn.execute("DELETE FROM bank_txns")
        conn.executemany(
            "INSERT INTO companies (code, name, short_name) VALUES (?, ?, ?)",
            companies,
        )
        conn.executemany("INSERT INTO users (code, name) VALUES (?, ?)", users)
        conn.executemany(
            """
            INSERT INTO permissions (user_code, enterprise_code, can_read, can_write)
            VALUES (?, ?, ?, ?)
            """,
            permissions,
        )
        for item in invoices:
            key = (item[0], item[1])
            purpose, remark = preserved.get(key, (item[6], item[7]))
            conn.execute(
                """
                INSERT INTO invoices (
                    enterprise_code, invoice_no, issued_on, seller, item_name,
                    amount_cents, purpose, remark
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(enterprise_code, invoice_no) DO UPDATE SET
                    issued_on = excluded.issued_on,
                    seller = excluded.seller,
                    item_name = excluded.item_name,
                    amount_cents = excluded.amount_cents,
                    purpose = ?,
                    remark = ?
                """,
                (*item[:6], purpose, remark, purpose, remark),
            )
        conn.executemany(
            """
            INSERT INTO bank_txns (
                enterprise_code, txn_no, txn_date, direction, amount_cents, counterparty, remark
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            banks,
        )
    return {
        "companies": len(companies),
        "users": len(users),
        "permissions": len(permissions),
        "invoices": len(invoices),
        "bank_txns": len(banks),
    }


def _scope(
    session_id: str,
    enterprise: str | None,
    month: str | None,
    *,
    db_path: Path | None,
    need_month: bool,
) -> tuple[str, str, str | None]:
    sqlite_file = _sqlite_file(db_path)
    if sqlite_file is not None and not sqlite_file.exists():
        raise LedgerFailure("ledger_db_missing")
    actor = get_actor(session_id, db_path=db_path)
    if actor is None:
        raise LedgerFailure("ledger_actor_unbound")
    user, default_enterprise = actor
    raw_enterprise = str(enterprise or "").strip()
    if raw_enterprise:
        code = normalize_enterprise(raw_enterprise)
        if code is None:
            raise LedgerFailure("ledger_enterprise_required")
    else:
        code = default_enterprise
    if code is None:
        raise LedgerFailure("ledger_enterprise_required")
    month_key = None
    if need_month:
        month_key = normalize_month(month)
        if month_key is None:
            raise LedgerFailure("ledger_month_required", enterprise=code)
    return user, code, month_key


def _require_read(db_path: Path | None, user: str, enterprise: str) -> None:
    read, _write = _permission(db_path, user, enterprise)
    if not read:
        raise LedgerFailure("ledger_permission_denied", enterprise=enterprise, user=user)


def _require_write(db_path: Path | None, user: str, enterprise: str) -> None:
    _read, write = _permission(db_path, user, enterprise)
    if not write:
        raise LedgerFailure("ledger_permission_denied", enterprise=enterprise, user=user)


def _permission(db_path: Path | None, user: str, enterprise: str) -> tuple[bool, bool]:
    with _connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT can_read, can_write
            FROM permissions
            WHERE user_code = ? AND enterprise_code = ?
            """,
            (user, enterprise),
        ).fetchone()
    if row is None:
        return False, False
    return bool(row["can_read"]), bool(row["can_write"])


def _invoice_dict(row: sqlite3.Row) -> dict[str, str]:
    return {
        "invoice_no": str(row["invoice_no"]),
        "issued_on": str(row["issued_on"]),
        "amount": from_cents(int(row["amount_cents"])),
        "purpose": str(row["purpose"] or ""),
        "remark": str(row["remark"] or ""),
    }


def _purpose_missing(row: sqlite3.Row, required: str, threshold: str | None) -> bool:
    if not is_blank(row["purpose"]):
        return False
    if required == "always":
        return True
    if required == "amount_gte":
        if not threshold:
            raise LedgerFailure("ledger_rule_missing")
        return int(row["amount_cents"]) >= to_cents(threshold)
    raise LedgerFailure("ledger_rule_missing")


def _read_parties(sheet) -> tuple[list[tuple[str, str, str]], list[tuple[str, str]], list[tuple[str, str, int, int]]]:
    companies: list[tuple[str, str, str]] = []
    users: list[tuple[str, str]] = []
    permissions: list[tuple[str, str, int, int]] = []
    rows = list(sheet.iter_rows(values_only=True))
    mode = ""
    enterprise_columns: list[tuple[int, str]] = []
    for row in rows:
        cells = [("" if cell is None else str(cell).strip()) for cell in row]
        if not any(cells):
            continue
        if cells[0] == "企业编号":
            mode = "company"
            continue
        if cells[0] == "用户编号":
            mode = "user"
            enterprise_columns = []
            for index, cell in enumerate(cells[2:], start=2):
                code = normalize_enterprise(cell)
                if code is None and "C001" in cell:
                    code = "C001"
                if code is None and "C002" in cell:
                    code = "C002"
                if code:
                    enterprise_columns.append((index, code))
            continue
        if mode == "company" and normalize_enterprise(cells[0]):
            companies.append((normalize_enterprise(cells[0]) or "", cells[1], cells[2] or cells[1]))
        elif mode == "user" and _USER_CODE.fullmatch(cells[0].upper()):
            user_code = cells[0].upper()
            users.append((user_code, cells[1]))
            for index, code in enterprise_columns:
                access = cells[index] if index < len(cells) else ""
                if "无权" in access or not access:
                    continue
                can_write = 1 if "修改" in access else 0
                can_read = 1 if ("查看" in access or can_write) else 0
                if can_read:
                    permissions.append((user_code, code, can_read, can_write))
    return companies, users, permissions


def _read_invoices(sheet) -> list[tuple[object, ...]]:
    records: list[tuple[object, ...]] = []
    started = False
    for row in sheet.iter_rows(values_only=True):
        cells = list(row)
        head = "" if cells[0] is None else str(cells[0]).strip()
        if head == "企业编号":
            started = True
            continue
        if not started or not normalize_enterprise(head):
            continue
        purpose = "" if len(cells) < 7 or cells[6] is None else str(cells[6]).strip()
        remark = "" if len(cells) < 8 or cells[7] is None else str(cells[7]).strip()
        records.append(
            (
                normalize_enterprise(head),
                str(cells[1]).strip().upper(),
                _date_text(cells[2]),
                str(cells[3] or "").strip(),
                str(cells[4] or "").strip(),
                to_cents(cells[5]),
                purpose,
                remark,
            )
        )
    return records


def _read_bank(sheet) -> list[tuple[object, ...]]:
    records: list[tuple[object, ...]] = []
    started = False
    for row in sheet.iter_rows(values_only=True):
        cells = list(row)
        head = "" if cells[0] is None else str(cells[0]).strip()
        if head == "企业编号":
            started = True
            continue
        if not started or not normalize_enterprise(head):
            continue
        remark = "" if len(cells) < 7 or cells[6] is None else str(cells[6]).strip()
        records.append(
            (
                normalize_enterprise(head),
                str(cells[1]).strip().upper(),
                _date_text(cells[2]),
                str(cells[3]).strip(),
                to_cents(cells[4]),
                str(cells[5] or "").strip(),
                remark,
            )
        )
    return records


def _date_text(value: object) -> str:
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")  # type: ignore[union-attr]
    text = str(value or "").strip()
    return text[:10]


def _sqlite_file(db_path: Path | None) -> Path | None:
    """显式路径或 LEDGER_DB_PATH 使用 SQLite。两者都没有时使用 PostgreSQL 的 ledger 模式。"""
    if db_path is not None:
        return Path(db_path)
    override = os.environ.get("LEDGER_DB_PATH", "").strip()
    if override:
        return Path(override)
    return None


def _lock_for(path: Path) -> threading.Lock:
    key = str(path.resolve()) if path.exists() else str(path)
    lock = _LOCKS.get(key)
    if lock is None:
        lock = threading.Lock()
        _LOCKS[key] = lock
    return lock


class _PostgresConnection:
    """把现有的问号参数 SQL 送到 psycopg。"""

    def __init__(self, connection: object) -> None:
        self._connection = connection

    def execute(self, sql: str, params: object = ()):
        return self._connection.execute(sql.replace("?", "%s"), params or ())  # type: ignore[attr-defined]

    def executemany(self, sql: str, seq: object) -> None:
        with self._connection.cursor() as cursor:  # type: ignore[attr-defined]
            cursor.executemany(sql.replace("?", "%s"), list(seq))


def _postgres_dsn() -> str:
    from app.core.config import settings

    url = settings.DATABASE_URL
    for prefix in ("postgresql+asyncpg://", "postgres+asyncpg://", "postgresql+psycopg://"):
        if url.startswith(prefix):
            return "postgresql://" + url[len(prefix) :]
    return url


def ensure_ledger_tables() -> None:
    from sqlalchemy import create_engine, text

    from app.core.config import settings
    from app.core.database import Base
    from app.models.ledger.finance import (
        LedgerBankTxn,
        LedgerCompany,
        LedgerInvoice,
        LedgerPermission,
        LedgerRuleDoc,
        LedgerOpenTask,
        LedgerSessionScope,
        LedgerUser,
    )

    url = settings.DATABASE_URL.replace("postgresql+asyncpg", "postgresql+psycopg").replace(
        "postgres+asyncpg", "postgresql+psycopg"
    )
    engine = create_engine(
        url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 5},
    )
    tables = [
        LedgerCompany.__table__,
        LedgerUser.__table__,
        LedgerPermission.__table__,
        LedgerInvoice.__table__,
        LedgerBankTxn.__table__,
        LedgerRuleDoc.__table__,
        LedgerSessionScope.__table__,
        LedgerOpenTask.__table__,
    ]
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS ledger"))
        Base.metadata.create_all(conn, tables=tables)
    engine.dispose()


@contextmanager
def _connection(db_path: Path | None = None) -> Iterator[object]:
    sqlite_file = _sqlite_file(db_path)
    if sqlite_file is not None:
        sqlite_file.parent.mkdir(parents=True, exist_ok=True)
        lock = _lock_for(sqlite_file)
        with lock:
            conn = sqlite3.connect(sqlite_file)
            conn.row_factory = sqlite3.Row
            try:
                conn.executescript(_SCHEMA)
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
        return

    import psycopg
    from psycopg.rows import dict_row

    ensure_ledger_tables()
    lock = _LOCKS.setdefault("postgres", threading.Lock())
    with lock:
        conn = psycopg.connect(
            _postgres_dsn(),
            row_factory=dict_row,
            connect_timeout=5,
        )
        try:
            conn.execute("SET search_path TO ledger")
            yield _PostgresConnection(conn)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
