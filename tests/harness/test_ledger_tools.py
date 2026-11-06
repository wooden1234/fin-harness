from __future__ import annotations

from pathlib import Path
import sqlite3

import pytest

from harness.session.store import InMemorySessionStore
from harness.tools.errors import ToolErrorClass, classify
from harness.tools.ledger import ledger_tool_definitions
from harness.tools.arguments import merge_tool_results
from harness.tools.pipeline import ToolPipeline
from retrieval.services.ledger_store import (
    bind_actor,
    bind_session_ledger_code,
    ensure_session_actor,
    get_actor,
    import_excel,
)


_ROOT = Path(__file__).resolve().parents[2]


def _xlsx() -> Path:
    matches = sorted(_ROOT.glob("赛题二*/02_示例数据.xlsx"))
    assert matches, "示例数据 Excel 不在仓库中"
    return matches[0]


def _tools(session_id: str):
    store = InMemorySessionStore()
    return {item.name: item for item in ledger_tool_definitions(store, session_id)}


def _prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, session_id: str, user: str, enterprise: str = ""):
    db = tmp_path / "finance.sqlite"
    monkeypatch.setenv("LEDGER_DB_PATH", str(db))
    import_excel(_xlsx(), db_path=db)
    bind_actor(session_id, user, enterprise_code=enterprise or None, db_path=db)
    return _tools(session_id)


def test_ledger_errors_are_not_retried_as_transient():
    for code in (
        "ledger_enterprise_required",
        "ledger_month_required",
        "ledger_permission_denied",
        "ledger_not_found",
        "ledger_empty_purpose",
        "ledger_rule_missing",
        "ledger_actor_unbound",
        "ledger_field_readonly",
    ):
        assert classify(code).error_class == ToolErrorClass.INVALID_INPUT
    assert classify("ledger_db_missing").error_class == ToolErrorClass.UNAVAILABLE


@pytest.mark.asyncio
async def test_invoice_and_bank_totals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    tools = _prepare(tmp_path, monkeypatch, "s-u001", "U001")
    invoices = await tools["ledger_records"].handler(
        {"action": "query_invoices", "enterprise": "甲企业", "month": "2026-09"}
    )
    bank = await tools["ledger_records"].handler(
        {"action": "query_bank", "enterprise": "C001", "month": "2026-09"}
    )
    assert invoices["ok"] is True
    assert invoices["count"] == 10
    assert invoices["total"] == "9622.06"
    assert invoices["empty"] is False
    assert bank["income"] == "13000.50"
    assert bank["expense"] == "7180.74"
    assert bank["count"] == 12
    with sqlite3.connect(tmp_path / "finance.sqlite") as conn:
        users = [row[0] for row in conn.execute("SELECT code FROM users ORDER BY code")]
    assert users == ["U001", "U002"]


@pytest.mark.asyncio
async def test_july_query_succeeds_with_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    tools = _prepare(tmp_path, monkeypatch, "s-july", "U001")
    invoices = await tools["ledger_records"].handler(
        {"action": "query_invoices", "enterprise": "C002", "month": "2026-07"}
    )
    bank = await tools["ledger_records"].handler(
        {"action": "query_bank", "enterprise": "C002", "month": "2026-07"}
    )
    assert invoices["empty"] is True
    assert invoices["total"] == "0.00"
    assert invoices["count"] == 0
    assert bank["empty"] is True
    assert bank["income"] == "0.00"
    assert bank["expense"] == "0.00"


@pytest.mark.asyncio
async def test_permission_denied_is_not_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    tools = _prepare(tmp_path, monkeypatch, "s-u002", "U002")
    denied = await tools["ledger_records"].handler(
        {"action": "query_invoices", "enterprise": "C002", "month": "2026-09"}
    )
    assert denied["ok"] is False
    assert denied["error"] == "ledger_permission_denied"
    assert "total" not in denied
    readonly = await tools["ledger_records"].handler(
        {
            "action": "update_invoice",
            "enterprise": "C001",
            "invoice_no": "F001",
            "purpose": "不应写上",
        }
    )
    assert readonly["error"] == "ledger_permission_denied"
    owner = _prepare(tmp_path, monkeypatch, "s-check", "U001")
    current = await owner["ledger_records"].handler(
        {"action": "query_invoices", "enterprise": "C001", "month": "2026-09"}
    )
    f001 = next(item for item in current["invoices"] if item["invoice_no"] == "F001")
    assert f001["purpose"] == "购买办公文具"


@pytest.mark.asyncio
async def test_purpose_check_follows_active_rule(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    tools = _prepare(tmp_path, monkeypatch, "s-check", "U001")
    company_a = await tools["ledger_records"].handler(
        {"action": "check_purpose", "enterprise": "C001", "month": "2026-09"}
    )
    company_b = await tools["ledger_records"].handler(
        {"action": "check_purpose", "enterprise": "乙企业", "month": "2026-09"}
    )
    assert company_a["rule"] == {
        "rule_id": "R002",
        "name": "甲企业发票用途规则",
        "version": "2.0",
    }
    assert {item["invoice_no"] for item in company_a["missing"]} == {
        "F002",
        "F003",
        "F004",
        "F006",
        "F009",
    }
    assert {item["invoice_no"] for item in company_b["missing"]} == {"F002", "F005", "F006"}
    assert company_b["rule"]["version"] == "1.0"


@pytest.mark.asyncio
async def test_update_rereads_and_rejects_blank_or_amount(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    tools = _prepare(tmp_path, monkeypatch, "s-save", "U001")
    saved = await tools["ledger_records"].handler(
        {
            "action": "update_invoice",
            "enterprise": "C001",
            "invoice_no": "F002",
            "purpose": "  补充办公用品 ",
        }
    )
    assert saved["ok"] is True
    assert saved["invoice"]["purpose"] == "补充办公用品"
    assert saved["invoice"]["amount"] == "800.00"
    blank = await tools["ledger_records"].handler(
        {"action": "update_invoice", "enterprise": "C001", "invoice_no": "F002", "purpose": "   "}
    )
    assert blank["error"] == "ledger_empty_purpose"
    amount = await tools["ledger_records"].handler(
        {
            "action": "update_invoice",
            "enterprise": "C001",
            "invoice_no": "F002",
            "purpose": "另一用途",
            "amount": "1.00",
        }
    )
    assert amount["error"] == "ledger_field_readonly"
    current = await tools["ledger_records"].handler(
        {"action": "query_invoices", "enterprise": "C001", "month": "2026-09"}
    )
    f002 = next(item for item in current["invoices"] if item["invoice_no"] == "F002")
    assert f002["purpose"] == "补充办公用品"
    assert f002["amount"] == "800.00"


@pytest.mark.asyncio
async def test_reimport_keeps_saved_purpose(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db = tmp_path / "finance.sqlite"
    monkeypatch.setenv("LEDGER_DB_PATH", str(db))
    import_excel(_xlsx(), db_path=db)
    bind_actor("s-keep", "U001", db_path=db)
    tools = _tools("s-keep")
    await tools["ledger_records"].handler(
        {
            "action": "update_invoice",
            "enterprise": "C001",
            "invoice_no": "F009",
            "purpose": "设备配件",
        }
    )
    import_excel(_xlsx(), db_path=db)
    current = await tools["ledger_records"].handler(
        {"action": "query_invoices", "enterprise": "C001", "month": "2026-09"}
    )
    f009 = next(item for item in current["invoices"] if item["invoice_no"] == "F009")
    assert f009["purpose"] == "设备配件"


@pytest.mark.asyncio
async def test_rules_tool_hides_disabled_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    tools = _prepare(tmp_path, monkeypatch, "s-rules", "U001", enterprise="甲企业")
    result = await tools["ledger_search_rules"].handler(
        {"month": "2026-09", "topic": "invoice_purpose"}
    )
    assert result["ok"] is True
    assert result["enterprise"] == "C001"
    assert [item["version"] for item in result["rules"]] == ["2.0"]
    assert result["rules"][0]["name"] == "甲企业发票用途规则"
    early = await tools["ledger_search_rules"].handler(
        {"enterprise": "C001", "month": "2026-07", "topic": "invoice_purpose"}
    )
    assert early["rules"] == []
    assert early["message"] == "没有适用规则"


@pytest.mark.asyncio
async def test_missing_actor_month_and_enterprise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db = tmp_path / "finance.sqlite"
    monkeypatch.setenv("LEDGER_DB_PATH", str(db))
    import_excel(_xlsx(), db_path=db)
    unbound = _tools("missing-session")
    actor = await unbound["ledger_records"].handler(
        {"action": "query_bank", "enterprise": "C001", "month": "2026-09"}
    )
    assert actor["error"] == "ledger_actor_unbound"
    tools = _prepare(tmp_path, monkeypatch, "s-ask", "U001")
    month = await tools["ledger_records"].handler(
        {"action": "query_invoices", "enterprise": "C001"}
    )
    enterprise = await tools["ledger_search_rules"].handler({"month": "2026-09"})
    assert month["error"] == "ledger_month_required"
    assert enterprise["error"] == "ledger_enterprise_required"


@pytest.mark.asyncio
async def test_bind_runtime_exposes_ledger_tools():
    from harness.control.services import AgentControl
    from harness.tools.runtime import ToolRuntime
    from harness.tools.skill import skill_definition

    store = InMemorySessionStore()
    header = await store.create(tenant_id="t", user_id="1")
    runtime = AgentControl(store, header.session_id).bind_runtime(
        turn=1,
        run_id="r1",
        base_runtime=ToolRuntime((skill_definition(),)),
    )
    assert runtime.resolve("ledger_search_rules").read_only is True
    records = runtime.resolve("ledger_records")
    assert records.read_only is False
    assert records.is_concurrency_safe is False


def test_account_ledger_code_binds_permission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db = tmp_path / "finance.sqlite"
    monkeypatch.setenv("LEDGER_DB_PATH", str(db))
    assert bind_session_ledger_code("s-account", None, db_path=db) is None
    assert bind_session_ledger_code("s-account", "guest", db_path=db) is None
    assert get_actor("s-account", db_path=db) is None
    assert bind_session_ledger_code("s-account", "U002", db_path=db) == "U002"
    bind_actor("s-account", "U002", enterprise_code="C001", db_path=db)
    assert bind_session_ledger_code("s-account", "u001", db_path=db) == "U001"
    assert get_actor("s-account", db_path=db) == ("U001", "C001")


def test_ensure_session_actor_keeps_existing_user(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db = tmp_path / "finance.sqlite"
    monkeypatch.setenv("LEDGER_DB_PATH", str(db))
    import_excel(_xlsx(), db_path=db)
    assert ensure_session_actor("fresh", db_path=db) == ("U001", None)
    bind_actor("kept", "U002", enterprise_code="C001", db_path=db)
    assert ensure_session_actor("kept", db_path=db) == ("U002", "C001")


def test_partial_batch_keeps_successful_results():
    merged = merge_tool_results(
        [
            {"ok": True, "invoice_total": "9622.06"},
            {"ok": False, "error": "ledger_action_invalid"},
        ]
    )
    assert merged["ok"] is True
    assert merged["results"][0]["invoice_total"] == "9622.06"
    assert "error" not in merged


@pytest.mark.asyncio
async def test_glued_record_calls_still_return_totals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    tools = _prepare(tmp_path, monkeypatch, "s-glued", "U001")
    raw = (
        '{"enterprise": "甲企业", "month": "2026-09"}'
        '{"action": "query_invoices", "enterprise": "甲企业", "month": "2026-09"}'
        '{"action": "query_bank", "enterprise": "甲企业", "month": "2026-09"}'
    )
    result = await ToolPipeline().run(tools["ledger_records"], raw)
    assert result["ok"] is True
    payloads = result["results"]
    invoices = next(item for item in payloads if item.get("action") == "query_invoices")
    bank = next(item for item in payloads if item.get("action") == "query_bank")
    assert invoices["total"] == "9622.06"
    assert bank["income"] == "13000.50"
    assert bank["expense"] == "7180.74"
