"""账房工具：规则检索与业务数据。用户身份从会话绑定读取，不接受模型传入。"""

from __future__ import annotations

from typing import Any

from harness.tools.definition import ToolDefinition, function_schema
from harness.tools.errors import error_result
from retrieval.services.ledger_rule_kb import active_rules
from retrieval.services.ledger_store import (
    LedgerFailure,
    check_purpose,
    query_bank,
    query_invoices,
    read_scope,
    update_invoice,
)

_ACTIONS = ("query_invoices", "query_bank", "check_purpose", "update_invoice")
_TOPICS = ("amount", "invoice_purpose", "bank")
_READONLY_FIELDS = ("amount", "seller", "item_name", "issued_on")

_RULE_PARAMETERS = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "enterprise": {
            "type": "string",
            "description": "企业编号或简称，如 C001、甲企业。省略时使用会话里选定的企业。",
        },
        "month": {"type": "string", "description": "业务月份，格式 YYYY-MM。"},
        "topic": {
            "type": "string",
            "enum": list(_TOPICS),
            "description": "规则主题。amount 金额，invoice_purpose 发票用途，bank 银行资料。可省略。",
        },
    },
    "required": ["month"],
}

_RECORD_PARAMETERS = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "action": {
            "type": "string",
            "enum": list(_ACTIONS),
            "description": "query_invoices 查发票，query_bank 查银行，check_purpose 查缺用途，update_invoice 保存用途或备注。",
        },
        "enterprise": {
            "type": "string",
            "description": "企业编号或简称，如 C001、甲企业。省略时使用会话里选定的企业。",
        },
        "month": {"type": "string", "description": "业务月份 YYYY-MM。查询和检查用途时必填。"},
        "invoice_no": {"type": "string", "description": "发票编号。保存时必填，与企业编号共同定位。"},
        "purpose": {"type": "string", "description": "要保存的用途。空白视为未填写并拒绝。"},
        "remark": {"type": "string", "description": "要保存的备注。备注不能代替用途。"},
        "amount": {"description": "价税合计不可修改。传入则本次保存会被拒绝。"},
        "seller": {"description": "销售方不可修改。传入则本次保存会被拒绝。"},
        "item_name": {"description": "项目名称不可修改。传入则本次保存会被拒绝。"},
        "issued_on": {"description": "开票日期不可修改。传入则本次保存会被拒绝。"},
    },
    "required": ["action"],
}


def ledger_tool_definitions(store: Any, session_id: str) -> tuple[ToolDefinition, ToolDefinition]:
    """构造当前会话的规则工具和业务数据工具。store 与记忆工具保持同一绑定签名。"""
    _ = store

    async def _search(arguments: dict[str, Any]) -> dict[str, Any]:
        topic = str(arguments.get("topic") or "").strip()
        if topic and topic not in _TOPICS:
            return error_result("ledger_action_invalid")
        try:
            _user, code, month = read_scope(
                session_id,
                arguments.get("enterprise"),
                arguments.get("month"),
            )
        except LedgerFailure as exc:
            return error_result(exc.code, **exc.extra)
        rules = [
            rule.public_dict()
            for rule in active_rules(enterprise=code, month=month or "", topic=topic or None)
        ]
        payload: dict[str, Any] = {
            "ok": True,
            "enterprise": code,
            "month": month,
            "topic": topic,
            "rules": rules,
        }
        if not rules:
            payload["message"] = "没有适用规则"
        return payload

    async def _records(arguments: dict[str, Any]) -> dict[str, Any]:
        action = str(arguments.get("action") or "").strip()
        if action not in _ACTIONS:
            return error_result("ledger_action_invalid")
        readonly = [name for name in _READONLY_FIELDS if name in arguments]
        if readonly:
            return error_result("ledger_field_readonly", fields=readonly)
        try:
            if action == "query_invoices":
                return query_invoices(
                    session_id=session_id,
                    enterprise=arguments.get("enterprise"),
                    month=arguments.get("month"),
                )
            if action == "query_bank":
                return query_bank(
                    session_id=session_id,
                    enterprise=arguments.get("enterprise"),
                    month=arguments.get("month"),
                )
            if action == "check_purpose":
                return check_purpose(
                    session_id=session_id,
                    enterprise=arguments.get("enterprise"),
                    month=arguments.get("month"),
                )
            return update_invoice(
                session_id=session_id,
                enterprise=arguments.get("enterprise"),
                invoice_no=arguments.get("invoice_no"),
                purpose=arguments.get("purpose"),
                remark=arguments.get("remark"),
                purpose_supplied="purpose" in arguments and arguments.get("purpose") is not None,
                remark_supplied="remark" in arguments and arguments.get("remark") is not None,
            )
        except LedgerFailure as exc:
            return error_result(exc.code, **exc.extra)

    search = ToolDefinition(
        tool_id="ledger_search_rules",
        name="ledger_search_rules",
        description=(
            "按企业、月份和主题检索当前启用的账房规则，返回规则名称、版本和正文。"
            "已停用版本不会返回。每次调用只传一个 JSON 对象。"
        ),
        handler=_search,
        openai_schema=function_schema(
            "ledger_search_rules",
            "检索当前启用的账房规则",
            _RULE_PARAMETERS,
        ),
        is_concurrency_safe=True,
        read_only=True,
    )
    records = ToolDefinition(
        tool_id="ledger_records",
        name="ledger_records",
        description=(
            "查询或保存比赛模拟企业的发票与银行流水。"
            "action 取 query_invoices、query_bank、check_purpose、update_invoice。"
            "金额、权限和用途是否必填由工具判断。不能在参数里指定用户。"
            "每次调用只传一个 JSON 对象和一个 action；查发票和银行要分成两次调用。"
        ),
        handler=_records,
        openai_schema=function_schema(
            "ledger_records",
            "查询发票、银行流水，检查用途或保存用途和备注",
            _RECORD_PARAMETERS,
        ),
        is_concurrency_safe=False,
        read_only=False,
    )
    return search, records
