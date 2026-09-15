"""天眼查 MCP 适配器：把常用企业查询注册为 Agent 工具。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from mcp.config import tyc_mcp_configured, tyc_mcp_timeout_sec
from mcp.gateway import McpGateway
from tools.core.mcp_proxy import register_mcp_tool

_TYC_SERVER = "tyc-mcp"
gateway = McpGateway()
_TIMEOUT = tyc_mcp_timeout_sec()


class TycCompanySearchInput(BaseModel):
    searchKey: str = Field(
        description="企业名称、简称或 18 位统一社会信用代码；多候选时需让用户确认后再查详情",
    )


class TycCompanyIdInput(BaseModel):
    searchKey: str = Field(
        description="企业名称、简称或统一社会信用代码；建议先用 tyc_company_search 锚定主体",
    )


class TycCompanyCapabilitiesInput(BaseModel):
    id: str = Field(description="企业 id（来自 tyc_company_search 返回的 items[*].id）")
    company_name: str = Field(
        default="",
        description="企业全称，与 id 一并传入可提高命中率",
    )


class TycCallToolInput(BaseModel):
    tool_name: str = Field(
        description="天眼查业务工具名，如 company.registration-info、risk.judicial-documents",
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="传给该业务工具的参数对象",
    )


def _capabilities_arguments(payload: dict[str, Any]) -> dict[str, Any]:
    arguments: dict[str, Any] = {"id": payload.get("id")}
    company_name = str(payload.get("company_name") or "").strip()
    if company_name:
        arguments["companyName"] = company_name
    return arguments


def register_tyc_tools() -> None:
    """注册天眼查 MCP 工具；未配置 env 时跳过，避免 Agent 绑定不可用工具。"""
    if not tyc_mcp_configured():
        return

    register_mcp_tool(
        tool_id="tyc.company.search",
        name="tyc_company_search",
        description=(
            "按企业名称或统一社会信用代码搜索/锚定主体，返回候选企业列表及 id。"
            "存在多个候选时必须让用户确认，禁止自动取第一条。"
        ),
        server=_TYC_SERVER,
        remote_tool="company.companies",
        gateway=gateway,
        args_schema=TycCompanySearchInput,
        timeout_seconds=_TIMEOUT,
    )
    register_mcp_tool(
        tool_id="tyc.company.registration",
        name="tyc_company_registration",
        description="查询企业工商登记信息（名称、法人、注册资本、成立日期、经营状态等）。",
        server=_TYC_SERVER,
        remote_tool="company.registration-info",
        gateway=gateway,
        args_schema=TycCompanyIdInput,
        timeout_seconds=_TIMEOUT,
    )
    register_mcp_tool(
        tool_id="tyc.company.shareholders",
        name="tyc_company_shareholders",
        description="查询企业股东及出资信息。",
        server=_TYC_SERVER,
        remote_tool="company.shareholders",
        gateway=gateway,
        args_schema=TycCompanyIdInput,
        timeout_seconds=_TIMEOUT,
    )
    register_mcp_tool(
        tool_id="tyc.company.judicial",
        name="tyc_company_judicial",
        description="查询企业司法风险相关公开信息（裁判文书等）。",
        server=_TYC_SERVER,
        remote_tool="risk.judicial-documents",
        gateway=gateway,
        args_schema=TycCompanyIdInput,
        timeout_seconds=_TIMEOUT,
    )
    register_mcp_tool(
        tool_id="tyc.company.capabilities",
        name="tyc_company_capabilities",
        description="查询指定企业 id 可调用的天眼查工具白名单（capabilities）。",
        server=_TYC_SERVER,
        remote_tool="company.capabilities",
        gateway=gateway,
        args_schema=TycCompanyCapabilitiesInput,
        argument_adapter=_capabilities_arguments,
        timeout_seconds=_TIMEOUT,
    )
    register_mcp_tool(
        tool_id="tyc.call_tool",
        name="tyc_call_tool",
        description=(
            "调用任意天眼查业务语义工具（162 个之一）。"
            "优先使用专用工具；仅在 capabilities 或文档明确要求时使用本入口。"
        ),
        server=_TYC_SERVER,
        remote_tool="call_tool",
        gateway=gateway,
        args_schema=TycCallToolInput,
        timeout_seconds=_TIMEOUT,
    )


register_tyc_tools()

__all__ = ["gateway", "register_tyc_tools"]
