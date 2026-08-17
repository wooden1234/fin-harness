"""工具错误码、分类与对模型的处理指引。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping


MALFORMED_ARGUMENTS = "malformed_arguments"


class ToolErrorClass(StrEnum):
    """错误语义分类。调度与 loop 只认这一层，不认散落的 error 字符串。"""

    TRANSIENT = "transient"  # 超时/网络，允许同工具再试一次
    INVALID_INPUT = "invalid_input"  # 参数或槽位不对，禁止原样重试
    EMPTY = "empty"  # 查到了但没数据，换源或如实缺口
    UNAVAILABLE = "unavailable"  # 数据源没配或挂了，不要再打同一工具
    POLICY = "policy"  # 运行时规则：未知工具、重试耗尽
    COMPENSATE = "compensate"  # 成稿模型不可用，主路径根据已有材料作答
    CONTROL = "control"  # 取消、等待审批，不是对用户的失败


class ToolErrorAction(StrEnum):
    """本批工具结果之后，运行时要做的事。"""

    SURFACE = "surface"  # 只写进 tool/result，让模型自己看
    INJECT = "inject"  # 再注入一条 plugin 说明
    PUBLISH_UNAVAILABLE = "publish_unavailable"  # 本轮没有成功数据则直接回复用户


@dataclass(frozen=True, slots=True)
class ToolErrorSpec:
    error_class: ToolErrorClass
    action: ToolErrorAction = ToolErrorAction.SURFACE
    guidance: str = ""


_GUIDANCE = {
    ToolErrorClass.TRANSIENT: "可对同一工具重试一次；不要改用不相关工具。",
    ToolErrorClass.INVALID_INPUT: "不要原样重试。修正参数，或缺的信息向用户澄清。",
    ToolErrorClass.EMPTY: "不要用同一查询再打同一工具。可换更匹配的 skill/工具，或如实告知未查到。",
    ToolErrorClass.UNAVAILABLE: "该数据源不可用。不要重试同一工具；可换其它来源，或告知用户稍后再试。",
    ToolErrorClass.POLICY: "不要再调用该工具。若本轮没有其它成功结果，用系统提示原话回复用户。",
    ToolErrorClass.COMPENSATE: (
        "分析模型不可用。不要再调用 finalign_analyze，也不要用其它 LLM 补偿。"
        "请根据本轮已检索的 tool/result 直接用正文回答用户。"
    ),
    ToolErrorClass.CONTROL: "",
}

_CATALOG: dict[str, ToolErrorSpec] = {}


def _add(code: str, error_class: ToolErrorClass, *, action: ToolErrorAction | None = None) -> None:
    _CATALOG[code] = ToolErrorSpec(
        error_class=error_class,
        action=action or ToolErrorAction.SURFACE,
        guidance=_GUIDANCE[error_class],
    )


for _code in ("retry_exhausted", "unknown_tool"):
    _add(_code, ToolErrorClass.POLICY, action=ToolErrorAction.PUBLISH_UNAVAILABLE)
_add("denied", ToolErrorClass.POLICY)
for _code in ("cancelled", "deferred_for_approval"):
    _add(_code, ToolErrorClass.CONTROL)
for _code in (
    MALFORMED_ARGUMENTS,
    "invalid_skill_name",
    "skill_not_found",
    "empty_question",
    "empty_city",
    "need_city",
    "empty_clarification",
    "fact_query_requires_clarification",
    "fact_query_scope_unsupported",
    "source_locked_doc_ids_required",
    "unsupported_memory_key",
    "invalid_memory_value",
    "invalid_todos",
    "no_evidence_to_analyze",
    "query_must_not_be_empty",
    "invalid_call_type",
    "invalid_calculation_batch_size",
):
    _add(_code, ToolErrorClass.INVALID_INPUT)
for _code in (
    "tool_timeout",
    "timeout",
    "iwencai_timeout",
    "iwencai_network_error",
    "http_error",
    "weather_api_error",
    "iwencai_http_error",
    "tool_execution_failed",
    "analysis_failed",
    "iwencai_invalid_json",
    "skill_runner_timeout",
    "skill_runner_failed",
    "memory_write_failed",
    "memory_delete_failed",
):
    _add(_code, ToolErrorClass.TRANSIENT)
for _code in (
    "empty_result",
    "fact_not_in_local_store",
    "city_not_found",
    "skill_runner_invalid_json",
):
    _add(_code, ToolErrorClass.EMPTY)
for _code in (
    "iwencai_not_configured",
    "not_configured",
    "invalid_api_key",
    "iwencai_auth_failed",
    "skill_runner_disabled",
):
    _add(_code, ToolErrorClass.UNAVAILABLE)
for _code in ("finalign_unavailable", "draft_disabled"):
    _add(_code, ToolErrorClass.COMPENSATE)


def _infer_class(code: str) -> ToolErrorClass:
    lowered = code.lower()
    if any(token in lowered for token in ("unavailable", "not_configured", "auth_failed")):
        return ToolErrorClass.UNAVAILABLE
    if any(token in lowered for token in ("timeout", "network", "http_error", "rate")):
        return ToolErrorClass.TRANSIENT
    if any(token in lowered for token in ("empty", "not_found", "no_evidence")):
        return ToolErrorClass.EMPTY
    if any(token in lowered for token in ("invalid", "malformed", "need_", "clarif")):
        return ToolErrorClass.INVALID_INPUT
    return ToolErrorClass.TRANSIENT


def classify(code: str | None) -> ToolErrorSpec:
    key = str(code or "").strip()
    if not key:
        return ToolErrorSpec(ToolErrorClass.TRANSIENT, guidance=_GUIDANCE[ToolErrorClass.TRANSIENT])
    known = _CATALOG.get(key)
    if known is not None:
        return known
    error_class = _infer_class(key)
    return ToolErrorSpec(error_class=error_class, guidance=_GUIDANCE[error_class])


def error_result(code: str, **extra: object) -> dict[str, object]:
    spec = classify(code)
    payload: dict[str, object] = {
        "ok": False,
        "error": code,
        "error_class": spec.error_class.value,
        "model_guidance": spec.guidance,
    }
    payload.update(extra)
    return payload


def enrich_tool_result(result: Mapping[str, Any] | None) -> dict[str, Any]:
    """给任意失败 payload 补上 error_class / model_guidance。成功结果原样返回。"""
    payload = dict(result or {})
    if payload.get("ok", True):
        return payload
    spec = classify(str(payload.get("error") or ""))
    payload.setdefault("error_class", spec.error_class.value)
    payload.setdefault("model_guidance", spec.guidance)
    return payload


def publishes_if_no_success(result: Mapping[str, Any] | None) -> bool:
    payload = dict(result or {})
    if payload.get("ok", True):
        return False
    spec = classify(str(payload.get("error") or ""))
    return spec.action == ToolErrorAction.PUBLISH_UNAVAILABLE


def injects_always(result: Mapping[str, Any] | None) -> bool:
    payload = dict(result or {})
    if payload.get("ok", True):
        return False
    spec = classify(str(payload.get("error") or ""))
    return spec.action == ToolErrorAction.INJECT
