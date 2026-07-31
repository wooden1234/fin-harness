"""上下文空间 Token 测量与显式业务窗口。"""

from __future__ import annotations

import inspect
import json
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AnyMessage

from agents.context_compressor.tokens import estimate_tokens, message_text
from agents.context_space.models import ContextBudgetPolicy, ContextMeasurement


def model_max_input_tokens(model: Any | None) -> int | None:
    """读取模型声明的最大输入；缺失时由业务显式上限兜底。"""
    profile = getattr(model, "profile", None)
    if not isinstance(profile, dict):
        return None
    value = profile.get("max_input_tokens")
    return int(value) if isinstance(value, int) and value > 0 else None


def effective_context_limit(
    policy: ContextBudgetPolicy,
    model: Any | None = None,
) -> int:
    """业务上限不会因为模型窗口变大而自动增长。"""
    model_limit = model_max_input_tokens(model)
    if model_limit is None:
        return policy.explicit_max_tokens
    return min(policy.explicit_max_tokens, model_limit)


def _tool_schema_text(tools: Sequence[Any]) -> str:
    payload: list[Any] = []
    for tool in tools:
        schema = getattr(tool, "args_schema", None)
        if schema is not None and hasattr(schema, "model_json_schema"):
            schema = schema.model_json_schema()
        payload.append(
            {
                "name": getattr(tool, "name", type(tool).__name__),
                "description": getattr(tool, "description", ""),
                "args_schema": schema,
            }
        )
    return json.dumps(payload, ensure_ascii=False, default=str)


def _model_token_count(
    model: Any,
    messages: list[AnyMessage],
    tools: Sequence[Any],
) -> int | None:
    counter = getattr(model, "get_num_tokens_from_messages", None)
    if not callable(counter):
        return None
    try:
        parameters = inspect.signature(counter).parameters
        if tools and ("tools" in parameters or any(
            item.kind is inspect.Parameter.VAR_KEYWORD
            for item in parameters.values()
        )):
            return int(counter(messages, tools=tools))
        count = int(counter(messages))
        if tools:
            count += estimate_tokens(_tool_schema_text(tools))
        return count
    except (TypeError, ValueError, NotImplementedError):
        return None


def measure_context(
    messages: Sequence[AnyMessage],
    *,
    policy: ContextBudgetPolicy,
    model: Any | None = None,
    tools: Sequence[Any] = (),
    extra_texts: Sequence[str] = (),
) -> ContextMeasurement:
    """测量完整模型输入，模型 tokenizer 不可用时采用保守近似值。"""
    materialized = list(messages)
    exact = _model_token_count(model, materialized, tools) if model is not None else None
    if exact is not None:
        estimated = exact + sum(estimate_tokens(text) for text in extra_texts)
        counter_kind = "model_tokenizer"
    else:
        raw = sum(estimate_tokens(message_text(message)) for message in materialized)
        raw += sum(estimate_tokens(text) for text in extra_texts)
        if tools:
            raw += estimate_tokens(_tool_schema_text(tools))
        estimated = int(raw * policy.approximate_safety_multiplier + 0.999999)
        counter_kind = "approximate"

    model_limit = model_max_input_tokens(model)
    limit = effective_context_limit(policy, model)
    return ContextMeasurement(
        estimated_tokens=estimated,
        counter_kind=counter_kind,
        model_max_tokens=model_limit,
        effective_limit=limit,
        utilization_ratio=estimated / limit,
        trigger_exceeded=estimated >= policy.trigger_tokens(limit),
        admission_exceeded=estimated > policy.admission_tokens(limit),
    )


__all__ = [
    "effective_context_limit",
    "measure_context",
    "model_max_input_tokens",
]
