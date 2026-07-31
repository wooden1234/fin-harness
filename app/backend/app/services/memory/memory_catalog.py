"""统一记忆类型、字段元数据与静态约束目录。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Protocol


MemoryLifecycle = Literal["persistent", "turn", "forbidden"]
MemorySensitivity = Literal["low", "medium", "high", "forbidden"]
MemoryValueType = Literal["enum", "string", "object"]


@dataclass(frozen=True, slots=True)
class MemoryTypeDefinition:
    name: str
    lifecycle: MemoryLifecycle
    sql_writable: bool
    cacheable: bool
    vectorizable: bool
    structured: bool
    dynamic_keys: bool = False


@dataclass(frozen=True, slots=True)
class MemoryKeyDefinition:
    memory_key: str
    memory_type: str
    value_type: MemoryValueType
    choices: tuple[str, ...] = ()
    default_value: Any = None
    ttl_days: int | None = None
    confidence_threshold: float = 1.0
    sensitivity: MemorySensitivity = "low"
    lifecycle: MemoryLifecycle = "persistent"
    sql_writable: bool = True
    cacheable: bool = True
    vectorizable: bool = False
    prompt_visible: bool = True
    token_priority: int = 100


class AgentMemorySpec(Protocol):
    agent_id: str
    memory_keys: tuple[str, ...]
    semantic_memory_types: tuple[str, ...]


MEMORY_TYPES: dict[str, MemoryTypeDefinition] = {
    "preference": MemoryTypeDefinition(
        name="preference",
        lifecycle="persistent",
        sql_writable=True,
        cacheable=True,
        vectorizable=False,
        structured=True,
    ),
    "episodic": MemoryTypeDefinition(
        name="episodic",
        lifecycle="persistent",
        sql_writable=True,
        cacheable=False,
        vectorizable=True,
        structured=False,
        dynamic_keys=True,
    ),
    "turn_context": MemoryTypeDefinition(
        name="turn_context",
        lifecycle="turn",
        sql_writable=False,
        cacheable=False,
        vectorizable=False,
        structured=True,
        dynamic_keys=True,
    ),
    "forbidden": MemoryTypeDefinition(
        name="forbidden",
        lifecycle="forbidden",
        sql_writable=False,
        cacheable=False,
        vectorizable=False,
        structured=False,
        dynamic_keys=True,
    ),
}


def _preference(
    memory_key: str,
    choices: tuple[str, ...],
    *,
    default_value: str,
    ttl_days: int | None = None,
    confidence_threshold: float = 0.85,
    sensitivity: MemorySensitivity = "low",
    token_priority: int = 100,
) -> MemoryKeyDefinition:
    return MemoryKeyDefinition(
        memory_key=memory_key,
        memory_type="preference",
        value_type="enum",
        choices=choices,
        default_value=default_value,
        ttl_days=ttl_days,
        confidence_threshold=confidence_threshold,
        sensitivity=sensitivity,
        token_priority=token_priority,
    )


MEMORY_KEYS: dict[str, MemoryKeyDefinition] = {
    "response_language": _preference(
        "response_language",
        ("zh-CN", "en-US"),
        default_value="zh-CN",
        token_priority=10,
    ),
    "response_detail_level": _preference(
        "response_detail_level",
        ("brief", "standard", "detailed"),
        default_value="standard",
        ttl_days=730,
        token_priority=20,
    ),
    "preferred_output_format": _preference(
        "preferred_output_format",
        ("plain_text", "markdown", "table"),
        default_value="markdown",
        ttl_days=365,
        token_priority=30,
    ),
    "default_currency": _preference(
        "default_currency",
        ("CNY", "USD", "HKD"),
        default_value="CNY",
        ttl_days=365,
        sensitivity="medium",
        token_priority=40,
    ),
    "default_market": _preference(
        "default_market",
        ("CN", "HK", "US"),
        default_value="CN",
        ttl_days=180,
        token_priority=40,
    ),
    "default_compare_period": _preference(
        "default_compare_period",
        ("YoY", "QoQ", "MoM"),
        default_value="YoY",
        ttl_days=180,
        token_priority=50,
    ),
    "citation_preference": _preference(
        "citation_preference",
        ("always", "when_available", "never"),
        default_value="when_available",
        ttl_days=365,
        token_priority=20,
    ),
    "credential_secret": MemoryKeyDefinition(
        memory_key="credential_secret",
        memory_type="forbidden",
        value_type="string",
        confidence_threshold=1.0,
        sensitivity="forbidden",
        lifecycle="forbidden",
        sql_writable=False,
        cacheable=False,
        vectorizable=False,
        prompt_visible=False,
    ),
    "sensitive_identity": MemoryKeyDefinition(
        memory_key="sensitive_identity",
        memory_type="forbidden",
        value_type="object",
        confidence_threshold=1.0,
        sensitivity="forbidden",
        lifecycle="forbidden",
        sql_writable=False,
        cacheable=False,
        vectorizable=False,
        prompt_visible=False,
    ),
}


def preference_definitions() -> dict[str, MemoryKeyDefinition]:
    """返回结构化长期偏好目录的独立映射。"""
    return {
        key: definition
        for key, definition in MEMORY_KEYS.items()
        if definition.memory_type == "preference"
    }


def validate_memory_catalog(
    *,
    memory_types: Mapping[str, MemoryTypeDefinition] = MEMORY_TYPES,
    memory_keys: Mapping[str, MemoryKeyDefinition] = MEMORY_KEYS,
) -> None:
    """在启动阶段验证目录内部约束，错误配置立即终止。"""
    errors: list[str] = []
    for name, definition in memory_types.items():
        if name != definition.name:
            errors.append(f"memory_type_name_mismatch:{name}")
        if definition.lifecycle == "turn" and definition.sql_writable:
            errors.append(f"turn_memory_cannot_write_sql:{name}")
        if definition.structured and definition.vectorizable:
            errors.append(f"structured_memory_type_cannot_vectorize:{name}")
        if definition.lifecycle == "forbidden" and (
            definition.sql_writable
            or definition.cacheable
            or definition.vectorizable
        ):
            errors.append(f"forbidden_memory_storage_enabled:{name}")

    for key, definition in memory_keys.items():
        memory_type = memory_types.get(definition.memory_type)
        if key != definition.memory_key:
            errors.append(f"memory_key_name_mismatch:{key}")
        if memory_type is None:
            errors.append(f"unknown_memory_type:{key}:{definition.memory_type}")
            continue
        if definition.lifecycle != memory_type.lifecycle:
            errors.append(f"memory_lifecycle_mismatch:{key}")
        if definition.ttl_days is not None and definition.ttl_days <= 0:
            errors.append(f"invalid_memory_ttl:{key}")
        if not 0.0 <= definition.confidence_threshold <= 1.0:
            errors.append(f"invalid_memory_confidence:{key}")
        if definition.value_type == "enum":
            if not definition.choices:
                errors.append(f"enum_memory_requires_choices:{key}")
            if definition.default_value not in definition.choices:
                errors.append(f"invalid_memory_default:{key}")
            if definition.vectorizable:
                errors.append(f"structured_enum_cannot_vectorize:{key}")
        if definition.sensitivity == "forbidden" and (
            definition.sql_writable
            or definition.cacheable
            or definition.vectorizable
        ):
            errors.append(f"forbidden_memory_key_storage_enabled:{key}")
        if definition.lifecycle == "turn" and definition.sql_writable:
            errors.append(f"turn_memory_key_cannot_write_sql:{key}")
        if definition.sql_writable and not memory_type.sql_writable:
            errors.append(f"memory_type_disallows_sql:{key}")
        if definition.cacheable and not memory_type.cacheable:
            errors.append(f"memory_type_disallows_cache:{key}")
        if definition.vectorizable and not memory_type.vectorizable:
            errors.append(f"memory_type_disallows_vector:{key}")
        if definition.lifecycle == "forbidden" and definition.prompt_visible:
            errors.append(f"forbidden_memory_key_prompt_visible:{key}")

    if errors:
        raise ValueError("memory_catalog_invalid:" + ",".join(sorted(errors)))


def validate_agent_memory_keys(
    agent_specs: Iterable[AgentMemorySpec],
    *,
    memory_keys: Mapping[str, MemoryKeyDefinition] = MEMORY_KEYS,
) -> None:
    """验证 Agent 白名单仅引用允许注入 Prompt 的已登记字段。"""
    errors: list[str] = []
    for spec in agent_specs:
        for memory_key in spec.memory_keys:
            definition = memory_keys.get(memory_key)
            if definition is None:
                errors.append(f"unknown_agent_memory_key:{spec.agent_id}:{memory_key}")
            elif not definition.prompt_visible:
                errors.append(f"agent_memory_key_not_visible:{spec.agent_id}:{memory_key}")
        for memory_type_name in getattr(spec, "semantic_memory_types", ()):
            memory_type = MEMORY_TYPES.get(memory_type_name)
            if memory_type is None:
                errors.append(
                    f"unknown_agent_semantic_memory_type:"
                    f"{spec.agent_id}:{memory_type_name}"
                )
            elif not memory_type.vectorizable:
                errors.append(
                    f"agent_semantic_memory_type_not_vectorizable:"
                    f"{spec.agent_id}:{memory_type_name}"
                )
    if errors:
        raise ValueError("agent_memory_catalog_invalid:" + ",".join(sorted(errors)))


def validate_memory_configuration(agent_specs: Iterable[AgentMemorySpec]) -> None:
    validate_memory_catalog()
    validate_agent_memory_keys(agent_specs)


__all__ = [
    "MEMORY_KEYS",
    "MEMORY_TYPES",
    "MemoryKeyDefinition",
    "MemoryLifecycle",
    "MemorySensitivity",
    "MemoryTypeDefinition",
    "MemoryValueType",
    "preference_definitions",
    "validate_agent_memory_keys",
    "validate_memory_catalog",
    "validate_memory_configuration",
]
