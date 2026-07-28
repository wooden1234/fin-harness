"""V2 任务身份、尝试和幂等键管理。"""

from __future__ import annotations

import hashlib
import json
import uuid

from agents.orchestrator.analyzer.validate import assert_plan_capabilities
from agents.orchestrator.contracts import TaskPlan, TaskSpec
from agents.orchestrator.domain_scope import ensure_task_domain_scope


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_idempotency_key(
    task: TaskSpec,
    *,
    scope: str = "",
) -> str:
    """为同一逻辑任务的同一次尝试生成稳定幂等键。"""
    payload = {
        "scope": scope,
        "logical_task_id": task.logical_task_id or task.task_id,
        "attempt_number": task.attempt_number,
        "agent_id": task.agent_id,
        "objective": task.objective,
        "input_data": task.input_data,
    }
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:20]
    return f"v2:{payload['logical_task_id']}:{task.attempt_number}:{digest}"


def ensure_task_identity(task: TaskSpec, *, scope: str = "") -> TaskSpec:
    """为任务补齐身份字段；已存在的身份保持不变以支持重放。"""
    logical_task_id = task.logical_task_id or task.task_id
    attempt_id = task.attempt_id or uuid.uuid4().hex
    attempt_number = max(1, int(task.attempt_number or 1))
    normalized = task.model_copy(
        update={
            "logical_task_id": logical_task_id,
            "attempt_id": attempt_id,
            "attempt_number": attempt_number,
        }
    )
    if normalized.idempotency_key:
        return normalized
    return normalized.model_copy(
        update={"idempotency_key": build_idempotency_key(normalized, scope=scope)}
    )


def validate_task_plan(plan: TaskPlan, *, scope: str = "") -> TaskPlan:
    """重新构造并校验 TaskPlan，确保依赖和 capability 都经过验证。"""
    tasks = [
        ensure_task_identity(
            ensure_task_domain_scope(task),
            scope=scope or plan.plan_id,
        )
        for task in plan.tasks
    ]
    validated = TaskPlan.model_validate(
        {
            **plan.model_dump(mode="python"),
            "tasks": [task.model_dump(mode="python") for task in tasks],
        }
    )
    assert_plan_capabilities(validated)

    identities = [
        (
            task.logical_task_id,
            task.attempt_number,
            task.attempt_id,
            task.idempotency_key,
        )
        for task in validated.tasks
    ]
    if len({(item[0], item[1]) for item in identities}) != len(identities):
        raise ValueError("logical_task_attempt_must_be_unique")
    if len({item[2] for item in identities}) != len(identities):
        raise ValueError("attempt_id_must_be_unique")
    if len({item[3] for item in identities}) != len(identities):
        raise ValueError("idempotency_key_must_be_unique")
    return validated


__all__ = ["build_idempotency_key", "ensure_task_identity", "validate_task_plan"]
