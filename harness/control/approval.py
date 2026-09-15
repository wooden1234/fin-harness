"""Approval orchestration for the Agent control plane."""

from __future__ import annotations

from typing import Any
import json

from harness.approval.service import validate_decision
from harness.session.types import EventDraft
from harness.tools.errors import enrich_tool_result


class ApprovalCoordinator:
    """Owns approval decisions while Runtime owns tool execution."""

    def validate(self, events, approval_id: str) -> dict[str, Any]:
        return validate_decision(events, approval_id)

    @staticmethod
    def is_allowed(decision: str) -> bool:
        return decision == "allow"

    @staticmethod
    def denied_payload() -> dict[str, Any]:
        return {"ok": False, "error": "denied"}

    async def apply(
        self,
        *,
        store,
        session_id: str,
        asked: dict[str, Any],
        approval_id: str,
        decision: str,
        turn: int,
        step: int,
        run_id: str,
        runtime,
    ) -> dict[str, Any]:
        """Record a decision and produce the corresponding tool result."""
        await store.append(
            session_id,
            EventDraft(
                event_type="approval/decided",
                turn=turn,
                step=step,
                run_id=run_id,
                data={"approval_id": approval_id, "decision": decision},
            ),
        )
        if not self.is_allowed(decision):
            result = self.denied_payload()
        else:
            arguments = asked.get("arguments") or "{}"
            try:
                parsed = json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
            except json.JSONDecodeError:
                parsed = {}
            definition = runtime.resolve(str(asked.get("name") or ""))
            if definition is None:
                result = enrich_tool_result({"ok": False, "error": "unknown_tool"})
            else:
                result = enrich_tool_result(
                    dict(await runtime.pipeline.run(definition, parsed if isinstance(parsed, dict) else {}))
                )
        content = result.get("content")
        if not isinstance(content, str):
            content = json.dumps(result, ensure_ascii=False)
        await store.append(
            session_id,
            EventDraft(
                event_type="tool/result",
                turn=turn,
                step=step,
                run_id=run_id,
                surface_op="append",
                data={
                    "call_id": asked.get("call_id"),
                    "name": asked.get("name"),
                    "ok": bool(result.get("ok", True)),
                    "content": content,
                    "evidence_id": result.get("evidence_id"),
                    "error": result.get("error"),
                    "error_class": result.get("error_class"),
                },
            ),
        )
        return result
