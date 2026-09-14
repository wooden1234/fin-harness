"""Control-plane policies used by the execution loop."""

from __future__ import annotations

from typing import Any

from harness.prompt.assembler import assemble_system
from harness.prompt.preferences import load_preference_context
from harness.prompt.sections import default_sections, preference_sections
from harness.session.store import SessionStore
from harness.tools.analysis import TOOL_ID as FINALIGN_TOOL_ID, bind_finalign_analyze, finalign_is_ready
from harness.tools.memory import memory_tool_definitions
from harness.tools.runtime import ToolRuntime
from harness.tools.skill import inject_skill_context
from harness.tools.todo import todo_write_definition
from harness.control.approval import ApprovalCoordinator


class AgentControl:
    """Product controls injected into the Runtime."""

    def __init__(self, store: SessionStore, session_id: str) -> None:
        self.store = store
        self.session_id = session_id
        self.approvals = ApprovalCoordinator()

    async def request_context(self, *, turn: int, run_id: str, base_runtime: ToolRuntime):
        events = await self.store.load_events(self.session_id)
        loaded = await load_preference_context(
            store=self.store, session_id=self.session_id, events=events, turn=turn
        )
        sections = list(default_sections())
        sections.extend(preference_sections(loaded.preferences, loaded.turn_overrides))
        runtime = self.bind_runtime(turn=turn, run_id=run_id, base_runtime=base_runtime)
        return assemble_system(sections), runtime.openai_tools()

    def bind_runtime(self, *, turn: int, run_id: str, base_runtime: ToolRuntime) -> ToolRuntime:
        runtime = base_runtime.with_extra([
            todo_write_definition(self.store, self.session_id, turn=turn, run_id=run_id),
            *memory_tool_definitions(self.store, self.session_id, run_id=run_id),
        ])
        if runtime.resolve(FINALIGN_TOOL_ID):
            if finalign_is_ready():
                runtime = runtime.replace_handler(
                    FINALIGN_TOOL_ID,
                    bind_finalign_analyze(self.store, self.session_id, turn=turn),
                )
            else:
                runtime = runtime.exclude(FINALIGN_TOOL_ID, "finalign_analyze")
        return runtime

    def validate_approval(self, events, approval_id: str):
        return self.approvals.validate(events, approval_id)

    async def inject_skill(self, result: dict[str, Any], *, turn: int, run_id: str) -> None:
        await inject_skill_context(
            self.store, self.session_id, result, turn=turn, run_id=run_id
        )
