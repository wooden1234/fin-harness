"""Control-plane policies used by the execution loop."""

from __future__ import annotations

from typing import Any

from harness.prompt.assembler import assemble_system
from harness.prompt.preferences import PreferenceContext, load_preference_context
from harness.prompt.sections import PromptSection, default_sections, preference_sections
from harness.session.store import SessionStore
from harness.tools.analysis import TOOL_ID as FINALIGN_TOOL_ID, bind_finalign_analyze, finalign_is_ready
from harness.tools.ledger import ledger_tool_definitions
from harness.tools.log_read import read_tool_log_definition
from harness.tools.memory import memory_tool_definitions
from harness.tools.runtime import ToolRuntime
from harness.tools.skill import inject_skill_context
from harness.tools.todo import todo_write_definition
from harness.control.approval import ApprovalCoordinator


def _ledger_open_task_text(app_user_id: str | None) -> str:
    if not app_user_id or not str(app_user_id).isdigit():
        return ""
    try:
        from retrieval.services.ledger_tasks import open_task_prompt

        return open_task_prompt(int(app_user_id))
    except Exception:
        return ""


class AgentControl:
    """Product controls injected into the Runtime."""

    def __init__(
        self,
        store: SessionStore,
        session_id: str,
        *,
        app_user_id: str | None = None,
    ) -> None:
        self.store = store
        self.session_id = session_id
        self.app_user_id = app_user_id
        self.approvals = ApprovalCoordinator()
        self._effective_preferences: dict[str, dict[str, Any]] = {}
        self._preference_contexts: dict[str, PreferenceContext] = {}

    async def request_context(self, *, turn: int, run_id: str, base_runtime: ToolRuntime):
        loaded = self._preference_contexts.get(run_id)
        if loaded is None:
            events = await self.store.load_events(self.session_id)
            loaded = await load_preference_context(
                store=self.store, session_id=self.session_id, events=events, turn=turn
            )
            if not loaded.memory_load_succeeded:
                loaded = await load_preference_context(
                    store=self.store, session_id=self.session_id, events=events, turn=turn
                )
            self._preference_contexts[run_id] = loaded
            self._effective_preferences[run_id] = loaded.effective()
        sections = list(default_sections())
        task_text = _ledger_open_task_text(self.app_user_id)
        if task_text:
            sections.append(PromptSection("ledger_open_tasks", 45, task_text))
        sections.extend(
            preference_sections(
                loaded.preferences,
                loaded.turn_overrides,
                loaded.session_overrides,
            )
        )
        runtime = self.bind_runtime(turn=turn, run_id=run_id, base_runtime=base_runtime)
        return assemble_system(sections), runtime.openai_tools()

    def bind_runtime(self, *, turn: int, run_id: str, base_runtime: ToolRuntime) -> ToolRuntime:
        runtime = base_runtime.with_extra([
            todo_write_definition(self.store, self.session_id, turn=turn, run_id=run_id),
            read_tool_log_definition(self.store, self.session_id),
            *memory_tool_definitions(self.store, self.session_id, run_id=run_id),
            *ledger_tool_definitions(self.store, self.session_id),
        ])
        if runtime.resolve(FINALIGN_TOOL_ID):
            if finalign_is_ready():
                runtime = runtime.replace_handler(
                    FINALIGN_TOOL_ID,
                    bind_finalign_analyze(
                        self.store,
                        self.session_id,
                        turn=turn,
                        preferences=self._effective_preferences.get(run_id, {}),
                    ),
                )
            else:
                runtime = runtime.exclude(FINALIGN_TOOL_ID, "finalign_analyze")
        return runtime

    def validate_approval(self, events, approval_id: str):
        return self.approvals.validate(events, approval_id)

    def finish_run(self, run_id: str) -> None:
        self._effective_preferences.pop(run_id, None)
        self._preference_contexts.pop(run_id, None)

    def effective_preferences(self, run_id: str) -> dict[str, Any]:
        return dict(self._effective_preferences.get(run_id, {}))

    async def inject_skill(self, result: dict[str, Any], *, turn: int, run_id: str) -> None:
        await inject_skill_context(
            self.store, self.session_id, result, turn=turn, run_id=run_id
        )
