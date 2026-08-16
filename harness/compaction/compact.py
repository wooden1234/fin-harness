"""压力超限时裁 tool/result；不拆 pair。"""

from __future__ import annotations

from typing import Any

from harness.compaction.meter import surface_tokens
from harness.compaction.policy import CompactPolicy
from harness.session.surface import derive_messages
from harness.session.types import EventDraft


def _trim(text: str, policy: CompactPolicy) -> str:
    if len(text) <= policy.prune_chars:
        return text
    return (
        text[: policy.head_chars]
        + "\n…[truncated]…\n"
        + text[-policy.tail_chars :]
    )


async def maybe_compact(
    *,
    store: Any,
    session_id: str,
    llm: Any = None,
    turn: int,
    run_id: str,
    token_limit: int,
    allow_llm: bool = False,
    trigger: str = "pressure",
    policy: CompactPolicy | None = None,
) -> bool:
    from harness.compaction.policy import CompactPolicy as Policy

    policy = policy or Policy()
    events = await store.load_events(session_id)
    messages = derive_messages(events)
    if surface_tokens(messages) < token_limit and trigger != "context-overflow":
        return False
    tool_results = [
        event
        for event in events
        if event.event_type == "tool/result" and event.surface_op == "append"
    ]
    if not tool_results:
        return False
    target = max(tool_results, key=lambda event: len(str(event.data.get("content") or "")))
    content = str(target.data.get("content") or "")
    if len(content) <= policy.prune_chars and trigger != "context-overflow":
        return False
    trimmed = _trim(content, policy)
    await store.append(
        session_id,
        EventDraft(
            event_type="tool/result",
            turn=turn,
            run_id=run_id,
            surface_op="replace",
            source_event_seqs=(target.seq,),
            data={
                "call_id": target.data.get("call_id"),
                "name": target.data.get("name"),
                "ok": target.data.get("ok", True),
                "content": trimmed,
                "evidence_id": target.data.get("evidence_id"),
            },
        ),
    )
    _ = llm, allow_llm
    return True
