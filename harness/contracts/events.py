"""Session 事件类型。"""

EVENT_TYPES = frozenset(
    {
        "turn/start",
        "turn/end",
        "step/start",
        "step/end",
        "user/message",
        "assistant/chunk",
        "assistant/message",
        "tool/call",
        "tool/result",
        "request/header",
        "request/context",
        "todo/write",
        "inbox/spliced",
        "inbox/claimed",
        "inbox/discarded",
        "approval/asked",
        "approval/decided",
        "compaction/start",
        "compaction/summary",
        "compaction/end",
        "answer/published",
        "session/seed-end",
        "invariant/violation",
    }
)

SURFACE_EVENT_TYPES = frozenset(
    {"user/message", "assistant/message", "tool/result", "compaction/summary"}
)

PUBLIC_SSE_EVENT_TYPES = frozenset(
    {
        "turn/end",
        "tool/call",
        "tool/result",
        "todo/write",
        "approval/asked",
        "approval/decided",
        "answer/published",
        "user/message",
    }
)

TURN_END_REASONS = frozenset(
    {
        "completed",
        "cancelled",
        "error",
        "max_tokens",
        "waiting_approval",
        "persist_failed",
        "rejected",
    }
)

USER_MESSAGE_SOURCES = frozenset(
    {"user", "inject", "steer", "skill", "vision", "legacy", "plugin", "compaction"}
)

MESSAGE_PROJECT_USER_SOURCES = frozenset({"user", "legacy", "vision"})
