"""压力超限时先裁 tool/result，仍超限再事务式摘要；不拆 pair。"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from harness.compaction.meter import count_request_tokens, estimate_tokens, surface_tokens
from harness.compaction.policy import CompactPolicy, retain_limit, target_limit
from harness.compaction.schema import (
    CompactionDelta,
    CompactionDeltaV2,
    CompactionFact,
    CompactionFactV2,
    CompactionItemV2,
    CompactionSummary,
    CompactionSummaryV2,
    render_summary_text,
    render_summary_text_v2,
    stable_item_id,
    summary_v1_to_v2,
)
from harness.contracts.errors import ContextBudgetExhaustedError
from harness.session.surface import SurfaceMessage, derive_messages
from harness.session.surface import messages_for_llm
from harness.session.types import EventDraft, SessionEvent

_SUMMARY_SYSTEM = (
    "你在压缩通用 Agent 的较早上下文。保留事实、决策、约束、偏好、实体、技能/资源引用、"
    "工具续接状态、任务和失败原因。不要编造数据或结论。控制在 400 字以内。"
)

_REAL_USER_SOURCES = frozenset({"user", "legacy", "vision"})
_V2_FIELDS = (
    "facts", "decisions", "constraints", "preferences", "entities", "skills",
    "resources", "tool_state", "completed_items", "open_items", "failures",
    "conversation_notes",
)


def _trim(text: str, policy: CompactPolicy) -> str:
    if len(text) <= policy.prune_chars:
        return text
    return (
        text[: policy.head_chars]
        + "\n…[truncated]…\n"
        + text[-policy.tail_chars :]
    )


def _units(messages: Sequence[SurfaceMessage]) -> list[list[SurfaceMessage]]:
    """把 assistant+后续 tool 收成一组，避免摘要时拆 pair。"""
    grouped: list[list[SurfaceMessage]] = []
    index = 0
    while index < len(messages):
        item = messages[index]
        if item.role == "assistant" and item.tool_calls:
            bundle = [item]
            index += 1
            while index < len(messages) and messages[index].role == "tool":
                bundle.append(messages[index])
                index += 1
            grouped.append(bundle)
            continue
        grouped.append([item])
        index += 1
    return grouped


def _is_incomplete_unit(unit: Sequence[SurfaceMessage]) -> bool:
    if not unit:
        return False
    head = unit[0]
    if head.role != "assistant" or not head.tool_calls:
        return False
    call_ids = {
        str(item.get("call_id") or item.get("id") or "")
        for item in head.tool_calls
        if isinstance(item, dict)
    }
    call_ids.discard("")
    got = {str(item.call_id or "") for item in unit[1:] if item.role == "tool"}
    return bool(call_ids - got)


def _split_prefix(
    messages: Sequence[SurfaceMessage],
    *,
    retain_tokens: int,
    turn: int | None = None,
    seq_to_turn: dict[int, int] | None = None,
) -> tuple[list[SurfaceMessage], list[SurfaceMessage]]:
    units = _units(messages)
    if len(units) < 2:
        return [], list(messages)
    seq_to_turn = seq_to_turn or {}
    pinned: set[int] = set()
    if _is_incomplete_unit(units[-1]):
        pinned.add(len(units) - 1)
    for index in range(len(units) - 1, -1, -1):
        found = False
        for item in units[index]:
            if item.role != "user" or item.source not in _REAL_USER_SOURCES:
                continue
            item_turn = seq_to_turn.get(item.seq)
            if turn is not None and item_turn is not None and item_turn != turn:
                continue
            pinned.add(index)
            found = True
            break
        if found:
            break
    kept_indices: set[int] = {len(units) - 1}
    used = surface_tokens(units[-1])
    cursor = len(units) - 2
    while cursor >= 0:
        chunk_tokens = surface_tokens(units[cursor])
        if used + chunk_tokens > max(1, retain_tokens):
            break
        kept_indices.add(cursor)
        used += chunk_tokens
        cursor -= 1
    kept_indices |= pinned
    prefix: list[SurfaceMessage] = []
    kept: list[SurfaceMessage] = []
    for index, unit in enumerate(units):
        if index in kept_indices:
            kept.extend(unit)
        else:
            prefix.extend(unit)
    return prefix, kept


def _format_for_summary(messages: Sequence[SurfaceMessage]) -> str:
    lines: list[str] = []
    for item in messages:
        content = str(item.content or "").strip()
        if item.role == "user":
            label = "用户" if item.source in _REAL_USER_SOURCES else f"上下文[{item.source}]"
            lines.append(f"{label}：{content}")
        elif item.role == "assistant":
            names = []
            for call in item.tool_calls:
                if isinstance(call, dict):
                    names.append(str(call.get("name") or ""))
            suffix = f"（调用 {', '.join(n for n in names if n)}）" if names else ""
            lines.append(f"助手{suffix}：{content}")
        elif item.role == "tool":
            label = item.name or "tool"
            lines.append(f"工具[{label}]：{content}")
    return "\n".join(lines)


def _seq_to_turn(events: Sequence[SessionEvent]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for event in events:
        if event.turn is None:
            continue
        mapping[event.seq] = int(event.turn)
    return mapping


def _committed_summary_seqs(events: Sequence[SessionEvent]) -> set[int]:
    committed: set[int] = set()
    open_start: int | None = None
    for event in events:
        if event.event_type == "compaction/start":
            open_start = event.seq
        elif event.event_type == "compaction/end" and open_start is not None:
            for item in events:
                if item.event_type == "compaction/summary" and open_start < item.seq < event.seq:
                    committed.add(item.seq)
            open_start = None
    return committed


def latest_committed_summary(events: Sequence[SessionEvent]) -> SessionEvent | None:
    committed = _committed_summary_seqs(events)
    dropped: set[int] = set()
    latest: SessionEvent | None = None
    for event in events:
        if event.event_type == "compaction/summary" and event.seq in committed:
            if event.seq not in dropped:
                latest = event
        if event.source_event_seqs:
            dropped.update(int(seq) for seq in event.source_event_seqs)
    return latest


def load_summary(event: SessionEvent | None) -> CompactionSummary | None:
    if event is None:
        return None
    raw = event.data.get("structured")
    if isinstance(raw, dict):
        try:
            return CompactionSummary.model_validate(raw)
        except (TypeError, ValueError):
            pass
    text = str(event.data.get("content") or event.data.get("summary") or "").strip()
    if text:
        return CompactionSummary(narrative=text)
    return None


def load_summary_v2(event: SessionEvent | None) -> CompactionSummaryV2 | None:
    """读取 v2；历史 v1 在内存中升级，不改写事件。"""
    if event is None:
        return None
    raw = event.data.get("structured")
    version = int(event.data.get("compaction_schema_version") or 0)
    if isinstance(raw, dict) and (version == 2 or int(raw.get("schema_version") or 0) == 2):
        try:
            return CompactionSummaryV2.model_validate(raw)
        except (TypeError, ValueError):
            pass
    legacy = load_summary(event)
    return summary_v1_to_v2(legacy) if legacy is not None else None


def _item_payload(item: CompactionItemV2) -> dict[str, Any]:
    if isinstance(item, CompactionFactV2):
        if item.company or item.period or item.metric:
            return {"company": item.company, "period": item.period, "metric": item.metric}
        return {"note": item.note or item.content, "evidence_id": item.evidence_id}
    key = item.metadata.get("key") or item.metadata.get("canonical_name")
    return {"key": key} if key else {"content": item.content}


def _stamp_item(item: CompactionItemV2, *, kind: str, turn: int) -> CompactionItemV2:
    payload = item.model_dump()
    payload["created_turn"] = int(payload.get("created_turn") or turn)
    payload["updated_turn"] = int(payload.get("updated_turn") or turn)
    if not str(payload.get("id") or "").strip():
        payload["id"] = stable_item_id(kind, _item_payload(item))
    return type(item).model_validate(payload)


def _upsert_items(
    old: Sequence[CompactionItemV2],
    new: Sequence[CompactionItemV2],
    *,
    kind: str,
    turn: int,
    removed: set[str],
) -> list[CompactionItemV2]:
    merged: dict[str, CompactionItemV2] = {}
    order: list[str] = []
    for raw in (*old, *new):
        item = _stamp_item(raw, kind=kind, turn=turn)
        if item.id in removed:
            continue
        if item.id not in merged:
            order.append(item.id)
        merged[item.id] = item
    return [merged[item_id] for item_id in order]


def _legacy_delta_to_v2(delta: CompactionDelta, *, turn: int) -> CompactionDeltaV2:
    facts = []
    for fact in delta.new_facts:
        raw = fact.model_dump()
        fact_turn = int(raw.pop("turn", 0) or turn)
        facts.append(
            CompactionFactV2(
                **raw,
                id=stable_item_id(
                    "fact",
                    {key: raw.get(key) for key in ("company", "period", "metric", "note", "evidence_id")},
                ),
                content=fact.note,
                created_turn=fact_turn,
                updated_turn=fact_turn,
                source=fact.source_tool or "compaction",
            )
        )
    def items(kind: str, values: Sequence[str]) -> list[CompactionItemV2]:
        return [
            CompactionItemV2(
                id=stable_item_id(kind, value), content=value,
                created_turn=turn, updated_turn=turn,
            )
            for value in values if str(value).strip()
        ]
    return CompactionDeltaV2(
        new_facts=facts,
        new_open_items=items("open", delta.new_open_items),
        new_completed_items=items("completed", delta.new_completed_items),
        narrative_delta=delta.narrative_delta,
    )


def merge_summary_v2(
    old: CompactionSummaryV2 | None,
    delta: CompactionDeltaV2,
    *,
    turn: int,
    todos: Sequence[dict[str, str]] | None = None,
    max_narrative_chars: int = 400,
) -> CompactionSummaryV2:
    previous = old or CompactionSummaryV2()
    removed = {str(value) for value in (*delta.resolved_ids, *delta.superseded_ids)}
    updates: dict[str, Any] = {}
    for field in _V2_FIELDS:
        updates[field] = _upsert_items(
            getattr(previous, field), getattr(delta, f"new_{field}"),
            kind=field, turn=turn, removed=removed,
        )
    if todos:
        open_items: list[CompactionItemV2] = []
        completed: list[CompactionItemV2] = []
        for row in todos:
            content = str(row.get("content") or "").strip()
            if not content:
                continue
            item = CompactionItemV2(
                id=stable_item_id("todo", content), content=content,
                created_turn=turn, updated_turn=turn, source="todo",
                priority="high" if row.get("status") == "in_progress" else "normal",
            )
            if row.get("status") in {"pending", "in_progress"}:
                open_items.append(item)
            elif row.get("status") == "completed":
                completed.append(item.model_copy(update={"status": "completed"}))
        known = {item.content for item in (*open_items, *completed)}
        updates["open_items"] = [*open_items, *[item for item in updates["open_items"] if item.content not in known]]
        updates["completed_items"] = [*completed, *[item for item in updates["completed_items"] if item.content not in known]]
    completed_content = {item.content for item in updates["completed_items"] if item.content}
    updates["open_items"] = [
        item for item in updates["open_items"] if item.content not in completed_content
    ]
    narrative = " ".join(
        value.strip() for value in (previous.narrative, delta.narrative_delta) if value.strip()
    )
    updates["narrative"] = clip_narrative(narrative, max_chars=max_narrative_chars)
    return CompactionSummaryV2(**updates)


def _protected(item: CompactionItemV2, field: str, *, turn: int) -> bool:
    if field == "open_items" and item.status == "active":
        return True
    if field == "completed_items" and item.updated_turn == turn:
        return True
    if field in {"constraints", "preferences", "entities"} and item.status == "active":
        return item.priority in {"high", "critical"} or field != "constraints"
    if field == "resources" and item.metadata.get("rebuildable") is False:
        return True
    if field == "failures" and item.metadata.get("retryable") is False and item.status == "active":
        return True
    if field == "facts" and (
        item.priority in {"high", "critical"} or item.updated_turn == turn
    ):
        return True
    return item.priority == "critical" or item.updated_turn == turn and field == "open_items"


def protected_summary_tokens(summary: CompactionSummaryV2, *, turn: int) -> int:
    payload = CompactionSummaryV2()
    updates = {
        field: [item for item in getattr(summary, field) if _protected(item, field, turn=turn)]
        for field in _V2_FIELDS
    }
    payload = payload.model_copy(update=updates)
    return estimate_tokens(render_summary_text_v2(payload))


def shrink_summary_v2(
    summary: CompactionSummaryV2,
    *,
    turn: int,
    token_budget: int,
    drop_narrative: bool = True,
) -> CompactionSummaryV2:
    """确定性按生命周期、类别、优先级和轮次收缩。"""
    current = summary
    inactive = {"resolved", "superseded", "expired"}
    current = current.model_copy(update={
        field: [item for item in getattr(current, field) if item.status not in inactive]
        for field in _V2_FIELDS
    })
    if estimate_tokens(render_summary_text_v2(current)) <= token_budget:
        return current
    if drop_narrative:
        current = current.model_copy(update={"narrative": ""})
    order = (
        "conversation_notes", "failures", "tool_state", "completed_items", "resources",
        "decisions", "facts", "skills", "preferences", "entities", "constraints",
    )
    rank = {"low": 0, "normal": 1, "high": 2, "critical": 3}
    while estimate_tokens(render_summary_text_v2(current)) > token_budget:
        candidates: list[tuple[int, int, int, str, str]] = []
        for field_index, field in enumerate(order):
            for item in getattr(current, field):
                if _protected(item, field, turn=turn):
                    continue
                candidates.append((field_index, rank[item.priority], item.updated_turn, field, item.id))
        if not candidates:
            break
        _, _, _, victim_field, victim_id = min(candidates)
        current = current.model_copy(update={
            victim_field: [item for item in getattr(current, victim_field) if item.id != victim_id]
        })
    return current


def current_turn_todos(events: Sequence[SessionEvent], *, turn: int) -> list[dict[str, str]]:
    latest: list[dict[str, str]] | None = None
    for event in events:
        if event.event_type != "todo/write" or event.turn != turn:
            continue
        rows = event.data.get("todos") or []
        if isinstance(rows, list):
            latest = [item for item in rows if isinstance(item, dict)]
    return latest or []


def _unique(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _fact_key(fact: CompactionFact) -> tuple[str, ...]:
    company = fact.company.strip()
    period = fact.period.strip()
    metric = fact.metric.strip()
    if company or period or metric:
        return ("metric", company, period, metric)
    return ("note", fact.note.strip(), fact.evidence_id.strip())


def merge_facts(old: Sequence[CompactionFact], new: Sequence[CompactionFact]) -> list[CompactionFact]:
    merged: dict[tuple[str, ...], CompactionFact] = {}
    order: list[tuple[str, ...]] = []
    for fact in (*old, *new):
        key = _fact_key(fact)
        if key not in merged:
            order.append(key)
        merged[key] = fact
    return [merged[key] for key in order]


def prune_facts(
    facts: Sequence[CompactionFact],
    *,
    turn: int,
    max_turns: int,
    keep_current: bool = True,
) -> list[CompactionFact]:
    floor = max(1, turn - max(1, max_turns) + 1)
    kept = [fact for fact in facts if int(fact.turn or 0) >= floor]
    if keep_current:
        current = [fact for fact in facts if int(fact.turn or 0) == turn]
        seen = {_fact_key(item) for item in kept}
        for fact in current:
            key = _fact_key(fact)
            if key not in seen:
                kept.append(fact)
                seen.add(key)
    return kept


def drop_oldest_turn_facts(facts: Sequence[CompactionFact], *, turn: int) -> list[CompactionFact]:
    older = [int(fact.turn or 0) for fact in facts if int(fact.turn or 0) != turn and int(fact.turn or 0) > 0]
    if not older:
        return list(facts)
    victim = min(older)
    return [fact for fact in facts if int(fact.turn or 0) != victim]


def clip_narrative(text: str, *, max_chars: int) -> str:
    value = str(text or "").strip()
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip()


def merge_summary(
    old: CompactionSummary | None,
    delta: CompactionDelta,
    *,
    turn: int,
    same_turn: bool,
    todos: Sequence[dict[str, str]] | None = None,
    max_turns: int = 10,
    max_narrative_chars: int = 400,
) -> CompactionSummary:
    previous = old or CompactionSummary()
    stamped: list[CompactionFact] = []
    for fact in delta.new_facts:
        payload = fact.model_dump()
        if int(payload.get("turn") or 0) <= 0:
            payload["turn"] = turn
        stamped.append(CompactionFact.model_validate(payload))
    facts = prune_facts(
        merge_facts(previous.facts, stamped),
        turn=turn,
        max_turns=max_turns,
    )
    new_open = _unique(delta.new_open_items)
    new_done = _unique(delta.new_completed_items)
    extra_open = new_open
    extra_done = new_done
    if same_turn:
        open_items = _unique([*previous.open_items, *new_open])
        open_items = [item for item in open_items if item not in set(new_done)]
        completed_items = _unique([*previous.completed_items, *new_done])
    else:
        open_items = list(new_open)
        completed_items = list(new_done)
    if todos:
        todo_open = [
            str(item.get("content") or "").strip()
            for item in todos
            if str(item.get("status") or "") in {"pending", "in_progress"}
        ]
        todo_done = [
            str(item.get("content") or "").strip()
            for item in todos
            if str(item.get("status") or "") == "completed"
        ]
        known = set(_unique([*todo_open, *todo_done]))
        open_items = _unique(todo_open)
        completed_items = _unique(todo_done)
        for item in extra_open:
            if item not in known:
                open_items.append(item)
                known.add(item)
        for item in extra_done:
            if item not in known:
                completed_items.append(item)
                known.add(item)
                open_items = [row for row in open_items if row != item]
    narrative = clip_narrative(
        " ".join(part for part in (previous.narrative, delta.narrative_delta) if part.strip()),
        max_chars=max_narrative_chars,
    )
    return CompactionSummary(
        facts=facts,
        open_items=open_items,
        completed_items=completed_items,
        narrative=narrative,
    )


def _fit_summary(
    summary: CompactionSummary,
    *,
    kept: Sequence[SurfaceMessage],
    token_limit: int,
    turn: int,
    max_narrative_chars: int,
) -> CompactionSummary:
    current = summary
    current = current.model_copy(
        update={"narrative": clip_narrative(current.narrative, max_chars=max_narrative_chars)}
    )
    rendered = render_summary_text(current)
    while estimate_tokens(rendered) + surface_tokens(kept) >= token_limit:
        shrunk = drop_oldest_turn_facts(current.facts, turn=turn)
        if len(shrunk) == len(current.facts):
            break
        current = current.model_copy(update={"facts": shrunk})
        rendered = render_summary_text(current)
    return current


async def _complete_summary(llm: Any, prompt: str) -> str:
    complete = getattr(llm, "complete", None)
    if complete is None:
        return ""
    text = await complete(system=_SUMMARY_SYSTEM, prompt=prompt)
    return str(text or "").strip()


async def _compress_narrative(llm: Any, narrative: str, *, max_chars: int) -> str:
    if not narrative.strip():
        return ""
    complete = getattr(llm, "complete", None)
    if not callable(complete):
        return clip_narrative(narrative, max_chars=max_chars)
    try:
        value = await complete(
            system="只压缩给定补充说明，不添加事实，不修改结构化状态。",
            prompt=f"压缩到 {max_chars} 字以内：\n{narrative}",
        )
    except Exception:  # noqa: BLE001
        value = narrative
    return clip_narrative(str(value or narrative), max_chars=max_chars)


async def _complete_delta_v2(llm: Any, prompt: str, *, turn: int) -> CompactionDeltaV2 | None:
    complete_structured = getattr(llm, "complete_structured", None)
    if complete_structured is None:
        return None
    try:
        result = await complete_structured(CompactionDeltaV2, prompt)
    except Exception:  # noqa: BLE001
        return None
    if isinstance(result, CompactionDeltaV2):
        return result
    try:
        return CompactionDeltaV2.model_validate(result)
    except (TypeError, ValueError):
        try:
            legacy = result if isinstance(result, CompactionDelta) else CompactionDelta.model_validate(result)
            return _legacy_delta_to_v2(legacy, turn=turn)
        except (TypeError, ValueError):
            return None


async def _trim_longest_tool(
    *,
    store: Any,
    session_id: str,
    events: Sequence[Any],
    turn: int,
    run_id: str,
    trigger: str,
    policy: CompactPolicy,
) -> bool:
    tool_results = [
        event
        for event in events
        if event.event_type == "tool/result" and event.surface_op == "append"
        and event.seq not in {
            seq
            for replacement in events
            if replacement.event_type == "tool/result" and replacement.surface_op == "replace"
            for seq in replacement.source_event_seqs
        }
    ]
    if not tool_results:
        return False
    target = max(tool_results, key=lambda event: len(str(event.data.get("content") or "")))
    content = str(target.data.get("content") or "")
    if len(content) <= policy.prune_chars and trigger != "context-overflow":
        return False
    trimmed = _trim(content, policy)
    if trimmed == content:
        return False
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
    return True


def _build_delta_prompt(
    *,
    old: CompactionSummaryV2 | None,
    older: Sequence[SurfaceMessage],
    current: Sequence[SurfaceMessage],
    turn: int,
) -> str:
    parts = [
        f"当前轮次是 {turn}。请只返回 JSON。",
        "输出 CompactionDeltaV2，只提取即将移除窗口相对已有摘要的增量。",
        "每项填写稳定 id、content、created_turn、updated_turn、status、priority、source。",
        "把客观事实、决策、约束、偏好、实体、技能引用、资源、工具续接状态、任务、失败和对话说明分别归类。",
        "Skill 只保存名称、版本/来源、用途和恢复状态，不复制完整指令。",
        "new_open_items / new_completed_items 只描述当前轮；更早轮次不能产生未完成事项。",
        "已解决或被替代的条目分别写入 resolved_ids / superseded_ids。无法归类但必须保留的信息写 narrative_delta。",
    ]
    if old is not None:
        parts.append("已有摘要 JSON（不要复述成新 facts 的重复项，只抽窗口增量）：")
        parts.append(old.model_dump_json(ensure_ascii=False))
    if older:
        parts.append("【更早已结束轮次，只抽 facts】")
        parts.append(_format_for_summary(older))
    if current:
        parts.append("【当前轮次已发生、即将被摘掉的步骤，可抽 open/completed】")
        parts.append(_format_for_summary(current))
    return "\n\n".join(parts)


async def _append_summary_events(
    *,
    store: Any,
    session_id: str,
    turn: int,
    run_id: str,
    source_seqs: tuple[int, ...],
    content: str,
    structured: dict[str, Any] | None = None,
) -> None:
    await store.append(
        session_id,
        EventDraft(
            event_type="compaction/start",
            turn=turn,
            run_id=run_id,
            data={"reason": "summary", "dropped": len(source_seqs)},
        ),
    )
    data: dict[str, Any] = {
        "content": content,
        "summary": content,
        "tokens": estimate_tokens(content),
    }
    if structured is not None:
        data["structured"] = structured
        data["compaction_schema_version"] = 2
    await store.append(
        session_id,
        EventDraft(
            event_type="compaction/summary",
            turn=turn,
            run_id=run_id,
            surface_op="replace",
            source_event_seqs=source_seqs,
            data=data,
        ),
    )
    await store.append(
        session_id,
        EventDraft(
            event_type="compaction/end",
            turn=turn,
            run_id=run_id,
            data={"reason": "summary"},
        ),
    )


async def _write_summary(
    *,
    store: Any,
    session_id: str,
    llm: Any,
    events: Sequence[SessionEvent],
    messages: Sequence[SurfaceMessage],
    turn: int,
    run_id: str,
    policy: CompactPolicy,
    token_limit: int,
) -> bool:
    seq_to_turn = _seq_to_turn(events)
    prefix, kept = _split_prefix(
        messages,
        retain_tokens=retain_limit(policy),
        turn=turn,
        seq_to_turn=seq_to_turn,
    )
    old_event = latest_committed_summary(events)
    old_summary = load_summary_v2(old_event)
    extract = [item for item in prefix if item.source != "compaction"]
    if not extract:
        return False
    source_seqs = tuple(
        dict.fromkeys(
            ([old_event.seq] if old_event is not None else [])
            + [item.seq for item in prefix]
        )
    )
    if not source_seqs:
        return False
    older = [item for item in extract if seq_to_turn.get(item.seq, turn) < turn]
    current = [item for item in extract if seq_to_turn.get(item.seq, turn) >= turn]
    prompt = _build_delta_prompt(old=old_summary, older=older, current=current, turn=turn)
    delta = await _complete_delta_v2(llm, prompt, turn=turn)
    if delta is not None:
        if not current:
            delta = delta.model_copy(update={"new_open_items": [], "new_completed_items": []})
        merged = merge_summary_v2(
            old_summary,
            delta,
            turn=turn,
            todos=current_turn_todos(events, turn=turn),
            max_narrative_chars=policy.max_narrative_chars,
        )
        summary_budget = max(
            1,
            token_limit - surface_tokens(kept),
        )
        fitted = shrink_summary_v2(
            merged, turn=turn, token_budget=summary_budget, drop_narrative=False
        )
        if estimate_tokens(render_summary_text_v2(fitted)) > summary_budget and fitted.narrative:
            narrative = await _compress_narrative(
                llm, fitted.narrative, max_chars=max(64, policy.max_narrative_chars // 2)
            )
            fitted = fitted.model_copy(update={"narrative": narrative})
        if estimate_tokens(render_summary_text_v2(fitted)) > summary_budget:
            fitted = shrink_summary_v2(fitted, turn=turn, token_budget=summary_budget)
        content = render_summary_text_v2(fitted)
        if not content.strip():
            return False
        await _append_summary_events(
            store=store,
            session_id=session_id,
            turn=turn,
            run_id=run_id,
            source_seqs=source_seqs,
            content=content,
            structured=fitted.model_dump(),
        )
        return True
    try:
        summary = await _complete_summary(llm, prompt)
    except Exception:  # noqa: BLE001
        return False
    if not summary:
        excerpt = _format_for_summary(extract or prefix)
        budget = max(200, policy.head_chars // 2)
        summary = excerpt[:budget] + ("\n…[truncated]…" if len(excerpt) > budget else "")
    if not summary.strip():
        return False
    fallback = CompactionSummaryV2(
        narrative=clip_narrative(summary, max_chars=policy.max_narrative_chars)
    )
    await _append_summary_events(
        store=store,
        session_id=session_id,
        turn=turn,
        run_id=run_id,
        source_seqs=source_seqs,
        content=render_summary_text_v2(fallback),
        structured=fallback.model_dump(),
    )
    return True


async def _rewrite_existing_summary(
    *,
    store: Any,
    session_id: str,
    llm: Any,
    events: Sequence[SessionEvent],
    turn: int,
    run_id: str,
    message_budget: int,
    policy: CompactPolicy,
) -> bool:
    """没有新原文可 fold 时，只收缩当前已提交摘要。"""
    old_event = latest_committed_summary(events)
    old = load_summary_v2(old_event)
    if old_event is None or old is None:
        return False
    others = [item for item in derive_messages(events) if item.source != "compaction"]
    summary_budget = max(1, message_budget - surface_tokens(others))
    fitted = shrink_summary_v2(old, turn=turn, token_budget=summary_budget, drop_narrative=False)
    if estimate_tokens(render_summary_text_v2(fitted)) > summary_budget and fitted.narrative:
        fitted = fitted.model_copy(update={
            "narrative": await _compress_narrative(
                llm, fitted.narrative, max_chars=max(64, policy.max_narrative_chars // 2)
            )
        })
    if estimate_tokens(render_summary_text_v2(fitted)) > summary_budget:
        fitted = shrink_summary_v2(fitted, turn=turn, token_budget=summary_budget)
    if fitted.model_dump() == old.model_dump():
        return False
    await _append_summary_events(
        store=store,
        session_id=session_id,
        turn=turn,
        run_id=run_id,
        source_seqs=(old_event.seq,),
        content=render_summary_text_v2(fitted),
        structured=fitted.model_dump(),
    )
    return True


def _is_over(
    messages: Sequence[SurfaceMessage],
    *,
    token_limit: int,
    trigger: str,
) -> bool:
    return surface_tokens(messages) >= token_limit or trigger == "context-overflow"


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
    system: str = "",
    tools: Sequence[Mapping[str, Any]] = (),
    enforce_budget: bool = False,
) -> bool:
    policy = policy or CompactPolicy()
    stages: list[str] = []

    async def current_tokens(events: Sequence[SessionEvent]) -> int:
        if not system and not tools:
            return surface_tokens(derive_messages(events))
        return await count_request_tokens(
            llm,
            system=system,
            messages=messages_for_llm(events),
            tools=tools,
        )

    events = await store.load_events(session_id)
    messages = derive_messages(events)
    measured = await current_tokens(events)
    if measured < token_limit and trigger != "context-overflow":
        return False

    target = target_limit(policy) if (system or tools) else token_limit
    changed = False
    while measured > target:
        trimmed = await _trim_longest_tool(
            store=store,
            session_id=session_id,
            events=events,
            turn=turn,
            run_id=run_id,
            trigger=trigger,
            policy=policy,
        )
        if not trimmed:
            break
        changed = True
        if "tool-trim" not in stages:
            stages.append("tool-trim")
        events = await store.load_events(session_id)
        messages = derive_messages(events)
        measured = await current_tokens(events)

    if measured > target and allow_llm and llm is not None:
        overhead = max(0, measured - surface_tokens(messages))
        summarized = await _write_summary(
            store=store,
            session_id=session_id,
            llm=llm,
            events=events,
            messages=messages,
            turn=turn,
            run_id=run_id,
            policy=policy,
            token_limit=max(1, target - overhead),
        )
        changed = changed or summarized
        if summarized:
            stages.extend(["summary-merge", "summary-shrink"])
            events = await store.load_events(session_id)
            measured = await current_tokens(events)

    if measured > target and allow_llm and llm is not None:
        overhead = max(0, measured - surface_tokens(derive_messages(events)))
        rewritten = await _rewrite_existing_summary(
            store=store,
            session_id=session_id,
            llm=llm,
            events=events,
            turn=turn,
            run_id=run_id,
            message_budget=max(1, target - overhead),
            policy=policy,
        )
        changed = changed or rewritten
        if rewritten:
            stages.append("summary-recompress")
            events = await store.load_events(session_id)
            measured = await current_tokens(events)

    if enforce_budget and measured > target:
        latest = load_summary_v2(latest_committed_summary(events))
        protected = protected_summary_tokens(latest, turn=turn) if latest is not None else 0
        await store.append(
            session_id,
            EventDraft(
                event_type="invariant/violation",
                turn=turn,
                run_id=run_id,
                data={
                    "code": "context_budget_exhausted",
                    "context_window": policy.context_window,
                    "trigger_tokens": token_limit,
                    "target_tokens": target,
                    "final_tokens": measured,
                    "protected_tokens": protected,
                    "stages": stages,
                },
            ),
        )
        raise ContextBudgetExhaustedError(
            context_window=policy.context_window,
            final_tokens=measured,
            stages=stages,
            protected_tokens=protected,
        )
    return changed
