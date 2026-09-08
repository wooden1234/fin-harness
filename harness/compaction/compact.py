"""压力超限时先裁 tool/result，仍超限再事务式摘要；不拆 pair。"""

from __future__ import annotations

from typing import Any, Sequence

from harness.compaction.meter import estimate_tokens, surface_tokens
from harness.compaction.policy import CompactPolicy, retain_limit
from harness.compaction.schema import (
    CompactionDelta,
    CompactionFact,
    CompactionSummary,
    render_summary_text,
)
from harness.session.surface import SurfaceMessage, derive_messages
from harness.session.types import EventDraft, SessionEvent

_SUMMARY_SYSTEM = (
    "你在压缩金融助手的较早对话。只写事实摘要：问题、已查到的数字、证据编号、未完成项。"
    "不要编造数据或结论。不要给出买卖建议。控制在 400 字以内。"
)

_REAL_USER_SOURCES = frozenset({"user", "legacy", "vision"})


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
            lines.append(f"用户：{content}")
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


async def _complete_delta(llm: Any, prompt: str) -> CompactionDelta | None:
    complete_structured = getattr(llm, "complete_structured", None)
    if complete_structured is None:
        return None
    try:
        result = await complete_structured(CompactionDelta, prompt)
    except Exception:  # noqa: BLE001
        return None
    if isinstance(result, CompactionDelta):
        return result
    validator = getattr(CompactionDelta, "model_validate", None)
    if not callable(validator):
        return None
    try:
        return validator(result)
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
    old: CompactionSummary | None,
    older: Sequence[SurfaceMessage],
    current: Sequence[SurfaceMessage],
    turn: int,
) -> str:
    parts = [
        f"当前轮次是 {turn}。请只返回 JSON。",
        "new_facts 可来自两段窗口；每条 fact 填写 turn。",
        "new_open_items / new_completed_items 只能描述当前轮次窗口里尚未结束的工作。",
        "更早轮次已经结束，不要把它们写成未完成。套不进指标的约束、失败原因写入 narrative_delta。",
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
    old_summary = load_summary(old_event)
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
    delta = await _complete_delta(llm, prompt)
    if delta is not None:
        if not current:
            delta = delta.model_copy(update={"new_open_items": [], "new_completed_items": []})
        same_turn = old_event is not None and old_event.turn == turn
        merged = merge_summary(
            old_summary,
            delta,
            turn=turn,
            same_turn=bool(same_turn),
            todos=current_turn_todos(events, turn=turn),
            max_turns=policy.max_summary_turns,
            max_narrative_chars=policy.max_narrative_chars,
        )
        fitted = _fit_summary(
            merged,
            kept=kept,
            token_limit=token_limit,
            turn=turn,
            max_narrative_chars=policy.max_narrative_chars,
        )
        content = render_summary_text(fitted)
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
    await _append_summary_events(
        store=store,
        session_id=session_id,
        turn=turn,
        run_id=run_id,
        source_seqs=source_seqs,
        content=summary,
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
) -> bool:
    policy = policy or CompactPolicy()
    events = await store.load_events(session_id)
    messages = derive_messages(events)
    if not _is_over(messages, token_limit=token_limit, trigger=trigger):
        return False

    changed = await _trim_longest_tool(
        store=store,
        session_id=session_id,
        events=events,
        turn=turn,
        run_id=run_id,
        trigger=trigger,
        policy=policy,
    )
    events = await store.load_events(session_id)
    messages = derive_messages(events)
    if not _is_over(messages, token_limit=token_limit, trigger=trigger):
        return changed
    if not allow_llm or llm is None:
        return changed
    summarized = await _write_summary(
        store=store,
        session_id=session_id,
        llm=llm,
        events=events,
        messages=messages,
        turn=turn,
        run_id=run_id,
        policy=policy,
        token_limit=token_limit,
    )
    return changed or summarized
