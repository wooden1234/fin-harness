"""Agent HTTP：session kernel + loop，SSE 只投影 public 事件。"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, Optional

from fastapi import APIRouter, Depends, Form, HTTPException
from fastapi.responses import StreamingResponse

from agents.guardrails.input.secrets import check_secrets
from app.core.cache import cache_metrics_snapshot
from app.core.logger import get_logger
from app.core.redis_client import redis_metrics
from app.core.security import get_current_user
from app.schemas.user import AuthUser
from app.services.agent.agent_run_service import AgentRunService
from app.services.conversation.conversation_service import ConversationService
from app.services.memory.memory_command import parse_memory_rule_action
from app.services.memory.memory_episodic_extraction import (
    decide_post_turn_trigger,
    detect_important_state_change,
    is_substantive_progress,
)
from app.services.persistence.outbox_service import OutboxService
from harness.agent.result import RunResult
from harness.approval.service import pending_approvals
from harness.contracts.errors import AgentBusyError, ApprovalError
from harness.projection.sse import project_session_event
from harness.runtime import product_manager

router = APIRouter(prefix="/agent", tags=["agent"])
logger = get_logger(service="agent")


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _decision_from_text(text: str) -> str:
    lowered = text.strip().lower()
    if lowered in {"deny", "reject", "拒绝", "否", "不允许", "cancel", "取消"}:
        return "deny"
    return "allow"


@router.get("/health")
async def agent_health(current_user: AuthUser = Depends(get_current_user)):
    return {"status": "agent module ready", "user_id": current_user.id}


@router.get("/metrics")
async def agent_metrics(current_user: AuthUser = Depends(get_current_user)):
    metrics: dict[str, Any] = await AgentRunService.metrics()
    metrics["redis"] = redis_metrics()
    metrics["cache"] = cache_metrics_snapshot()
    return metrics


async def _prepare_conversation(
    conversation_id: Optional[str],
    current_user: AuthUser,
) -> int | None:
    if conversation_id is None:
        return None
    try:
        conversation_pk = int(conversation_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="conversation_id 格式错误") from exc
    conversation = await ConversationService.get_owned_conversation(
        conversation_pk,
        current_user.id,
        current_user.tenant_id,
    )
    if conversation is None:
        raise HTTPException(status_code=404, detail="会话不存在或无权访问")
    return conversation_pk


async def _effective_query(
    query: str,
    attachment_id: Optional[str],
    current_user: AuthUser,
) -> tuple[str, str | None]:
    attachment_id_value = str(attachment_id or "").strip() or None
    if not attachment_id_value:
        return query, None
    from app.services.attachments.attachment_service import (
        download_attachment_bytes,
        require_owned_attachment,
    )
    from app.services.vision.image_understanding import (
        compose_effective_query,
        understand_image,
    )

    meta = await require_owned_attachment(
        attachment_id_value,
        user_id=current_user.id,
        tenant_id=current_user.tenant_id,
    )
    understanding = None
    try:
        image_bytes = download_attachment_bytes(meta)
        understanding = await understand_image(
            image_bytes=image_bytes,
            content_type=meta.content_type,
            user_text=query,
        )
    except Exception:
        logger.exception("vision understand failed attachment_id={}", attachment_id_value)
    return compose_effective_query(query, understanding), attachment_id_value


async def _stream_agent(
    *,
    agent,
    store,
    run_task: asyncio.Task[RunResult],
    run_id: str,
    conversation_pk: int | None,
    conversation_key,
    current_user: AuthUser,
    query: str,
    user_message,
    memory_action,
    started: float,
) -> Any:
    after_seq = 0
    try:
        while not run_task.done():
            events = await store.wait_events(agent.session_id, after_seq=after_seq, timeout=0.2)
            for event in events:
                after_seq = event.seq
                for payload in project_session_event(event):
                    if payload.get("type") == "interrupt":
                        payload["conversation_id"] = str(conversation_key)
                    yield _sse(payload)
        result = await run_task
        events = await store.load_events(agent.session_id, after_seq=after_seq)
        for event in events:
            after_seq = event.seq
            for payload in project_session_event(event):
                if payload.get("type") == "interrupt":
                    payload["conversation_id"] = str(conversation_key)
                yield _sse(payload)

        if result.waiting_approval:
            await AgentRunService.mark_graph_completed(
                run_id,
                summary_snapshot={
                    "content": "",
                    "route": "waiting_approval",
                    "execution_status": "waiting_approval",
                    "duration_ms": round((time.monotonic() - started) * 1000, 2),
                },
            )
            return

        published = result.published_answer or ""
        follow_ups = list(result.follow_ups or [])
        assistant_message_id = None
        persistence_status = "not_required"
        if conversation_pk is not None and published:
            try:
                assistant_message = await ConversationService.save_assistant_message(
                    user_id=current_user.id,
                    conversation_id=conversation_pk,
                    content=published,
                    run_id=run_id,
                    tenant_id=current_user.tenant_id,
                )
                assistant_message_id = assistant_message.id
                await AgentRunService.mark_persisted(
                    run_id,
                    response_message_id=assistant_message_id,
                )
                persistence_status = "persisted"
            except Exception as save_err:
                logger.error("Failed to save conversation: {}", save_err)
                await AgentRunService.mark_persist_pending(run_id, error_message=str(save_err))
                try:
                    await OutboxService.enqueue_assistant_persist(
                        run_id=run_id,
                        user_id=current_user.id,
                        conversation_id=conversation_pk,
                        content=published,
                        tenant_id=current_user.tenant_id,
                    )
                except Exception:
                    logger.exception("failed to enqueue assistant persistence: {}", run_id)
                persistence_status = "pending_retry"
        elif conversation_pk is None:
            persistence_status = "session_only"

        await AgentRunService.mark_graph_completed(
            run_id,
            summary_snapshot={
                "content": published,
                "route": result.finish_reason,
                "execution_status": result.finish_reason,
                "duration_ms": round((time.monotonic() - started) * 1000, 2),
            },
        )

        if memory_action.kind == "implicit":
            try:
                await OutboxService.enqueue_memory_extraction(
                    run_id=run_id,
                    user_id=current_user.id,
                    tenant_id=current_user.tenant_id,
                    conversation_id=conversation_pk,
                    message_id=user_message.id if user_message else None,
                    source_text=query,
                    agent_id="fin_agent",
                    task_id="memory-extraction",
                    trace_id=run_id,
                )
            except Exception:
                logger.exception("failed to enqueue memory extraction: {}", run_id)

        try:
            context_window = await OutboxService.episodic_context_window(
                tenant_id=current_user.tenant_id,
                user_id=current_user.id,
                conversation_id=conversation_pk,
                query=query,
                final_response=published,
            )
            episodic_decision = decide_post_turn_trigger(
                query=query,
                final_response=published,
                execution_status=result.finish_reason,
                task_count=0,
                turn_count=context_window.turn_count,
                uncompressed_tokens=context_window.uncompressed_tokens,
            )
            needs_progress_check = conversation_pk is not None and is_substantive_progress(
                query=query,
                final_response=published,
                execution_status=result.finish_reason,
            )
            if not detect_important_state_change(query) and (
                episodic_decision.should_enqueue or needs_progress_check
            ):
                await OutboxService.enqueue_episodic_extraction(
                    run_id=run_id,
                    user_id=current_user.id,
                    tenant_id=current_user.tenant_id,
                    conversation_id=conversation_pk,
                    message_id=assistant_message_id,
                    query=query,
                    final_response=published,
                    execution_status=result.finish_reason,
                    task_count=0,
                    trigger_reasons=episodic_decision.reasons or ("progress_check",),
                    forced=episodic_decision.forced,
                    agent_id="fin_agent",
                    task_id="episodic-extraction",
                    trace_id=run_id,
                )
        except Exception:
            logger.exception("failed to enqueue episodic extraction: {}", run_id)

        if result.error and not published:
            yield _sse({"type": "error", "run_id": run_id, "message": "本轮未能发布回答。"})
            return
        yield _sse(
            {
                "type": "done",
                "run_id": run_id,
                "message_id": assistant_message_id,
                "persistence_status": persistence_status,
                "content": published,
                "citations": [],
                "follow_ups": follow_ups,
                "charts": [],
            }
        )
    except asyncio.CancelledError:
        agent.cancel()
        run_task.cancel()
        logger.info("agent query cancelled run_id={}", run_id)
        raise
    except AgentBusyError as exc:
        yield _sse({"type": "error", "run_id": run_id, "message": str(exc)})
    except ApprovalError as exc:
        yield _sse({"type": "error", "run_id": run_id, "message": str(exc)})
    except Exception as exc:
        logger.exception("agent_query stream error")
        try:
            await AgentRunService.mark_failed(run_id, error_message=str(exc))
        except Exception:
            logger.exception("failed to update agent run status: {}", run_id)
        yield _sse({"type": "error", "run_id": run_id, "message": str(exc)})


@router.post("/query")
async def agent_query(
    query: str = Form(...),
    conversation_id: Optional[str] = Form(None),
    client_message_id: Optional[str] = Form(None),
    attachment_id: Optional[str] = Form(None),
    current_user: AuthUser = Depends(get_current_user),
):
    secret_decision = check_secrets(query)
    if not secret_decision.should_continue:
        raise HTTPException(
            status_code=400,
            detail="输入中包含密码、Token、验证码、私钥或其他凭据，请删除秘密值后重试",
        )
    conversation_pk = await _prepare_conversation(conversation_id, current_user)
    effective_query, attachment_id_value = await _effective_query(
        query, attachment_id, current_user
    )
    conversation_key = conversation_pk if conversation_pk is not None else uuid.uuid4()
    run_id = str(uuid.uuid4())
    client_message_id = client_message_id or str(uuid.uuid4())
    manager = product_manager()
    try:
        agent = await manager.get(
            tenant_id=current_user.tenant_id,
            user_id=current_user.id,
            conversation_id=conversation_key,
            owner_id=str(current_user.id),
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    events = await manager.store.load_events(agent.session_id)
    if pending_approvals(events):
        raise HTTPException(status_code=409, detail="会话等待审批，请调用 /api/agent/resume")
    try:
        await AgentRunService.create_run(
            run_id=run_id,
            user_id=current_user.id,
            conversation_id=conversation_pk,
            tenant_id=current_user.tenant_id,
            thread_id=agent.session_id,
            trace_id=run_id,
        )
        await AgentRunService.mark_running(run_id)
        user_message = None
        if conversation_pk is not None:
            user_message = await ConversationService.save_user_message(
                user_id=current_user.id,
                conversation_id=conversation_pk,
                tenant_id=current_user.tenant_id,
                content=query,
                run_id=run_id,
                client_message_id=client_message_id,
                message_type="image" if attachment_id_value else "text",
            )
        memory_action = parse_memory_rule_action(query)
    except AgentBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as start_err:
        try:
            await AgentRunService.mark_failed(run_id, error_message=str(start_err))
        except Exception:
            logger.exception("failed to mark startup run failed: {}", run_id)
        raise

    started = time.monotonic()
    run_task = asyncio.create_task(agent.prompt(effective_query, source="user"))

    async def process_stream():
        async for chunk in _stream_agent(
            agent=agent,
            store=manager.store,
            run_task=run_task,
            run_id=run_id,
            conversation_pk=conversation_pk,
            conversation_key=conversation_key,
            current_user=current_user,
            query=query,
            user_message=user_message,
            memory_action=memory_action,
            started=started,
        ):
            yield chunk

    response = StreamingResponse(process_stream(), media_type="text/event-stream")
    response.headers["X-Conversation-ID"] = str(conversation_key)
    response.headers["X-Agent-Run-ID"] = run_id
    response.headers["X-Session-ID"] = agent.session_id
    response.headers["Cache-Control"] = "no-cache"
    return response


@router.post("/resume")
async def agent_resume(
    conversation_id: str = Form(...),
    query: str = Form(""),
    approval_id: Optional[str] = Form(None),
    decision: Optional[str] = Form(None),
    current_user: AuthUser = Depends(get_current_user),
):
    conversation_pk = await _prepare_conversation(conversation_id, current_user)
    conversation_key = conversation_pk if conversation_pk is not None else conversation_id
    run_id = str(uuid.uuid4())
    manager = product_manager()
    try:
        agent = await manager.get(
            tenant_id=current_user.tenant_id,
            user_id=current_user.id,
            conversation_id=conversation_key,
            owner_id=str(current_user.id),
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    events = await manager.store.load_events(agent.session_id)
    pending = pending_approvals(events)
    if not pending:
        raise HTTPException(status_code=409, detail="没有待审批的工具调用")
    target_id = approval_id or str(pending[-1].get("approval_id") or "")
    chosen = decision or _decision_from_text(query)
    note = query.strip()
    if note and chosen == "allow" and note.lower() not in {"allow", "ok", "同意", "批准", "是"}:
        await agent.inject(note, source="user")
    try:
        await AgentRunService.create_run(
            run_id=run_id,
            user_id=current_user.id,
            conversation_id=conversation_pk,
            tenant_id=current_user.tenant_id,
            thread_id=agent.session_id,
            trace_id=run_id,
        )
        await AgentRunService.mark_running(run_id)
    except Exception as start_err:
        try:
            await AgentRunService.mark_failed(run_id, error_message=str(start_err))
        except Exception:
            logger.exception("failed to mark resume run failed: {}", run_id)
        raise
    started = time.monotonic()
    run_task = asyncio.create_task(
        agent.resume_approval(target_id, decision=chosen)
    )

    async def process_stream():
        async for chunk in _stream_agent(
            agent=agent,
            store=manager.store,
            run_task=run_task,
            run_id=run_id,
            conversation_pk=conversation_pk,
            conversation_key=conversation_key,
            current_user=current_user,
            query=query,
            user_message=None,
            memory_action=parse_memory_rule_action(""),
            started=started,
        ):
            yield chunk

    response = StreamingResponse(process_stream(), media_type="text/event-stream")
    response.headers["X-Conversation-ID"] = str(conversation_key)
    response.headers["X-Agent-Run-ID"] = run_id
    response.headers["X-Session-ID"] = agent.session_id
    response.headers["Cache-Control"] = "no-cache"
    return response
