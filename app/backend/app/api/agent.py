import asyncio
from collections.abc import AsyncIterable, AsyncIterator
import json
import time
import uuid
from typing import Any, Optional
from fastapi import APIRouter, Depends, Form, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from agents.checkpoint import make_thread_config
from agents.guardrails.input.secrets import check_secrets
from agents.orchestrator.graph import get_orchestrator_graph
from agents.runtime_context import AgentRuntimeContext
from app.api.agent_progress import (
    VISIBLE_TASK_NODES,
    build_public_step_event,
    build_todo_snapshot_event,
    extract_agent_todos_snapshot,
    map_node_to_public_step,
)
from app.core.config import settings
from app.core.logger import get_logger
from app.core.cache import cache_metrics_snapshot
from app.core.redis_client import redis_metrics
from app.core.security import get_current_user
from app.schemas.user import AuthUser
from app.services.conversation.conversation_service import ConversationService
from app.services.agent.agent_run_service import AgentRunService
from app.services.persistence.outbox_service import OutboxService
from app.services.conversation.conversation_lock_service import (
    ConversationBusyError,
    ConversationLockService,
)
from app.services.agent.checkpoint_rebuild_service import CheckpointRebuildService
from app.services.agent.checkpoint_registry_service import CheckpointRegistryService
from app.services.memory.memory_command import parse_memory_rule_action
from app.services.memory.memory_episodic_extraction import (
    decide_post_turn_trigger,
    detect_important_state_change,
    is_substantive_progress,
)


router = APIRouter(prefix="/agent", tags=["agent"])
logger = get_logger(service="agent")

# 候选答案在 general_agent / summarize 生成时尚未通过合规审查，禁止提前发送。
STREAMABLE_ANSWER_NODES = frozenset({"final_answer"})


async def _stream_with_timeout(
    stream: AsyncIterable[Any],
    timeout_seconds: float | None,
) -> AsyncIterator[Any]:
    """限制图流式执行总时长，并让取消继续传播到图和子任务。"""
    if timeout_seconds is None:
        async for chunk in stream:
            yield chunk
        return
    if timeout_seconds <= 0:
        raise TimeoutError("agent_run_deadline_exceeded")
    async with asyncio.timeout(timeout_seconds):
        async for chunk in stream:
            yield chunk

def _sse(payload: dict) -> str:
    """格式化为 SSE 行：data: {...}\\n\\n"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _extract_incremental_text(previous_content: str, current_content: str) -> str:
    """提取本次流式事件相对已发送内容的新增部分。"""
    if not current_content:
        return ""
    if not previous_content:
        return current_content
    if current_content == previous_content:
        return ""
    if current_content.startswith(previous_content):
        return current_content[len(previous_content):]
    if previous_content.startswith(current_content):
        return ""

    max_overlap = min(len(previous_content), len(current_content))
    for overlap in range(max_overlap, 0, -1):
        if previous_content.endswith(current_content[:overlap]):
            return current_content[overlap:]
    return current_content


def _extract_final_response(values: dict) -> str:
    """从图状态中提取最终可展示的回答文本。"""
    for msg in reversed(list(values.get("messages") or [])):
        if isinstance(msg, AIMessage):
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    summary = values.get("summary")
    if isinstance(summary, str):
        return summary
    return ""

@router.get("/health")
async def agent_health(current_user: AuthUser = Depends(get_current_user)):
    """W3 前占位：验证 Agent 路由走 JWT"""
    return {"status": "agent module ready", "user_id": current_user.id}


@router.get("/metrics")
async def agent_metrics(current_user: AuthUser = Depends(get_current_user)):
    """返回运行与 outbox 基础指标；生产环境应接入 Prometheus。"""
    metrics: dict[str, Any] = await AgentRunService.metrics()
    metrics["redis"] = redis_metrics()
    metrics["cache"] = cache_metrics_snapshot()
    return metrics

@router.post("/query")
async def agent_query(
    query: str = Form(...),
    conversation_id: Optional[str] = Form(None),
    client_message_id: Optional[str] = Form(None),
    current_user: AuthUser = Depends(get_current_user)):
    secret_decision = check_secrets(query)
    if not secret_decision.should_continue:
        raise HTTPException(
            status_code=400,
            detail="输入中包含密码、Token、验证码、私钥或其他凭据，请删除秘密值后重试",
        )
    conversation_pk: int | None = None
    if conversation_id is not None:
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

    conversation_key = conversation_pk if conversation_pk is not None else uuid.uuid4()
    lock_token: str | None = None
    if conversation_pk is not None:
        try:
            lock_token = await ConversationLockService.acquire(conversation_pk)
        except ConversationBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
    thread_config = make_thread_config(
        conversation_key,
        user_id=current_user.id,
        tenant_id=current_user.tenant_id,
    )
    run_id = str(uuid.uuid4())
    client_message_id = client_message_id or str(uuid.uuid4())
    try:
        await AgentRunService.create_run(
            run_id=run_id,
            user_id=current_user.id,
            conversation_id=conversation_pk,
            tenant_id=current_user.tenant_id,
            thread_id=thread_config["configurable"]["thread_id"],
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
            )
        memory_action = parse_memory_rule_action(query)
    except Exception as start_err:
        try:
            await AgentRunService.mark_failed(run_id, error_message=str(start_err))
        except Exception:
            logger.exception("failed to mark startup run failed: {}", run_id)
        if conversation_pk is not None and lock_token is not None:
            await ConversationLockService.release(conversation_pk, lock_token)
        raise
    try:
        graph = get_orchestrator_graph(with_checkpointer=True)
    except Exception:
        if conversation_pk is not None and lock_token is not None:
            await ConversationLockService.release(conversation_pk, lock_token)
        raise
    input_payload = {"messages": [HumanMessage(content=query)]}
    runtime_context = AgentRuntimeContext.from_user(
        current_user,
        conversation_id=conversation_key,
        run_id=run_id,
        deadline_seconds=(
            settings.MAIN_AGENT_API_DEADLINE_SEC
        ),
        max_concurrency=settings.AGENT_V2_MAX_CONCURRENCY,
    )
    # 只发送 final_answer 审查后的内容，避免候选答案先于合规结果泄露。
    async def process_stream():
        nonlocal graph, thread_config
        assistant_full_response = ""
        assistant_message_id: int | None = None
        generating_answer_active = False
        emitted_main_tool_steps: set[str] = set()
        last_main_todo_snapshot: tuple[tuple[str, str], ...] | None = None
        try:
            if conversation_pk is not None:
                await CheckpointRebuildService.rebuild_if_missing(
                    conversation_id=conversation_pk,
                    user_id=current_user.id,
                    tenant_id=current_user.tenant_id,
                    thread_config=thread_config,
                    graph=graph,
                    exclude_run_id=run_id,
                )
            async for chunk in _stream_with_timeout(
                graph.astream(
                    input_payload,
                    config=thread_config,
                    context=runtime_context,
                    stream_mode=["messages", "tasks", "updates", "custom"],
                    subgraphs=True,
                ),
                runtime_context.remaining_seconds(),
            ):
                if not isinstance(chunk, tuple) or len(chunk) != 3:
                    continue
                namespace, mode, data = chunk
                ns_tuple = namespace if isinstance(namespace, tuple) else ()

                if mode == "custom" and isinstance(data, dict):
                    if data.get("kind") == "main_finalization":
                        yield _sse({
                            "type": "step",
                            "id": "main-finalization",
                            "label": "资料覆盖完成，正在整理答案",
                            "status": "running",
                            "category": "answer",
                            "short_label": "整理答案",
                        })
                        continue
                    if data.get("kind") != "main_tool_progress":
                        continue
                    step_id = str(data.get("step_id") or "")
                    if not step_id:
                        continue
                    family = str(data.get("source_family") or "tool")
                    labels = {
                        "weather": "天气数据",
                        "market": "市场数据",
                        "web": "联网搜索",
                        "research": "研报与公告",
                        "financial": "财务数据",
                        "knowledge": "知识库",
                        "calculation": "受限计算",
                    }
                    status = str(data.get("status") or "running")
                    if status in {"done", "error"}:
                        emitted_main_tool_steps.add(step_id)
                    yield _sse({
                        "type": "step",
                        "id": step_id,
                        "label": (
                            f"正在查询{labels.get(family, '资料')}"
                            if status == "running"
                            else f"已完成{labels.get(family, '资料查询')}"
                        ),
                        "status": status,
                        "category": family,
                        "short_label": labels.get(family, "工具"),
                    })
                    continue

                if mode == "tasks" and isinstance(data, dict):
                    node_name = data.get("name")
                    if not node_name:
                        continue

                    if "result" in data or "error" in data:
                        if node_name == "supervisor" and not ns_tuple:
                            result_payload = data.get("result")
                            if isinstance(result_payload, dict):
                                route = result_payload.get("route")
                                if route:
                                    yield _sse({"type": "meta", "route": route})
                    if node_name not in VISIBLE_TASK_NODES:
                        continue

                    public_step = map_node_to_public_step(node_name)
                    if not public_step:
                        continue

                    if "input" in data and "result" not in data and "error" not in data:
                        event = build_public_step_event(public_step, "running")
                        if event:
                            if public_step == "generating_answer":
                                generating_answer_active = True
                            yield _sse(event)
                        continue

                    if "result" in data or "error" in data:
                        status = "error" if data.get("error") else "done"
                        event = build_public_step_event(public_step, status)
                        if event:
                            if public_step == "generating_answer" and status == "done":
                                generating_answer_active = False
                            yield _sse(event)
                    continue

                if mode == "updates" and isinstance(data, dict):
                    in_main_deep_agent = any(
                        str(part).split(":", 1)[0] == "main_deep_agent"
                        for part in ns_tuple
                    ) or "main_deep_agent" in data
                    if in_main_deep_agent:
                        todo_snapshot = extract_agent_todos_snapshot(data)
                        if todo_snapshot is not None:
                            fingerprint = tuple(
                                (item["content"], item["status"])
                                for item in todo_snapshot
                            )
                            if fingerprint != last_main_todo_snapshot:
                                last_main_todo_snapshot = fingerprint
                                yield _sse(build_todo_snapshot_event(todo_snapshot))
                    main_update = data.get("main_deep_agent")
                    if isinstance(main_update, dict):
                        journal = main_update.get("main_agent_journal") or {}
                        for index, entry in enumerate(journal.get("entries") or [], start=1):
                            tool_id = str(entry.get("tool_id") or "tool")
                            step_id = f"main-tool-{index}-{tool_id}"
                            if step_id in emitted_main_tool_steps:
                                continue
                            emitted_main_tool_steps.add(step_id)
                            family = str(entry.get("source_family") or "tool")
                            labels = {
                                "weather": "天气数据",
                                "market": "市场数据",
                                "web": "联网搜索",
                                "research": "研报与公告",
                                "financial": "财务数据",
                                "knowledge": "知识库",
                                "calculation": "受限计算",
                            }
                            yield _sse({
                                "type": "step",
                                "id": step_id,
                                "label": f"已完成{labels.get(family, '资料查询')}",
                                "status": "done" if entry.get("status") == "completed" else "error",
                                "category": family,
                                "short_label": labels.get(family, "工具"),
                            })
                    continue

                if mode != "messages":
                    continue

                msg, metadata = data
                node = metadata.get("langgraph_node")
                if node not in STREAMABLE_ANSWER_NODES:
                    continue
                if not getattr(msg, "content", None):
                    continue
                if getattr(msg, "additional_kwargs", {}).get("tool_calls"):
                    continue
                if not isinstance(msg, AIMessage):
                    continue
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                incremental_content = _extract_incremental_text(
                    assistant_full_response,
                    content,
                )
                if not incremental_content:
                    continue
                if generating_answer_active:
                    event = build_public_step_event("generating_answer", "done")
                    if event:
                        yield _sse(event)
                    generating_answer_active = False
                assistant_full_response += incremental_content
                yield _sse({"type": "token", "content": incremental_content})

            state = await graph.aget_state(thread_config)
            values = (state.values if state else {}) or {}
            final_todos = extract_agent_todos_snapshot(
                values.get("main_agent_journal") or {}
            )
            if final_todos is not None:
                final_fingerprint = tuple(
                    (item["content"], item["status"])
                    for item in final_todos
                )
                if final_fingerprint != last_main_todo_snapshot:
                    last_main_todo_snapshot = final_fingerprint
                    yield _sse(build_todo_snapshot_event(final_todos))
            checkpoint_id = None
            if state is not None:
                checkpoint_id = (state.config or {}).get("configurable", {}).get("checkpoint_id")
            graph_completion = AgentRunService.mark_graph_completed(
                run_id,
                checkpoint_id=str(checkpoint_id) if checkpoint_id else None,
                summary_snapshot={
                    "content": _extract_final_response(values),
                    "citations": values.get("citations") or [],
                    "route": values.get("execution_mode") or values.get("route"),
                    "execution_mode": values.get("execution_mode"),
                    "execution_status": values.get("execution_status"),
                    "budget_tier": runtime_context.budget_tier,
                    "model_rounds": (values.get("main_agent_journal") or {}).get("model_rounds", 0),
                    "tool_calls": len((values.get("main_agent_journal") or {}).get("entries", [])),
                    "source_families": (values.get("main_agent_journal") or {}).get("source_families", []),
                    "evidence_count": len(values.get("evidence") or []),
                    "statement_coverage": getattr(values.get("quality_report"), "claim_coverage", 0.0),
                    **dict(values.get("main_quality_metrics") or {}),
                    "finalization_started_at": (values.get("main_agent_journal") or {}).get("finalization_started_at"),
                    "tool_budget_exhausted": sum(
                        1
                        for entry in (values.get("main_agent_journal") or {}).get("entries", [])
                        if "budget_exhausted" in str(entry.get("error") or "")
                    ),
                    "insufficient_tool_result": sum(
                        1
                        for entry in (values.get("main_agent_journal") or {}).get("entries", [])
                        if entry.get("error") == "insufficient_tool_result"
                    ),
                    "salvaged": (
                        values.get("execution_mode") == "partial"
                        and bool(values.get("evidence"))
                        and (
                            "timeout" in str((values.get("main_agent_journal") or {}).get("agent_status") or "")
                            or bool((values.get("main_agent_journal") or {}).get("soft_deadline_reached"))
                        )
                    ),
                    "salvage_eligible": (
                        "timeout" in str((values.get("main_agent_journal") or {}).get("agent_status") or "")
                        or bool((values.get("main_agent_journal") or {}).get("soft_deadline_reached"))
                    ) and bool(values.get("evidence")),
                    "timed_out": (
                        "timeout" in str((values.get("main_agent_journal") or {}).get("agent_status") or "")
                        or bool((values.get("main_agent_journal") or {}).get("soft_deadline_reached"))
                    ),
                    "unauthorized_tool_attempts": sum(
                        1
                        for entry in (values.get("main_agent_journal") or {}).get("entries", [])
                        if entry.get("error") == "tool_not_authorized"
                    ),
                    "duration_ms": round(
                        (time.monotonic() - runtime_context.started_monotonic)
                        * 1000,
                        2,
                    ),
                    "compliance_action": values.get("compliance_action"),
                    "compliance_reason_code": values.get("compliance_reason_code"),
                },
            )
            if conversation_pk is not None:
                await asyncio.gather(
                    graph_completion,
                    CheckpointRegistryService.record(
                        conversation_id=conversation_pk,
                        user_id=current_user.id,
                        tenant_id=current_user.tenant_id,
                        thread_id=thread_config["configurable"]["thread_id"],
                        checkpoint_id=str(checkpoint_id) if checkpoint_id else None,
                    ),
                )
            else:
                await graph_completion
            citations = values.get("citations") or []
            final_response = _extract_final_response(values)
            if not assistant_full_response and final_response:
                assistant_full_response = final_response
            if generating_answer_active:
                event = build_public_step_event("generating_answer", "done")
                if event:
                    yield _sse(event)
                generating_answer_active = False
            # 持久化消息到数据库
            persistence_status = "not_required"
            if conversation_pk is not None and assistant_full_response:
                try:
                    assistant_message = await ConversationService.save_assistant_message(
                        user_id=current_user.id,
                        conversation_id=conversation_pk,
                        content=assistant_full_response,
                        run_id=run_id,
                        tenant_id=current_user.tenant_id,
                    )
                    assistant_message_id = assistant_message.id
                    logger.info(
                        "conversation saved: user={}, conv={}",
                        current_user.id, conversation_id,
                    )
                    await AgentRunService.mark_persisted(
                        run_id,
                        response_message_id=assistant_message_id,
                    )
                    persistence_status = "persisted"
                except Exception as save_err:
                    logger.error("Failed to save conversation: {}", save_err)
                    await AgentRunService.mark_persist_pending(
                        run_id,
                        error_message=str(save_err),
                    )
                    try:
                        await OutboxService.enqueue_assistant_persist(
                            run_id=run_id,
                            user_id=current_user.id,
                            conversation_id=conversation_pk,
                            content=assistant_full_response,
                            tenant_id=current_user.tenant_id,
                        )
                    except Exception:
                        logger.exception("failed to enqueue assistant persistence: {}", run_id)
                    persistence_status = "pending_retry"
            elif conversation_pk is None:
                # 无业务会话时，checkpoint 已保存运行态，消息留档由调用方后续关联。
                persistence_status = "checkpoint_only"

            async def enqueue_implicit_memory() -> None:
                if memory_action.kind != "implicit":
                    return
                try:
                    await OutboxService.enqueue_memory_extraction(
                        run_id=run_id,
                        user_id=current_user.id,
                        tenant_id=current_user.tenant_id,
                        conversation_id=conversation_pk,
                        message_id=user_message.id if user_message else None,
                        source_text=query,
                        agent_id="orchestrator",
                        task_id="memory-extraction",
                        trace_id=run_id,
                    )
                except Exception:
                    # 异步偏好提取是增强能力，登记失败不能改变本轮回答结果。
                    logger.exception("failed to enqueue memory extraction: {}", run_id)

            task_plan = values.get("task_plan")
            if isinstance(task_plan, dict):
                tasks = list(task_plan.get("tasks") or [])
            elif task_plan is not None:
                tasks = list(getattr(task_plan, "tasks", []) or [])
            else:
                tasks = []
            execution_status = str(values.get("execution_status") or "")
            async def load_episodic_context_window():
                try:
                    return await OutboxService.episodic_context_window(
                        tenant_id=current_user.tenant_id,
                        user_id=current_user.id,
                        conversation_id=conversation_pk,
                        query=query,
                        final_response=final_response,
                    )
                except Exception:
                    # 预判读取失败不影响回答；后台 worker 入队后仍会重新权威核对。
                    logger.exception(
                        "failed to load episodic context window: {}",
                        run_id,
                    )
                    return await OutboxService.episodic_context_window(
                        tenant_id=current_user.tenant_id,
                        user_id=current_user.id,
                        conversation_id=None,
                        query=query,
                        final_response=final_response,
                    )

            _, context_window = await asyncio.gather(
                enqueue_implicit_memory(),
                load_episodic_context_window(),
            )
            episodic_decision = decide_post_turn_trigger(
                query=query,
                final_response=final_response,
                execution_status=execution_status,
                task_count=len(tasks),
                turn_count=context_window.turn_count,
                uncompressed_tokens=context_window.uncompressed_tokens,
            )
            needs_progress_check = (
                conversation_pk is not None
                and is_substantive_progress(
                    query=query,
                    final_response=final_response,
                    execution_status=execution_status,
                )
            )
            if (
                not detect_important_state_change(query)
                and (episodic_decision.should_enqueue or needs_progress_check)
            ):
                try:
                    trigger_reasons = (
                        episodic_decision.reasons
                        if episodic_decision.reasons
                        else ("progress_check",)
                    )
                    await OutboxService.enqueue_episodic_extraction(
                        run_id=run_id,
                        user_id=current_user.id,
                        tenant_id=current_user.tenant_id,
                        conversation_id=conversation_pk,
                        message_id=assistant_message_id,
                        query=query,
                        final_response=final_response,
                        execution_status=execution_status,
                        task_count=len(tasks),
                        trigger_reasons=trigger_reasons,
                        forced=episodic_decision.forced,
                        agent_id="orchestrator",
                        task_id="episodic-extraction",
                        trace_id=run_id,
                    )
                except Exception:
                    # 事件摘要属于回答后的增强能力，登记失败不改变本轮结果。
                    logger.exception(
                        "failed to enqueue episodic extraction: {}",
                        run_id,
                    )

            yield _sse({
                "type": "done",
                "run_id": run_id,
                "message_id": assistant_message_id,
                "persistence_status": persistence_status,
                "content": final_response,
                "citations": citations,
                "route": values.get("execution_mode") or values.get("route"),
                "compliance_action": values.get("compliance_action"),
                "compliance_reason_code": values.get("compliance_reason_code"),
            })

        except asyncio.CancelledError:
            logger.info("agent query cancelled run_id={}", run_id)
            raise
        except TimeoutError:
            logger.warning("agent query deadline exceeded run_id={}", run_id)
            try:
                await AgentRunService.mark_failed(
                    run_id,
                    error_code="agent_run_deadline_exceeded",
                    error_message="V2 Agent 运行超过总时限",
                )
            except Exception:
                logger.exception("failed to mark deadline run failed: {}", run_id)
            yield _sse(
                {
                    "type": "error",
                    "run_id": run_id,
                    "message": "请求处理超时，请缩小查询范围后重试。",
                }
            )
        except Exception as e:
            logger.exception("agent_query stream error")
            try:
                await AgentRunService.mark_failed(run_id, error_message=str(e))
            except Exception:
                logger.exception("failed to update agent run status: {}", run_id)
            yield _sse({"type": "error", "run_id": run_id, "message": str(e)})
        finally:
            if conversation_pk is not None and lock_token is not None:
                await ConversationLockService.release(conversation_pk, lock_token)
    response = StreamingResponse(process_stream(), media_type="text/event-stream")
    # 对外仍返回业务会话 ID；内部 thread_id 只用于 checkpoint 隔离。
    response.headers["X-Conversation-ID"] = str(conversation_key)
    response.headers["X-Agent-Run-ID"] = run_id
    response.headers["Cache-Control"] = "no-cache"
    return response
