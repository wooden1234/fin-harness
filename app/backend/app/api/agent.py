import json
import uuid
from typing import Optional
from fastapi import APIRouter, Depends, Form, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from agents.checkpoint import make_thread_config
from agents.graph import get_graph
from app.api.agent_progress import (
    VISIBLE_TASK_NODES,
    build_public_step_event,
    map_node_to_public_step,
)
from app.core.logger import get_logger
from app.core.security import get_current_user
from app.models.user import User
from app.services.conversation_service import ConversationService


router = APIRouter(prefix="/agent", tags=["agent"])
logger = get_logger(service="agent")

# 候选答案在 general_agent / summarize 生成时尚未通过合规审查，禁止提前发送。
STREAMABLE_ANSWER_NODES = frozenset({"final_answer"})

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
async def agent_health(current_user: User = Depends(get_current_user)):
    """W3 前占位：验证 Agent 路由走 JWT"""
    return {"status": "agent module ready", "user_id": current_user.id}

@router.post("/query")
async def agent_query(
    query: str = Form(...),
    conversation_id: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user)):
    conversation_pk: int | None = None
    if conversation_id is not None:
        try:
            conversation_pk = int(conversation_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="conversation_id 格式错误") from exc

        conversation = await ConversationService.get_owned_conversation(
            conversation_pk,
            current_user.id,
        )
        if conversation is None:
            raise HTTPException(status_code=404, detail="会话不存在或无权访问")

    conversation_key = conversation_pk if conversation_pk is not None else uuid.uuid4()
    thread_config = make_thread_config(
        conversation_key,
        user_id=current_user.id,
    )
    graph = get_graph()
    input_payload = {"messages": [HumanMessage(content=query)]}
    # 只发送 final_answer 审查后的内容，避免候选答案先于合规结果泄露。
    async def process_stream():
        assistant_full_response = ""
        generating_answer_active = False
        try:
            async for chunk in graph.astream(
                input_payload,
                config=thread_config,
                stream_mode=["messages", "tasks", "updates"],
                subgraphs=True,
            ):
                if not isinstance(chunk, tuple) or len(chunk) != 3:
                    continue
                namespace, mode, data = chunk
                ns_tuple = namespace if isinstance(namespace, tuple) else ()

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
                        if node_name == "risk_triage":
                            result_payload = data.get("result")
                            if isinstance(result_payload, dict):
                                risk_level = result_payload.get("risk_level")
                                if risk_level:
                                    yield _sse({"type": "meta", "risk_level": risk_level})

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
            citations = values.get("citations") or []
            final_response = _extract_final_response(values)
            if not assistant_full_response and final_response:
                assistant_full_response = final_response
            if generating_answer_active:
                event = build_public_step_event("generating_answer", "done")
                if event:
                    yield _sse(event)
                generating_answer_active = False
            yield _sse({
                "type": "done",
                "content": final_response,
                "citations": citations,
                "route": values.get("route"),
                "risk_level": values.get("risk_level"),
                "compliance_action": values.get("compliance_action"),
                "compliance_reason_code": values.get("compliance_reason_code"),
            })

            # 持久化消息到数据库
            if conversation_pk is not None and assistant_full_response:
                try:
                    await ConversationService.save_message(
                        user_id=current_user.id,
                        conversation_id=conversation_pk,
                        messages=[{"role": "user", "content": query}],
                        response=assistant_full_response,
                    )
                    logger.info(
                        "conversation saved: user={}, conv={}",
                        current_user.id, conversation_id,
                    )
                except Exception as save_err:
                    logger.error("Failed to save conversation: {}", save_err)

        except Exception as e:
            logger.exception("agent_query stream error")
            yield _sse({"type": "error", "message": str(e)})
    response = StreamingResponse(process_stream(), media_type="text/event-stream")
    # 对外仍返回业务会话 ID；内部 thread_id 只用于 checkpoint 隔离。
    response.headers["X-Conversation-ID"] = str(conversation_key)
    response.headers["Cache-Control"] = "no-cache"
    return response

