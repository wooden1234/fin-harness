"""长期记忆管理 API。"""

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.security import get_current_user
from app.models.identity.user import User
from app.schemas.memory import (
    EpisodicMemoryCreate,
    MemoryCreate,
    MemoryResponse,
    MemoryUpdate,
)
from app.services.memory.memory_service import MemoryService
from app.services.memory.memory_metrics import snapshot

router = APIRouter(prefix="/memories", tags=["memories"])


@router.get("/metrics")
async def memory_metrics(current_user: User = Depends(get_current_user)):
    """返回当前进程的记忆召回质量与成本指标。"""
    _ = current_user
    return snapshot()


@router.get("", response_model=list[MemoryResponse])
async def list_memories(current_user: User = Depends(get_current_user)):
    records = await MemoryService.list(
        tenant_id=current_user.tenant_id,
        user_id=current_user.id,
    )
    return [MemoryResponse.from_record(record) for record in records]


@router.get("/candidates", response_model=list[MemoryResponse])
async def list_memory_candidates(current_user: User = Depends(get_current_user)):
    records = await MemoryService.list_candidates(
        tenant_id=current_user.tenant_id,
        user_id=current_user.id,
    )
    return [MemoryResponse.from_record(record) for record in records]


@router.get("/episodic", response_model=list[MemoryResponse])
async def list_episodic_memories(current_user: User = Depends(get_current_user)):
    records = await MemoryService.list_episodic(
        tenant_id=current_user.tenant_id,
        user_id=current_user.id,
    )
    return [MemoryResponse.from_record(record) for record in records]


@router.post("/episodic", response_model=MemoryResponse, status_code=status.HTTP_201_CREATED)
async def create_episodic_memory(
    payload: EpisodicMemoryCreate,
    current_user: User = Depends(get_current_user),
):
    record = await MemoryService.create_episodic(
        tenant_id=current_user.tenant_id,
        user_id=current_user.id,
        event_key=payload.event_key,
        value=payload.value,
        display_text=payload.display_text,
        expires_at=payload.expires_at,
        source_conversation_id=payload.source_conversation_id,
        source_message_id=payload.source_message_id,
        source_run_id=payload.source_run_id,
        actor_id=str(current_user.id),
    )
    return MemoryResponse.from_record(record)


@router.post("/{memory_id}/confirm", response_model=MemoryResponse)
async def confirm_memory_candidate(
    memory_id: str,
    current_user: User = Depends(get_current_user),
):
    record = await MemoryService.decide_candidate(
        tenant_id=current_user.tenant_id,
        user_id=current_user.id,
        memory_id=memory_id,
        decision="confirm",
        actor_id=str(current_user.id),
    )
    if record is None:
        raise HTTPException(status_code=404, detail="候选记忆不存在或无权访问")
    return MemoryResponse.from_record(record)


@router.post("/{memory_id}/reject", response_model=MemoryResponse)
async def reject_memory_candidate(
    memory_id: str,
    current_user: User = Depends(get_current_user),
):
    record = await MemoryService.decide_candidate(
        tenant_id=current_user.tenant_id,
        user_id=current_user.id,
        memory_id=memory_id,
        decision="reject",
        actor_id=str(current_user.id),
    )
    if record is None:
        raise HTTPException(status_code=404, detail="候选记忆不存在或无权访问")
    return MemoryResponse.from_record(record)


@router.post("", response_model=MemoryResponse, status_code=status.HTTP_201_CREATED)
async def create_memory(
    payload: MemoryCreate,
    current_user: User = Depends(get_current_user),
):
    try:
        record = await MemoryService.create(
            tenant_id=current_user.tenant_id,
            user_id=current_user.id,
            memory_key=payload.memory_key or "",
            value=payload.value,
            display_text=payload.display_text,
            ttl_days=payload.ttl_days,
            provenance=payload.provenance,
            actor_id=str(current_user.id),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return MemoryResponse.from_record(record)


@router.patch("/{memory_id}", response_model=MemoryResponse)
async def update_memory(
    memory_id: str,
    payload: MemoryUpdate,
    current_user: User = Depends(get_current_user),
):
    try:
        record = await MemoryService.update(
            tenant_id=current_user.tenant_id,
            user_id=current_user.id,
            memory_id=memory_id,
            value=payload.value,
            display_text=payload.display_text,
            ttl_days=payload.ttl_days,
            expected_version=payload.expected_version,
            actor_id=str(current_user.id),
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if record is None:
        raise HTTPException(status_code=404, detail="记忆不存在或无权访问")
    return MemoryResponse.from_record(record)


@router.delete("/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_memory(
    memory_id: str,
    current_user: User = Depends(get_current_user),
):
    deleted = await MemoryService.revoke(
        tenant_id=current_user.tenant_id,
        user_id=current_user.id,
        memory_id=memory_id,
        actor_id=str(current_user.id),
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="记忆不存在或无权访问")
