"""长期记忆管理 API。"""

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.security import get_current_user
from app.models.identity.user import User
from app.schemas.memory import MemoryCreate, MemoryResponse, MemoryUpdate
from app.services.memory.memory_service import MemoryService

router = APIRouter(prefix="/memories", tags=["memories"])


@router.get("", response_model=list[MemoryResponse])
async def list_memories(current_user: User = Depends(get_current_user)):
    records = await MemoryService.list(
        tenant_id=current_user.tenant_id,
        user_id=current_user.id,
    )
    return [MemoryResponse.from_record(record) for record in records]


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
