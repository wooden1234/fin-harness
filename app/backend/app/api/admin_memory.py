"""管理员长期记忆合规工具。"""

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.core.security import require_compliance_user
from app.models.identity.user import User
from app.schemas.memory import MemoryResponse
from app.services.memory.memory_compliance import (
    MemoryComplianceService,
    scan_memory,
)

router = APIRouter(prefix="/admin/memories", tags=["admin-memory"])


class AdminRevokeRequest(BaseModel):
    reason: str = Field(..., min_length=1, max_length=500)


@router.get("", response_model=list[MemoryResponse])
async def admin_list_memories(
    tenant_id: str | None = None,
    user_id: int | None = None,
    limit: int = Query(default=100, ge=1, le=500),
    current_user: User = Depends(require_compliance_user),
):
    if current_user.role != "platform_admin":
        tenant_id = current_user.tenant_id
    records = await MemoryComplianceService.list_records(
        tenant_id=tenant_id,
        user_id=user_id,
        limit=limit,
    )
    return [MemoryResponse.from_record(record) for record in records]


@router.get("/{memory_id}/scan")
async def admin_scan_memory(
    memory_id: str,
    current_user: User = Depends(require_compliance_user),
):
    tenant_id = current_user.tenant_id if current_user.role != "platform_admin" else None
    records = await MemoryComplianceService.list_records(tenant_id=tenant_id, limit=500)
    record = next((item for item in records if item.id == memory_id), None)
    if record is None:
        raise HTTPException(status_code=404, detail="记忆不存在")
    return scan_memory(record)


@router.post("/{memory_id}/revoke", response_model=MemoryResponse)
async def admin_revoke_memory(
    memory_id: str,
    payload: AdminRevokeRequest,
    current_user: User = Depends(require_compliance_user),
):
    tenant_id = current_user.tenant_id
    if current_user.role == "platform_admin":
        records = await MemoryComplianceService.list_records(limit=500)
        target = next((item for item in records if item.id == memory_id), None)
        if target is None:
            raise HTTPException(status_code=404, detail="记忆不存在")
        tenant_id = target.tenant_id
    record = await MemoryComplianceService.revoke(
        memory_id=memory_id,
        tenant_id=tenant_id,
        actor_id=current_user.id,
        reason=payload.reason,
    )
    if record is None:
        raise HTTPException(status_code=404, detail="记忆不存在或无权操作")
    return MemoryResponse.from_record(record)
