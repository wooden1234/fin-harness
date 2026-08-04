"""聊天附件上传 API。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, UploadFile

from app.core.security import get_current_user
from app.schemas.user import AuthUser
from app.services.attachments.attachment_service import upload_image_attachment

router = APIRouter(prefix="/attachments", tags=["attachments"])


@router.post("")
async def create_attachment(
    file: UploadFile = File(...),
    current_user: AuthUser = Depends(get_current_user),
) -> dict:
    meta = await upload_image_attachment(
        file,
        user_id=current_user.id,
        tenant_id=current_user.tenant_id,
    )
    return {
        "attachment_id": meta.attachment_id,
        "object_key": meta.object_key,
        "content_type": meta.content_type,
        "size": meta.size,
    }
