from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.security import get_current_user
from app.models.user import User
from app.services.conversation_service import ConversationService

router = APIRouter(prefix="/conversations", tags=["conversations"])


class UpdateConversationNameRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)


@router.post("")
async def create_conversation(current_user: User = Depends(get_current_user)):
    """创建新会话，返回 conversation_id（后续 Agent 多轮会用到）"""
    try:
        conversation_id = await ConversationService.create_conversation(current_user.id)
        return {"conversation_id": conversation_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("")
async def list_conversations(current_user: User = Depends(get_current_user)):
    """当前登录用户的会话列表"""
    try:
        return await ConversationService.get_user_conversations(current_user.id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{conversation_id}/messages")
async def get_conversation_messages(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
):
    """某会话的消息历史（会校验会话是否属于当前用户）"""
    try:
        return await ConversationService.get_conversation_messages(
            conversation_id, current_user.id
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
):
    """删除当前用户的会话。"""
    try:
        await ConversationService.delete_conversation(conversation_id, current_user.id)
        return {"message": "会话已删除"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{conversation_id}/name")
async def update_conversation_name(
    conversation_id: int,
    request: UpdateConversationNameRequest,
    current_user: User = Depends(get_current_user),
):
    """修改会话标题"""
    try:
        await ConversationService.update_conversation_name(
            conversation_id,
            current_user.id,
            request.name,
        )
        return {"message": "会话名称已更新"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
