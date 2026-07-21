from typing import List, Dict
from app.core.database import AsyncSessionLocal
from app.models.conversation import Conversation, DialogueType
from app.models.message import Message
from app.models.outbox_event import OutboxEvent
from app.models.audit_log import AuditLog
from app.core.logger import get_logger
from sqlalchemy import func, select

logger = get_logger(service="conversation")

class ConversationService:
    @staticmethod
    async def save_user_message(
        *,
        user_id: int,
        conversation_id: int,
        content: str,
        run_id: str,
        client_message_id: str,
    ) -> Message:
        """在 Agent 开始执行前写入用户消息，确保请求事实先落档。"""
        async with AsyncSessionLocal() as db:
            conversation = await db.scalar(
                select(Conversation).where(
                    Conversation.id == conversation_id,
                    Conversation.user_id == user_id,
                )
            )
            if conversation is None:
                raise ValueError("会话不存在或无权访问")

            duplicate = await db.scalar(
                select(Message).where(
                    Message.conversation_id == conversation_id,
                    Message.client_message_id == client_message_id,
                )
            )
            if duplicate is not None:
                return duplicate

            max_sequence = await db.scalar(
                select(func.max(Message.sequence_no)).where(
                    Message.conversation_id == conversation_id
                )
            )
            if not max_sequence:
                conversation.title = ConversationService.get_conversation_title(content)

            message = Message(
                conversation_id=conversation_id,
                sender="user",
                content=content,
                run_id=run_id,
                sequence_no=(max_sequence or 0) + 1,
                client_message_id=client_message_id,
            )
            db.add(message)
            await db.commit()
            await db.refresh(message)
            return message

    @staticmethod
    async def save_assistant_message(
        *, user_id: int, conversation_id: int, content: str, run_id: str
    ) -> Message:
        """保存已完成运行的助手消息，并与 agent_run 关联。"""
        async with AsyncSessionLocal() as db:
            conversation = await db.scalar(
                select(Conversation).where(
                    Conversation.id == conversation_id,
                    Conversation.user_id == user_id,
                )
            )
            if conversation is None:
                raise ValueError("会话不存在或无权访问")
            # run_id 是助手消息的幂等键，worker 重试时直接复用已有记录。
            existing = await db.scalar(
                select(Message).where(
                    Message.conversation_id == conversation_id,
                    Message.run_id == run_id,
                    Message.sender == "assistant",
                )
            )
            if existing is not None:
                return existing
            max_sequence = await db.scalar(
                select(func.max(Message.sequence_no)).where(
                    Message.conversation_id == conversation_id
                )
            )
            message = Message(
                conversation_id=conversation_id,
                sender="assistant",
                content=content,
                run_id=run_id,
                sequence_no=(max_sequence or 0) + 1,
            )
            db.add(message)
            await db.commit()
            await db.refresh(message)
            return message

    @staticmethod
    async def get_owned_conversation(
        conversation_id: int,
        user_id: int,
    ) -> Conversation | None:
        """按会话 ID 和用户 ID 查询，统一执行归属校验。"""
        async with AsyncSessionLocal() as db:
            stmt = select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
                Conversation.deleted_at.is_(None),
            )
            result = await db.execute(stmt)
            return result.scalar_one_or_none()

    @staticmethod
    def get_conversation_title(message: str, max_length: int = 20) -> str:
        """从消息中提取第一句作为会话标题"""
        # 按常见断句符号取第一句
        for sep in ("。", "？", "！", "\n", "；"):
            idx = message.find(sep)
            if idx >= 0:
                title = message[:idx].strip()
                break
        else:
            title = message.strip()
        # 压缩多余空格
        title = " ".join(title.split())
        if len(title) > max_length:
            title = title[:max_length] + "..."
        return title if title else "新会话"

    @staticmethod
    async def create_conversation(user_id: int) -> int:
        """创建新会话"""
        async with AsyncSessionLocal() as db:
            conversation = Conversation(
                user_id=user_id,
                title="新会话",
                dialogue_type=DialogueType.NORMAL
            )
            db.add(conversation)
            await db.commit()
            await db.refresh(conversation)
            
            logger.info(f"Created new conversation {conversation.id} for user {user_id}")
            return conversation.id

    @staticmethod
    async def save_message(
        user_id: int, 
        conversation_id: int, 
        messages: List[Dict], 
        response: str
    ):
        """保存对话消息"""
        try:
            async with AsyncSessionLocal() as db:
                # 查询会话
                stmt = select(Conversation).where(
                    Conversation.id == conversation_id,
                    Conversation.user_id == user_id,
                    Conversation.deleted_at.is_(None),
                )
                result = await db.execute(stmt)
                conversation = result.scalar_one_or_none()

                if not conversation:
                    raise ValueError("会话不存在或无权访问")

                # 查询现有消息数量
                stmt = select(Message).where(Message.conversation_id == conversation_id)
                result = await db.execute(stmt)
                messages_count = len(result.all())
                
                # 获取用户的问题内容
                user_content = next((msg["content"] for msg in messages if msg["role"] == "user"), "")
                
                # 如果是第一条消息，更新会话标题
                if messages_count == 0:
                    title = ConversationService.get_conversation_title(user_content)
                    conversation.title = title
                
                # 保存用户消息
                user_message = Message(
                    conversation_id=conversation_id,
                    sender="user",
                    content=user_content
                )
                db.add(user_message)
                
                # 保存助手回复
                assistant_message = Message(
                    conversation_id=conversation_id,
                    sender="assistant",
                    content=response
                )
                db.add(assistant_message)
                
                await db.commit()
                
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error saving conversation: {str(e)}", exc_info=True)
            logger.error(f"Error details - user_id: {user_id}, conversation_id: {conversation_id}")
            logger.error(f"Messages: {messages}")
            # 让上层把业务消息落库失败记录为可重试状态。
            raise

    @staticmethod
    async def get_user_conversations(user_id: int) -> List[Dict]:
        """获取用户的所有会话"""
        try:
            async with AsyncSessionLocal() as db:
                # 查询用户的所有会话，排除未使用的新会话
                stmt = select(Conversation).where(
                    Conversation.user_id == user_id,
                    Conversation.title != "新会话",
                    Conversation.deleted_at.is_(None),
                ).order_by(Conversation.created_at.desc())
                
                result = await db.execute(stmt)
                conversations = result.scalars().all()
                
                return [
                    {
                        "id": conv.id,
                        "title": conv.title,
                        "created_at": conv.created_at.isoformat(),
                        "status": conv.status,
                        "dialogue_type": conv.dialogue_type.value
                    }
                    for conv in conversations
                ]
                
        except Exception as e:
            logger.error(f"Error getting conversations for user {user_id}: {str(e)}", exc_info=True)
            raise

    @staticmethod
    async def get_conversation_messages(conversation_id: int, user_id: int) -> List[Dict]:
        """获取会话的所有消息"""
        try:
            async with AsyncSessionLocal() as db:
                # 首先验证会话属于该用户
                stmt = select(Conversation).where(
                    Conversation.id == conversation_id,
                    Conversation.user_id == user_id
                )
                result = await db.execute(stmt)
                conversation = result.scalar_one_or_none()
                
                if not conversation:
                    raise ValueError(f"Conversation {conversation_id} not found or not owned by user {user_id}")
                
                # 查询会话的所有消息
                stmt = select(Message).where(
                    Message.conversation_id == conversation_id
                ).order_by(Message.created_at)
                
                result = await db.execute(stmt)
                messages = result.scalars().all()
                
                return [
                    {
                        "id": msg.id,
                        "sender": msg.sender,
                        "content": msg.content,
                        "created_at": msg.created_at.isoformat(),
                        "message_type": msg.message_type,
                        "run_id": msg.run_id,
                        "sequence_no": msg.sequence_no,
                        "client_message_id": msg.client_message_id,
                    }
                    for msg in messages
                ]
                
        except Exception as e:
            logger.error(f"Error getting messages for conversation {conversation_id}: {str(e)}", exc_info=True)
            raise

    @staticmethod
    async def delete_conversation(conversation_id: int, user_id: int):
        """软删除会话，并通过 outbox 异步清理 checkpoint。"""
        try:
            async with AsyncSessionLocal() as db:
                stmt = select(Conversation).where(
                    Conversation.id == conversation_id,
                    Conversation.user_id == user_id,
                    Conversation.deleted_at.is_(None),
                )
                result = await db.execute(stmt)
                conversation = result.scalar_one_or_none()

                if not conversation:
                    raise ValueError("会话不存在或无权访问")

                conversation.deleted_at = func.now()
                conversation.deleted_by = user_id
                conversation.status = "deleted"
                db.add(
                    OutboxEvent(
                        event_key=f"conversation_checkpoint_delete:{conversation_id}",
                        event_type="conversation.checkpoint_delete",
                        aggregate_id=str(conversation_id),
                        payload={"conversation_id": conversation_id, "user_id": user_id},
                        status="pending",
                    )
                )
                db.add(
                    AuditLog(
                        user_id=user_id,
                        action="conversation.soft_delete",
                        resource_type="conversation",
                        resource_id=str(conversation_id),
                    )
                )
                await db.commit()

                logger.info(f"已软删除会话 {conversation_id}，checkpoint 进入 outbox")
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"删除会话失败: {str(e)}", exc_info=True)
            raise

    @staticmethod
    async def update_conversation_name(
        conversation_id: int,
        user_id: int,
        name: str,
    ):
        """更新会话名称"""
        try:
            async with AsyncSessionLocal() as db:
                stmt = select(Conversation).where(
                    Conversation.id == conversation_id,
                    Conversation.user_id == user_id,
                )
                result = await db.execute(stmt)
                conversation = result.scalar_one_or_none()

                if not conversation:
                    raise ValueError("会话不存在或无权访问")
                
                # 更新名称
                conversation.title = name
                await db.commit()
                
                logger.info(f"已更新会话 {conversation_id} 的名称为 {name}")
        except Exception as e:
            logger.error(f"更新会话名称失败: {str(e)}", exc_info=True)
            raise
