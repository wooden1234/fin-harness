from fastapi import APIRouter

from app.api.auth import router as auth_router
from app.api.conversations import router as conversations_router
from app.api.agent import router as agent_router
from app.api.rag import router as rag_router
from app.api.memories import router as memories_router
from app.api.admin_memory import router as admin_memory_router

api_router = APIRouter()

api_router.include_router(auth_router, tags=["authentication"])
api_router.include_router(conversations_router)  # prefix 已在 conversations.py 里写了
api_router.include_router(agent_router)
api_router.include_router(rag_router)
api_router.include_router(memories_router)
api_router.include_router(admin_memory_router)
