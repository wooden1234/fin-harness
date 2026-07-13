from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.agents.checkpoint import close_checkpoint, init_checkpoint
from app.api import api_router
from app.core.config import settings
from app.core.logger import get_logger
from app.core.middleware import LoggingMiddleware  # 需从 AssistGen 迁 middleware.py
from fastapi.middleware.cors import CORSMiddleware

logger = get_logger(service="main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动
    logger.info("fin-agent-platform 启动中")
    logger.info(f"环境: {settings.APP_ENV}")
    await init_checkpoint()
    yield
    await close_checkpoint()
    logger.info("fin-agent-platform 正在关闭")


app = FastAPI(
    title="fin-agent-platform",
    version="0.1.0",
    description="金融 Multi-Agent 智能客服（W1 地基）",
    lifespan=lifespan,
)

# 每个 HTTP 请求打一行（类似 nginx access log）
app.add_middleware(LoggingMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")
logger.info("已挂载 api_router，前缀 /api")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "fin-agent-platform"}