from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

# LangSmith / LangChain 从 os.environ 读追踪开关；pydantic Settings 不会把
# 未声明字段写入环境变量，因此这里显式 load，保证 LANGSMITH_* 生效。
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(_PROJECT_ROOT / ".env", override=False)

from fastapi import FastAPI, Response, status

from agents.checkpoint import close_checkpoint, init_checkpoint
from agents.orchestrator.agent_registry import list_agent_specs
from agents.orchestrator.graph import reset_orchestrator_graph_cache
from app.services.memory.memory_store import close_memory_store, init_memory_store
from app.services.memory.memory_catalog import validate_memory_configuration
from app.api import api_router
from app.core.config import settings
from app.core.logger import get_logger
from app.core.middleware import LoggingMiddleware  # 需从 AssistGen 迁 middleware.py
from app.core.redis_client import close_redis, init_redis, redis_health
from fastapi.middleware.cors import CORSMiddleware

logger = get_logger(service="main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("fin-agent-platform 启动中")
    logger.info(f"环境: {settings.APP_ENV}")
    reset_orchestrator_graph_cache()
    validate_memory_configuration(list_agent_specs())
    try:
        await init_checkpoint()
        await init_memory_store()
        await init_redis()
        yield
    finally:
        await close_redis()
        await close_checkpoint()
        await close_memory_store()
        reset_orchestrator_graph_cache()
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
    allow_origins=[
        "http://localhost:4173",
        "http://127.0.0.1:4173",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")
logger.info("已挂载 api_router，前缀 /api")


@app.get("/health")
async def health(response: Response):
    redis_status = await redis_health()
    degraded = redis_status["status"] == "degraded"
    if degraded and settings.REDIS_REQUIRED:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": "degraded" if degraded else "ok",
        "service": "fin-agent-platform",
        "components": {
            "redis": redis_status,
        },
    }
