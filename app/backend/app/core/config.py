from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = BACKEND_DIR.parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env.example"


class Settings(BaseSettings):
    """W1 地基：所有配置从 .env / 环境变量读取，不在代码里写死。"""

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE if ENV_FILE.exists() else DEFAULT_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    APP_ENV: str
    DATABASE_URL: str
    PGVECTOR_DATABASE_URL: str = ""
    PGVECTOR_TABLE_NAME: str = ""
    PGVECTOR_COLLECTION_FAQ: str = "faq_md_vectors"
    PGVECTOR_COLLECTION_MACRO: str = "fin_macro_vectors"
    PGVECTOR_COLLECTION_ANNUAL_REPORT: str = "fin_annual_report_vectors"
    PGVECTOR_COLLECTION_RESEARCH_REPORT: str = "fin_research_report_vectors"
    PGVECTOR_COLLECTION_INDUSTRY_WHITEPAPER: str = "fin_industry_whitepaper_vectors"
    PGVECTOR_COLLECTION_POLICY: str = "fin_policy_vectors"
    EMBEDDING_DIM: int = 1536
    LANGGRAPH_CHECKPOINT_URL: str = ""
    REDIS_URL: str
    SECRET_KEY: str
    ALGORITHM: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int

    # LLM（W3+ Supervisor / Agent）
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"
    DEEPSEEK_MODEL: str = "deepseek-v4-flash"
    AGENT_ROUTER_TEMPERATURE: float = 0.0
    AGENT_FAQ_TEMPERATURE: float = 0.3
    FAQ_MIN_RELEVANCE_SCORE: float = 0.35
    PDF_MIN_RELEVANCE_SCORE: float = 0.35

    # Embedding（DashScope 兼容 OpenAI API）
    QWEN_API_KEY: str = ""
    DASHSCOPE_API_KEY: str = ""
    EMBEDDING_MODEL: str = "text-embedding-v2"
    QWEN_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # 联网搜索。默认关闭，配置 TAVILY_API_KEY 后启用 Tavily。
    WEB_SEARCH_PROVIDER: str = "tavily"
    TAVILY_API_KEY: str = ""
    TAVILY_SEARCH_URL: str = "https://api.tavily.com/search"
    WEB_SEARCH_MAX_RESULTS: int = 5

    # LangGraph Checkpoint（W3 Day 5）：postgres | memory
    AGENT_CHECKPOINT_BACKEND: str = "postgres"

    # MinerU PDF 解析
    MINERU_API_KEY: str = ""
    MINERU_TOKEN: str = ""
    MINERU_MAX_PAGES: int = 200
    MINERU_BASE_URL: str = "https://mineru.net/api/v4"
    MINERU_MODEL_VERSION: str = "vlm"

    # text_to_sql：规则通过后是否启用 LLM 结果质检
    FINANCIAL_SQL_LLM_VALIDATION: bool = False
    # text_to_sql：连续相同 validation_error_type 多少次后提前放弃纠错
    FINANCIAL_SQL_MAX_REPEAT_SAME_ERROR: int = 2


settings = Settings()
