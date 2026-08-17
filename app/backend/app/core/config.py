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
    FINANCIAL_QUERY_DATABASE_URL: str = ""
    FINANCIAL_QUERY_STATEMENT_TIMEOUT_MS: int = 5000
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
    REDIS_URL: str = ""
    REDIS_ENABLED: bool = True
    REDIS_REQUIRED: bool = False
    REDIS_KEY_PREFIX: str = "fin"
    REDIS_MAX_CONNECTIONS: int = 50
    REDIS_CONNECT_TIMEOUT_SEC: float = 1.0
    REDIS_SOCKET_TIMEOUT_SEC: float = 1.0
    REDIS_HEALTH_CHECK_INTERVAL_SEC: int = 30
    REDIS_RETRY_ATTEMPTS: int = 2
    REDIS_RETRY_BASE_DELAY_MS: int = 50
    MEMORY_CACHE_ENABLED: bool = True
    MEMORY_CACHE_TTL_SEC: int = 600
    MEMORY_CACHE_NEGATIVE_TTL_SEC: int = 45
    MEMORY_CACHE_TTL_JITTER_RATIO: float = 0.1
    MEMORY_CACHE_LOCK_TTL_SEC: int = 5
    MEMORY_CACHE_LOCK_WAIT_MS: int = 30
    # 共享缓存 TTL jitter；各 domain 最终 TTL 至少 1 秒
    CACHE_TTL_JITTER_RATIO: float = 0.1
    # JWT 鉴权用户 Cache-Aside（默认关闭）
    AUTH_USER_CACHE_ENABLED: bool = False
    AUTH_USER_CACHE_TTL_SEC: int = 60
    AUTH_USER_CACHE_NEGATIVE_TTL_SEC: int = 30
    # Query Embedding Cache-Aside（默认关闭）
    EMBEDDING_CACHE_ENABLED: bool = False
    EMBEDDING_CACHE_TTL_SEC: int = 43200
    EMBEDDING_CACHE_LOCK_TTL_SEC: int = 75
    EMBEDDING_CACHE_LOCK_WAIT_MS: int = 100
    # FAQ/PDF 精确检索证据包缓存；版本需在知识库重建后递增
    RETRIEVAL_EVIDENCE_CACHE_ENABLED: bool = False
    RETRIEVAL_EVIDENCE_CACHE_VERSION: str = "v1"
    RETRIEVAL_EVIDENCE_FAQ_TTL_SEC: int = 1800
    RETRIEVAL_EVIDENCE_PDF_TTL_SEC: int = 600
    RETRIEVAL_EVIDENCE_CACHE_MAX_BYTES: int = 262144
    # iwencai 只读结果 Cache-Aside（默认关闭）
    IWENCAI_CACHE_ENABLED: bool = False
    IWENCAI_CACHE_MARKET_TTL_SEC: int = 30
    IWENCAI_CACHE_DOCUMENT_TTL_SEC: int = 600
    IWENCAI_CACHE_SCREEN_TTL_SEC: int = 60
    IWENCAI_CACHE_MAX_BYTES: int = 524288
    # 天气当前结果 Cache-Aside；Redis 不可用时自动回源 OpenWeather
    WEATHER_CACHE_ENABLED: bool = True
    WEATHER_CACHE_TTL_SEC: int = 600
    WEATHER_CACHE_MAX_BYTES: int = 262144
    SECRET_KEY: str
    ALGORITHM: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int

    # Elasticsearch（BM25 / 全文检索）
    ELASTICSEARCH_URL: str = ""
    ELASTICSEARCH_INDEX_PREFIX: str = "fin_agent"
    ELASTICSEARCH_INDEX_FAQ: str = "fin_agent_faq"
    ELASTICSEARCH_INDEX_MACRO: str = "fin_agent_macro_research"
    ELASTICSEARCH_INDEX_ANNUAL_REPORT: str = "fin_agent_annual_reports"
    ELASTICSEARCH_INDEX_RESEARCH_REPORT: str = "fin_agent_research_reports"
    ELASTICSEARCH_INDEX_INDUSTRY_WHITEPAPER: str = "fin_agent_industry_whitepapers"
    ELASTICSEARCH_INDEX_POLICY: str = "fin_agent_policy"
    ELASTICSEARCH_USERNAME: str = ""
    ELASTICSEARCH_PASSWORD: str = ""
    ELASTICSEARCH_ENABLED: bool = False
    ES_BM25F_TEXT_WEIGHT: float = 1.5
    ES_BM25F_LEAF_TEXT_WEIGHT: float = 1.0
    ES_BM25F_TITLE_WEIGHT: float = 5.0
    ES_BM25F_SECTION_WEIGHT: float = 4.0
    ES_BM25F_SOURCE_WEIGHT: float = 2.0
    ES_BM25_MODE: str = "combined_fields"

    # Milvus（向量检索）
    MILVUS_ENABLED: bool = False
    MILVUS_URI: str = "http://localhost:19530"
    MILVUS_TOKEN: str = ""
    MILVUS_COLLECTION_PREFIX: str = "fin_agent"
    MILVUS_COLLECTION_FAQ: str = "fin_agent_faq"
    MILVUS_COLLECTION_MACRO: str = "fin_agent_macro_research"
    MILVUS_COLLECTION_ANNUAL_REPORT: str = "fin_agent_annual_reports"
    MILVUS_COLLECTION_RESEARCH_REPORT: str = "fin_agent_research_reports"
    MILVUS_COLLECTION_INDUSTRY_WHITEPAPER: str = "fin_agent_industry_whitepapers"
    MILVUS_COLLECTION_POLICY: str = "fin_agent_policy"
    MILVUS_DIM: int = 1536
    MILVUS_METRIC_TYPE: str = "COSINE"
    MILVUS_INDEX_TYPE: str = "HNSW"
    MILVUS_M: int = 16
    MILVUS_EF_CONSTRUCTION: int = 200
    MILVUS_SEARCH_EF: int = 64
    VECTOR_CANDIDATE_MULTIPLIER: int = 8
    VECTOR_DIVERSITY_ENABLED: bool = True
    VECTOR_DIVERSITY_TARGET_DUPLICATE_RATE: float = 0.70
    VECTOR_DIVERSITY_STRENGTH: float = 0.10
    VECTOR_DIVERSITY_MAX_PENALTY: float = 0.20
    VECTOR_DIVERSITY_MIN_SCORE_RATIO: float = 0.85
    AUTO_MERGE_MIN_CHILDREN: int = 2

    # LLM（W3+ Supervisor / Agent）
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"
    DEEPSEEK_MODEL: str = "deepseek-v4-flash"
    # DeepSeek Thinking 模式当前不支持结构化工具选择，工具 Agent 默认关闭。
    DEEPSEEK_THINKING_ENABLED: bool = False
    AGENT_ROUTER_TEMPERATURE: float = 0.0
    PDF_QUERY_FILTER_MIN_CONFIDENCE: float = 0.85
    PDF_KB_UNSUPPORTED_MIN_CONFIDENCE: float = 0.90
    AGENT_FAQ_TEMPERATURE: float = 0.3
    # DeepAgent 主推理 / 结构化成稿：本地金融微调（OpenAI 兼容，如 vLLM）
    FINANCE_LLM_API_KEY: str = "EMPTY"
    FINANCE_LLM_BASE_URL: str = ""
    FINANCE_LLM_MODEL: str = ""
    FINANCE_LLM_TEMPERATURE: float = 0.0
    FINANCE_LLM_TIMEOUT_SEC: float = 120.0
    FINANCE_LLM_ENABLE_THINKING: bool = False
    # vLLM 探活超时；不可达时 get_finance_llm 回退 DeepSeek
    FINANCE_LLM_PROBE_TIMEOUT_SEC: float = 2.0
    FINANCE_LLM_PROBE_TTL_SEC: float = 30.0
    # DeepSeek 跑工具规划；finalign 仅通过 finalign.analyze 成稿
    FINANCE_LLM_DRAFT_ENABLED: bool = True
    # 仅当把 finance LLM 当 Agent 主模型时才需要 tool-call 探活
    FINANCE_LLM_REQUIRE_TOOL_CALLS: bool = False
    FAQ_MIN_RELEVANCE_SCORE: float = 0.35
    PDF_MIN_RELEVANCE_SCORE: float = 0.35
    PDF_RETRIEVAL_TOP_K: int = 10
    PDF_RETRIEVAL_QUALITY_MODEL_PATH: str = str(
        PROJECT_ROOT / "retrieval/models/pdf_retrieval_quality.json"
    )

    # Embedding（OpenAI 兼容：DashScope / 讯飞星辰 MaaS）
    EMBEDDING_PROVIDER: str = "dashscope"
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_MODEL: str = "text-embedding-v2"
    EMBEDDING_BASE_URL: str = ""
    EMBEDDING_BATCH_SIZE: int = 0
    MEMORY_RECALL_TOP_K: int = 8
    MEMORY_RECALL_TOKEN_BUDGET: int = 512
    MEMORY_LLM_EXTRACTION_ENABLED: bool = True
    MEMORY_LLM_EXTRACTION_TIMEOUT_SEC: float = 3.0
    MEMORY_LLM_EXTRACTION_MIN_CONFIDENCE: float = 0.85
    EPISODIC_MEMORY_TTL_DAYS: int = 60
    EPISODIC_MEMORY_TURN_THRESHOLD: int = 10
    EPISODIC_MEMORY_TOKEN_THRESHOLD: int = 8000
    EPISODIC_MEMORY_PROGRESS_MINUTES: int = 20
    EPISODIC_MEMORY_MAX_SOURCE_MESSAGES: int = 30
    EPISODIC_MEMORY_MIN_CONFIDENCE: float = 0.80
    QWEN_API_KEY: str = ""
    DASHSCOPE_API_KEY: str = ""
    QWEN_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # 阿里云 OSS（聊天图片上传；密钥仅后端使用）
    OSS_ACCESS_KEY_ID: str = ""
    OSS_ACCESS_KEY_SECRET: str = ""
    OSS_ENDPOINT: str = ""
    OSS_BUCKET: str = ""
    OSS_PREFIX: str = "chat-images/"
    OSS_PUBLIC_BASE_URL: str = ""
    OSS_SIGN_URL_EXPIRES_SEC: int = 3600
    ATTACHMENT_MAX_BYTES: int = 5_242_880  # 5MB
    ATTACHMENT_TTL_SEC: int = 86_400

    # Vision（看图预处理；与 Embedding 的 QWEN_* 用途分离）
    VISION_MODEL: str = "qwen3.8-max"
    VISION_BASE_URL: str = ""
    VISION_API_KEY: str = ""
    VISION_TIMEOUT_SEC: float = 120.0

    RERANK_ENABLED: bool = False
    RERANK_PROVIDER: str = "dashscope"
    RERANK_API_KEY: str = ""
    RERANK_MODEL: str = "qwen3-rerank"
    RERANK_BASE_URL: str = (
        "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
    )
    RERANK_CANDIDATE_TOP_K: int = 20
    RERANK_MIN_SCORE: float = 0.0
    RERANK_TIMEOUT_SEC: float = 30.0
    RERANK_RETURN_DOCUMENTS: bool = True

    # 联网搜索。默认关闭，配置 TAVILY_API_KEY 后启用 Tavily。
    WEB_SEARCH_PROVIDER: str = "tavily"
    TAVILY_API_KEY: str = ""
    TAVILY_SEARCH_URL: str = "https://api.tavily.com/search"
    WEB_SEARCH_MAX_RESULTS: int = 4
    # 低于阈值的结果一律丢弃（不保底）；中文金融检索建议 0.35 起。
    WEB_SEARCH_MIN_SCORE: float = 0.35
    # A 股常规收盘（上海时区），用于「今日」→最近已结束交易日近似；无节假日日历。
    A_SHARE_SESSION_CLOSE_HOUR: int = 15
    A_SHARE_SESSION_CLOSE_MINUTE: int = 0
    A_SHARE_TZ: str = "Asia/Shanghai"

    # 天气（OpenWeatherMap）。配置 OPENWEATHER_API_KEY 后启用。
    OPENWEATHER_API_KEY: str = ""
    OPENWEATHER_GEOCODE_URL: str = "https://api.openweathermap.org/geo/1.0/direct"
    OPENWEATHER_CURRENT_URL: str = "https://api.openweathermap.org/data/2.5/weather"
    OPENWEATHER_TIMEOUT_SEC: float = 10.0

    # 同花顺问财 SkillHub OpenAPI。密钥只从环境变量读取。
    IWENCAI_BASE_URL: str = "https://openapi.iwencai.com"
    IWENCAI_API_KEY: str = ""
    IWENCAI_TIMEOUT_SEC: float = 30.0
    IWENCAI_MAX_LIMIT: int = 100
    IWENCAI_SKILL_ROOT: str = str(
        PROJECT_ROOT / ".iwencai-skills"
    )
    IWENCAI_SKILL_RUNNER_ENABLED: bool = True
    IWENCAI_SKILL_RUNNER_TIMEOUT_SEC: float = 30.0
    IWENCAI_SKILL_RUNNER_MAX_OUTPUT_BYTES: int = 1_000_000

    # LangGraph Checkpoint（W3 Day 5）：postgres | memory
    AGENT_CHECKPOINT_BACKEND: str = "postgres"

    # V2 请求预算：软时限停止扩张并降级回答，硬时限终止执行。
    AGENT_V2_SIMPLE_SOFT_DEADLINE_SEC: float = 6.0
    AGENT_V2_SIMPLE_HARD_DEADLINE_SEC: float = 12.0
    AGENT_V2_SINGLE_SOFT_DEADLINE_SEC: float = 15.0
    AGENT_V2_SINGLE_HARD_DEADLINE_SEC: float = 25.0
    AGENT_V2_COMPOUND_SOFT_DEADLINE_SEC: float = 25.0
    AGENT_V2_COMPOUND_HARD_DEADLINE_SEC: float = 40.0
    AGENT_V2_FINALIZATION_GRACE_SEC: float = 3.0
    AGENT_V2_INFLIGHT_GRACE_SEC: float = 3.0
    # V2 执行单元预算，实际超时还会受本轮动态硬时限约束。
    AGENT_V2_DETERMINISTIC_TIMEOUT_SEC: float = 3.0
    AGENT_V2_AGENT_TIMEOUT_SEC: float = 12.0
    AGENT_V2_WORKFLOW_TIMEOUT_SEC: float = 30.0
    AGENT_V2_TOOL_SKILL_TIMEOUT_SEC: float = 10.0
    AGENT_V2_MAX_CONCURRENCY: int = 4

    # Main DeepAgent 根据实际工具行为升级预算，不做前置题型路由。
    MAIN_AGENT_DIRECT_HARD_DEADLINE_SEC: float = 22.0
    MAIN_AGENT_STANDARD_SOFT_DEADLINE_SEC: float = 25.0
    MAIN_AGENT_STANDARD_HARD_DEADLINE_SEC: float = 35.0
    MAIN_AGENT_RESEARCH_SOFT_DEADLINE_SEC: float = 42.0
    MAIN_AGENT_RESEARCH_HARD_DEADLINE_SEC: float = 50.0
    MAIN_AGENT_RESEARCH_TOOL_CUTOFF_SEC: float = 40.0
    MAIN_AGENT_INFLIGHT_GRACE_SEC: float = 2.0
    MAIN_AGENT_API_DEADLINE_SEC: float = 53.0
    MAIN_AGENT_MAX_TOOL_CALLS: int = 12
    MAIN_AGENT_RECURSION_LIMIT: int = 30

    # 上下文空间：业务窗口不会随模型物理窗口自动增长。
    CONTEXT_STRUCTURED_SUMMARY_MODE: str = "off"
    CONTEXT_CONVERSATION_MAX_TOKENS: int = 16_000
    CONTEXT_TOOL_LOOP_MAX_TOKENS: int = 12_000
    CONTEXT_RESEARCH_MAX_TOKENS: int = 16_000
    CONTEXT_TRIGGER_RATIO: float = 0.75
    CONTEXT_TARGET_RATIO: float = 0.50
    CONTEXT_ADMISSION_RATIO: float = 0.85
    CONTEXT_APPROXIMATE_SAFETY_MULTIPLIER: float = 1.20

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
