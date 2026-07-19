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
    ES_BM25F_TITLE_PHRASE_BOOST: float = 3.0
    ES_BM25F_SECTION_PHRASE_BOOST: float = 2.5
    ES_BM25F_MAX_PHRASE_QUERIES: int = 4
    ES_BM25F_MIN_PHRASE_LENGTH: int = 2
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
    AGENT_ROUTER_TEMPERATURE: float = 0.0
    PDF_QUERY_FILTER_MIN_CONFIDENCE: float = 0.85
    PDF_KB_UNSUPPORTED_MIN_CONFIDENCE: float = 0.90
    AGENT_FAQ_TEMPERATURE: float = 0.3
    FAQ_MIN_RELEVANCE_SCORE: float = 0.35
    PDF_MIN_RELEVANCE_SCORE: float = 0.35
    PDF_RETRIEVAL_QUALITY_MODEL_PATH: str = str(
        PROJECT_ROOT / "retrieval/models/pdf_retrieval_quality.json"
    )

    # Embedding（OpenAI 兼容：DashScope / 讯飞星辰 MaaS）
    EMBEDDING_PROVIDER: str = "dashscope"
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_MODEL: str = "text-embedding-v2"
    EMBEDDING_BASE_URL: str = ""
    EMBEDDING_BATCH_SIZE: int = 0
    QWEN_API_KEY: str = ""
    DASHSCOPE_API_KEY: str = ""
    QWEN_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    RERANK_ENABLED: bool = False
    RERANK_PROVIDER: str = "dashscope"
    RERANK_API_KEY: str = ""
    RERANK_MODEL: str = "qwen3-rerank"
    RERANK_BASE_URL: str = (
        "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
    )
    RERANK_CANDIDATE_TOP_K: int = 20
    RERANK_TIMEOUT_SEC: float = 30.0
    RERANK_RETURN_DOCUMENTS: bool = True

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
