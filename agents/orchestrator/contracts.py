"""顶层编排器使用的稳定数据契约。

这些模型只描述模块之间传递的数据，不承担 LangGraph 运行状态的存储职责。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

RequestComplexity = Literal["simple", "single_capability", "compound"]
DataSourceType = Literal[
    "market",
    "research",
    "finance_rag",
    "upstream_data",
    "none",
]
OperationType = Literal[
    "acquire",
    "retrieve",
    "compute",
    "analyze",
    "answer",
    "deep_research",
]
TaskStatus = Literal[
    "pending",
    "running",
    "completed",
    "partial",
    "uncovered",
    "clarify",
    "failed",
]
AgentResultStatus = Literal[
    "completed",
    "partial",
    "uncovered",
    "clarify",
    "failed",
]
MarketFilterOperator = Literal[
    "eq",
    "ne",
    "gt",
    "gte",
    "lt",
    "lte",
    "in",
    "not_in",
    "contains",
    "between",
]
MarketSortDirection = Literal["asc", "desc"]
DocumentChannel = Literal["report", "announcement", "news", "unknown"]


class MarketFilter(BaseModel):
    """市场数据过滤条件。"""

    field: str = Field(min_length=1)
    operator: MarketFilterOperator
    value: Any


class MarketSort(BaseModel):
    """市场数据排序条件。"""

    field: str = Field(min_length=1)
    direction: MarketSortDirection = "desc"


class MarketQueryPlan(BaseModel):
    """自然语言市场问题解析后的确定性查询计划。"""

    universe: str = Field(min_length=1, description="A股、港股、美股、基金等范围")
    filters: list[MarketFilter] = Field(default_factory=list)
    enrichments: list[str] = Field(default_factory=list)
    sort: list[MarketSort] = Field(default_factory=list)
    group_by: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    limit: int = Field(default=20, ge=1, le=500)
    as_of: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CandidateSet(BaseModel):
    """选股及市场查询产生的标准化候选数据集。"""

    schema_version: str = "1.0"
    dataset_id: str = Field(min_length=1)
    universe: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    as_of: str
    rows: list[dict[str, Any]] = Field(default_factory=list)
    query_plan: MarketQueryPlan | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentHit(BaseModel):
    """公告或研报搜索命中的一条文档元数据。"""

    document_id: str = ""
    title: str = ""
    summary: str = ""
    published_at: str | None = None
    organization: str = ""
    rating: str = ""
    target_price: str | float | int | None = None
    url: str | None = None
    pdf_url: str | None = None
    provider: str = "iwencai"
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentHitSet(BaseModel):
    """研报、公告等搜索结果的统一文档契约。"""

    schema_version: str = "1.0"
    query: str
    channel: DocumentChannel = "unknown"
    documents: list[DocumentHit] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DeepResearchReport(BaseModel):
    """Deep Agent 研究结论的可追溯结构化输出。"""

    schema_version: str = "1.0"
    query: str
    summary: str
    source_tools: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RequestProfile(BaseModel):
    """入口分析后的请求画像。"""

    original_query: str
    normalized_query: str = ""
    domain: str = "finance"
    intents: list[str] = Field(default_factory=list)
    complexity: RequestComplexity = "simple"
    data_sources: list[DataSourceType] = Field(default_factory=list)
    operation_type: OperationType = "answer"
    freshness_required: bool = False
    entities: list[str] = Field(default_factory=list)
    constraints: dict[str, Any] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)
    preferred_agent: str | None = None


class TaskSpec(BaseModel):
    """任务图中的一个可执行任务。"""

    task_id: str
    objective: str
    agent_id: str
    depends_on: list[str] = Field(default_factory=list)
    required_capabilities: list[str] = Field(default_factory=list)
    input_data: dict[str, Any] = Field(default_factory=dict)
    output_schema: str = "AgentResult"
    priority: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskPlan(BaseModel):
    """一次请求对应的任务计划及其依赖关系。"""

    plan_id: str
    query: str
    tasks: list[TaskSpec] = Field(default_factory=list)
    rationale: str = ""
    max_replans: int = Field(default=1, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_task_dependencies(self) -> "TaskPlan":
        """校验任务 ID、依赖引用和依赖环。"""
        task_ids = [task.task_id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("task_id_must_be_unique")

        known_ids = set(task_ids)
        dependencies = {task.task_id: set(task.depends_on) for task in self.tasks}
        for task_id, depends_on in dependencies.items():
            unknown = depends_on - known_ids
            if unknown:
                raise ValueError(
                    f"unknown_task_dependency:{task_id}:{','.join(sorted(unknown))}"
                )
            if task_id in depends_on:
                raise ValueError(f"task_cannot_depend_on_itself:{task_id}")

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(task_id: str) -> None:
            if task_id in visiting:
                raise ValueError(f"cyclic_task_dependency:{task_id}")
            if task_id in visited:
                return
            visiting.add(task_id)
            for dependency in dependencies[task_id]:
                visit(dependency)
            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in task_ids:
            visit(task_id)
        return self


class Evidence(BaseModel):
    """可被答案综合和引用的单条证据。"""

    evidence_id: str
    task_id: str = ""
    source_type: str
    provider: str = ""
    title: str = ""
    content: str = ""
    url: str | None = None
    published_at: str | None = None
    observed_at: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentResult(BaseModel):
    """专业 Agent 对一个任务的标准输出。"""

    task_id: str
    agent_id: str
    status: AgentResultStatus
    answer: str = ""
    structured_data: dict[str, Any] = Field(default_factory=dict)
    evidence: list[Evidence] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    suggested_tasks: list[TaskSpec] = Field(default_factory=list)
    error_code: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class QualityReport(BaseModel):
    """一次执行波次的质量检查结果。"""

    passed: bool = False
    missing_evidence: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    failed_task_ids: list[str] = Field(default_factory=list)
    suggested_tasks: list[TaskSpec] = Field(default_factory=list)
    reason: str = ""


__all__ = [
    "AgentResult",
    "CandidateSet",
    "DataSourceType",
    "DeepResearchReport",
    "DocumentChannel",
    "DocumentHit",
    "DocumentHitSet",
    "Evidence",
    "MarketFilter",
    "MarketFilterOperator",
    "MarketQueryPlan",
    "MarketSort",
    "MarketSortDirection",
    "OperationType",
    "QualityReport",
    "RequestComplexity",
    "RequestProfile",
    "TaskPlan",
    "TaskSpec",
    "TaskStatus",
]
