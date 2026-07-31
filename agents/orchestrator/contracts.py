"""顶层编排器使用的稳定数据契约。

这些模型只描述模块之间传递的数据，不承担 LangGraph 运行状态的存储职责。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

BudgetTier = Literal["light", "standard", "research"]
ExecutionMode = Literal[
    "clarify",
    "general_answer",
    "faq_lookup",
    "structured_finance",
    "document_qa",
    "market_acquire",
    "research_retrieve",
    "stock_screen",
    "market_compute",
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
ErrorAction = Literal["retry", "fallback", "clarify", "fail"]
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
ClaimType = Literal["fact", "calculation", "inference", "opinion"]
ClaimImportance = Literal["critical", "major", "minor"]
EvidenceRelation = Literal["supports", "refutes", "context"]
SourceGrade = Literal["A", "B", "C", "D", "E"]
ConflictType = Literal[
    "hard_conflict",
    "temporal_update",
    "scope_difference",
    "definition_difference",
]
StatementType = Literal["fact", "inference", "caveat"]
DomainFallbackPolicy = Literal["deny", "within_scope"]


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


class ExecutionDecision(BaseModel):
    """由确定性 Resolver 生成的执行授权，模型不能直接构造。"""

    model_config = ConfigDict(extra="forbid")

    mode: ExecutionMode
    budget_tier: BudgetTier
    allowed_capabilities: list[str] = Field(default_factory=list)
    data_sources: list[str] = Field(default_factory=list)
    knowledge_scope: list[str] = Field(default_factory=list)
    tool_id: str | None = None


class RequestProfile(BaseModel):
    """语义画像与确定性执行决策。"""

    model_config = ConfigDict(extra="forbid")

    original_query: str
    normalized_query: str = ""
    domain: str = "finance"
    intents: list[str] = Field(default_factory=list)
    freshness_required: bool = False
    entities: list[str] = Field(default_factory=list)
    constraints: dict[str, Any] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)
    clarification_message: str = Field(default="", max_length=1800)
    execution: ExecutionDecision


class EvidencePolicy(BaseModel):
    """定义单个任务完成时必须满足的证据条件。"""

    required: bool = False
    min_count: int = Field(default=0, ge=0, le=100)
    require_provenance: bool = False
    require_structured_data: bool = False


class DomainPlanningScope(BaseModel):
    """Root 授予领域 Planner 的能力、来源和任务预算边界。"""

    parent_task_id: str = Field(min_length=1)
    parent_logical_task_id: str = Field(min_length=1)
    allowed_capabilities: list[str] = Field(default_factory=list)
    allowed_data_sources: list[str] = Field(default_factory=list)
    allowed_intents: list[str] = Field(default_factory=list)
    fallback_policy: DomainFallbackPolicy = "within_scope"
    freshness_required: bool = False
    evidence_policy: EvidencePolicy = Field(default_factory=EvidencePolicy)
    entities: list[str] = Field(default_factory=list)
    constraints: dict[str, Any] = Field(default_factory=dict)
    max_subtasks: int = Field(default=4, ge=1, le=16)


class TaskSpec(BaseModel):
    """任务图中的一个可执行任务。"""

    task_id: str
    logical_task_id: str | None = None
    attempt_id: str = ""
    attempt_number: int = Field(default=1, ge=1)
    idempotency_key: str = ""
    objective: str
    agent_id: str
    depends_on: list[str] = Field(default_factory=list)
    required_capabilities: list[str] = Field(default_factory=list)
    input_data: dict[str, Any] = Field(default_factory=dict)
    output_schema: str = "AgentResult"
    priority: int = Field(default=0, ge=0)
    evidence_policy: EvidencePolicy | None = None
    semantic_history: bool = False
    semantic_memory_type: str = "episodic"
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


class Claim(BaseModel):
    """答案中的一条可独立核验陈述。"""

    claim_id: str
    task_id: str = ""
    text: str = Field(min_length=1)
    subject: str = ""
    predicate: str = ""
    value: Any = None
    unit: str = ""
    currency: str = ""
    period: str | None = None
    as_of: str | None = None
    claim_type: ClaimType = "fact"
    importance: ClaimImportance = "major"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClaimEvidenceLink(BaseModel):
    """Claim 与 Evidence 之间可审计的支撑关系。"""

    claim_id: str
    evidence_id: str
    relation: EvidenceRelation = "supports"
    strength: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence_span: str = ""
    reason: str = ""


class EvidenceAssessment(BaseModel):
    """由系统规则计算的证据可靠性与时效结果。"""

    evidence_id: str
    source_grade: SourceGrade
    authority_score: float = Field(ge=0.0, le=1.0)
    freshness_score: float = Field(ge=0.0, le=1.0)
    completeness_score: float = Field(ge=0.0, le=1.0)
    usable: bool = False
    stale: bool = False
    rejection_reasons: list[str] = Field(default_factory=list)


class EvidenceConflict(BaseModel):
    """同一规范化事实下的证据冲突。"""

    conflict_id: str
    claim_key: str
    conflict_type: ConflictType
    claim_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    values: list[str] = Field(default_factory=list)
    resolved: bool = False
    preferred_claim_id: str | None = None
    resolution: str = ""


class AnswerStatement(BaseModel):
    """受约束答案中的一个句子及其事实依据。"""

    text: str = Field(min_length=1)
    claim_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    statement_type: StatementType = "fact"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ConstrainedAnswer(BaseModel):
    """只能引用已通过质量检查 Claim 的结构化答案。"""

    statements: list[AnswerStatement] = Field(default_factory=list)
    unresolved_claim_ids: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class AgentResult(BaseModel):
    """专业 Agent 对一个任务的标准输出。"""

    task_id: str
    logical_task_id: str | None = None
    attempt_id: str | None = None
    attempt_number: int = Field(default=1, ge=1)
    idempotency_key: str = ""
    agent_id: str
    status: AgentResultStatus
    answer: str = ""
    structured_data: dict[str, Any] = Field(default_factory=dict)
    evidence: list[Evidence] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    suggested_tasks: list[TaskSpec] = Field(default_factory=list)
    error_code: str = ""
    error_action: ErrorAction | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class QualityReport(BaseModel):
    """一次执行波次的质量检查结果。"""

    passed: bool = False
    missing_evidence: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    unsupported_claim_ids: list[str] = Field(default_factory=list)
    stale_evidence_ids: list[str] = Field(default_factory=list)
    rejected_evidence_ids: list[str] = Field(default_factory=list)
    conflict_ids: list[str] = Field(default_factory=list)
    claim_coverage: float = Field(default=1.0, ge=0.0, le=1.0)
    failed_task_ids: list[str] = Field(default_factory=list)
    suggested_tasks: list[TaskSpec] = Field(default_factory=list)
    reason: str = ""


__all__ = [
    "AgentResult",
    "AnswerStatement",
    "CandidateSet",
    "Claim",
    "ClaimEvidenceLink",
    "ClaimImportance",
    "ClaimType",
    "ConflictType",
    "ConstrainedAnswer",
    "BudgetTier",
    "DeepResearchReport",
    "DomainFallbackPolicy",
    "DomainPlanningScope",
    "DocumentChannel",
    "DocumentHit",
    "DocumentHitSet",
    "ErrorAction",
    "ExecutionDecision",
    "ExecutionMode",
    "Evidence",
    "EvidenceAssessment",
    "EvidenceConflict",
    "EvidenceRelation",
    "EvidencePolicy",
    "MarketFilter",
    "MarketFilterOperator",
    "MarketQueryPlan",
    "MarketSort",
    "MarketSortDirection",
    "QualityReport",
    "RequestProfile",
    "SourceGrade",
    "StatementType",
    "TaskPlan",
    "TaskSpec",
    "TaskStatus",
]
