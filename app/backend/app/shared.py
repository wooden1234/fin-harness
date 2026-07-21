"""共享类型：多组件复用的 TypedDict、类型别名、CoreState。

注意：本模块不从 `app.agents.states` 或各 agent 子包导入任何东西，避免循环依赖。
"""

from __future__ import annotations

from operator import add
from typing import Annotated, Literal, NotRequired
from typing_extensions import TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field

# ---------- 共享类型别名 ----------
AgentRoute = Literal["faq", "pdf", "account", "general", "plan"]
RiskLevel = Literal["L1", "L2", "L3", "L4"]
SubTaskType = Literal["faq", "pdf", "financial_query", "web_search", "general"]

# 意图分类：Planner 输出意图，数据源由 resolve_evidence 按意图映射（不由 LLM 直接选源）
SubTaskIntent = Literal[
    "concept_explain",    # 概念/术语/规则解释
    "product_policy",     # 产品费率、办理条件、业务政策（如信用卡年费）
    "document_qa",        # 年报/公告/研报等文档依据问答
    "structured_metric",  # 结构化财务指标查数
    "market_event",       # 最新市场/监管/实时信息
]

# 证据覆盖状态：worker 对「证据是否足以回答」的自评
CoverageStatus = Literal["covered", "partial", "uncovered", "clarify"]


# ---------- 共享 TypedDict ----------
class Citation(TypedDict, total=False):
    """检索引用；FAQ / PDF 节点写入，SSE done 带回前端。"""
    source: str
    snippet: str
    page: int
    url: str
    title: str
    published_at: str
    source_type: str
    sub_task_id: str
    parent_chunk_id: str
    root_chunk_id: str
    parent_node_id: str
    child_chunk_ids: list[str]
    evidence_child_ids: list[str]
    auto_merged: bool
    auto_merge_child_count: int


class TaskResult(TypedDict, total=False):
    """单个子任务的 Worker 返回结果"""
    sub_task_id: str
    question: str
    type: str              # SubTaskType，松散字符串避免跨模块耦合
    context: str           # 检索到的上下文原文
    citations: list[Citation]
    coverage: str          # CoverageStatus：covered/partial/uncovered/clarify
    confidence: float      # 证据置信度（可选，检索分数等）
    fallback_to_web: bool  # 兼容旧字段；等价于 coverage == "uncovered"
    fallback_reason: str
    rag_trace: dict[str, object]


# ---------- 共享 Pydantic 模型（Planner / Supervisor 等共用）----------
class Router(BaseModel):
    """Supervisor 对用户问题的分类结果：general（闲聊）还是 plan（RAG）。"""
    type: Literal["general", "plan"] = Field(
        description="general=闲聊/泛化/回溯对话；plan=需要检索知识库或文档"
    )
    logic: str = Field(
        description="一两句话说明为何选该路由"
    )


class SubTask(BaseModel):
    """单个子任务 — Planner 分解产物。

    Planner 只产出 ``intent``；``type``（首选证据工具）与 ``evidence_chain``
    （降级链）由 resolve_evidence 节点按意图映射填充。
    """
    id: str = Field(default="", description="子任务唯一标识，用于结果匹配")
    question: str = Field(description="独立的子问题，可直接检索")
    intent: str = Field(
        default="",
        description=(
            "意图类别：concept_explain=概念/规则解释 / product_policy=产品费率与业务政策 / "
            "document_qa=文档依据问答 / structured_metric=结构化财务指标查数 / "
            "market_event=最新市场或监管信息"
        ),
    )
    type: SubTaskType = Field(
        default="faq",
        description=(
            "证据工具 id（由 resolve_evidence 填充，LLM 无需输出）："
            "faq=知识库 / pdf=文档库 / financial_query=结构化财务查询 / "
            "web_search=联网检索"
        ),
    )
    evidence_chain: list[str] = Field(
        default_factory=list,
        description="有序证据降级链（由 resolve_evidence 填充，LLM 无需输出）",
    )


class PlannerOutput(BaseModel):
    """Planner 的 LLM 结构化输出"""
    tasks: list[SubTask] = Field(
        default=[],
        description="子任务列表；简单问题返回空列表"
    )


# ---------- 会话状态与 CoreState ----------
class ConversationState(TypedDict):
    """会话级短期记忆，随 LangGraph checkpoint 跨轮保存。"""

    messages: Annotated[list[AnyMessage], add_messages]

    # 会话级压缩记忆：此前多轮对话摘要，同一 conversation/thread 内长期保留。
    # 由 context_compressor 增量更新；不要在 final_answer 清空。
    conversation_summary: NotRequired[str]
    conversation_summary_until: NotRequired[str]

    # 本轮追问改写结果（压缩后、路由前写入；不进入 messages）。
    # 生命周期：本轮有效 → final_answer 收口清空；下轮 query_rewrite 再检查兜底。
    rewritten_query: NotRequired[str]
    # 改写状态：success=完成补全，passthrough=无需补全，
    # uncertain=上下文不足不可安全补全，fallback=模型异常回退原文。
    rewrite_status: NotRequired[str]


class CoreState(ConversationState):
    """所有子图的公共字段基类。"""

    # 本轮执行步骤日志；reducer=add 会累加，必须在 begin_turn_workspace 用 Overwrite([]) 重置。
    steps: NotRequired[Annotated[list[str], add]]
