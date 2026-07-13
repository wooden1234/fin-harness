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


class TaskResult(TypedDict, total=False):
    """单个子任务的 Worker 返回结果"""
    sub_task_id: str
    question: str
    type: str              # SubTaskType，松散字符串避免跨模块耦合
    context: str           # 检索到的上下文原文
    citations: list[Citation]
    fallback_to_web: bool
    fallback_reason: str


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
    """单个子任务 — Planner 分解产物"""
    id: str = Field(default="", description="子任务唯一标识，用于结果匹配")
    question: str = Field(description="独立的子问题，可直接检索")
    type: SubTaskType = Field(
        default="faq",
        description=(
            "faq=知识库 / pdf=文档库 / financial_query=结构化财务查询 / "
            "web_search=联网检索 / general=无需检索"
        ),
    )


class PlannerOutput(BaseModel):
    """Planner 的 LLM 结构化输出"""
    tasks: list[SubTask] = Field(
        default=[],
        description="子任务列表；简单问题返回空列表"
    )


# ---------- CoreState：所有图共享的基类 ----------
class CoreState(TypedDict):
    """所有子图的公共字段基类。"""
    messages: Annotated[list[AnyMessage], add_messages]
    steps: NotRequired[Annotated[list[str], add]]
