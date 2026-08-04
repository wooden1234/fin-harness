"""Main DeepAgent 的结构化输出契约。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MainAgentStatement(BaseModel):
    """可独立验证的答案陈述。"""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=1800)
    statement_type: Literal["fact", "calculation", "inference", "caveat"] = "fact"
    evidence_ids: list[str] = Field(default_factory=list)
    entity: str = Field(default="", max_length=120)
    metric: str = Field(default="", max_length=120)
    period: str = Field(default="", max_length=80)
    unit: str = Field(default="", max_length=40)


class MainAgentTableRow(BaseModel):
    """表格中的一行数据，整行共享同一组证据引用。"""

    model_config = ConfigDict(extra="forbid")

    cells: list[str] = Field(min_length=1, max_length=12)
    # 允许空列表以便结构化输出先解析成功；无有效证据的行由质量门剔除。
    evidence_ids: list[str] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def validate_cells(self) -> "MainAgentTableRow":
        """拒绝空单元格和异常长文本，避免借表格绕过输出边界。"""
        if any(not str(cell).strip() for cell in self.cells):
            raise ValueError("table_cell_must_not_be_empty")
        if any(len(str(cell)) > 500 for cell in self.cells):
            raise ValueError("table_cell_too_long")
        self.evidence_ids = [
            item.strip()
            for item in self.evidence_ids
            if isinstance(item, str) and item.strip()
        ]
        return self


class MainAgentTable(BaseModel):
    """可逐行验证 Evidence 的结构化表格。"""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(
        default="",
        max_length=120,
        description=(
            "表格自然标题，如「最近一个完整财年财务对比」。"
            "不要写「主表：」，不要写「A vs B …」第二标题。"
            "本对象只有 title/columns/rows。"
        ),
    )
    columns: list[str] = Field(
        min_length=2,
        max_length=12,
        description="表头列名，至少 2 列。",
    )
    rows: list[MainAgentTableRow] = Field(
        min_length=1,
        max_length=50,
        description="数据行；每行 cells 长度必须等于 columns。",
    )

    @model_validator(mode="after")
    def validate_shape(self) -> "MainAgentTable":
        """所有数据行必须与表头保持相同列数。"""
        if any(not str(column).strip() for column in self.columns):
            raise ValueError("table_column_must_not_be_empty")
        if any(len(row.cells) != len(self.columns) for row in self.rows):
            raise ValueError("table_row_column_count_mismatch")
        return self


_RESPONSE_KNOWN_KEYS = frozenset({
    "mode",
    "direct_answer",
    "clarification",
    "heading",
    "statements",
    "tables",
    "gaps",
    "follow_ups",
})


class MainAgentResponse(BaseModel):
    """Main DeepAgent 的唯一允许输出格式（扁平：无 sections 嵌套）。"""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["direct", "grounded", "clarify"]
    direct_answer: str = Field(default="", max_length=4000)
    clarification: str = Field(default="", max_length=1000)
    heading: str = Field(
        default="",
        max_length=80,
        description=(
            "可选标题。多实体财务比较时用「结论」或留空；"
            "不要用「主表：」或与 table.title 重复的标题。"
        ),
    )
    statements: list[MainAgentStatement] = Field(
        default_factory=list,
        description=(
            "结论/解释写 fact 或 inference，表下口径差异写 caveat。"
            "比较题结论不要逐条复述表内金额。"
        ),
    )
    tables: list[MainAgentTable] = Field(
        default_factory=list,
        max_length=5,
        description=(
            "对比/数据表，仅放 title/columns/rows。"
            "Markdown 表格必须写在此处，不要塞进 statement.text。"
        ),
    )
    gaps: list[str] = Field(
        default_factory=list,
        description="缺口说明，写在响应根级 gaps。",
    )
    follow_ups: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="追问建议，写在响应根级 follow_ups，最多 3 条。",
    )

    @model_validator(mode="before")
    @classmethod
    def drop_unknown_top_level_keys(cls, data: Any) -> Any:
        """丢弃不认识的顶层字段（如模型又发明的 sections），不做语义折回。"""
        if not isinstance(data, Mapping):
            return data
        return {key: value for key, value in data.items() if key in _RESPONSE_KNOWN_KEYS}

    @model_validator(mode="after")
    def validate_mode_payload(self) -> "MainAgentResponse":
        """防止模型把无证据长文伪装成 Direct。"""
        if self.mode == "direct" and not self.direct_answer.strip():
            raise ValueError("direct_answer_required")
        if self.mode == "clarify" and not self.clarification.strip():
            raise ValueError("clarification_required")
        if self.mode == "grounded" and not (self.statements or self.tables):
            raise ValueError("grounded_content_required")
        return self


class ResearchFinding(BaseModel):
    """上下文摘要中的结论及其证据引用。"""

    claim: str = Field(min_length=1, max_length=500)
    evidence_ids: list[str] = Field(default_factory=list, max_length=16)


class ResearchContextSummary(BaseModel):
    """工具轨迹压缩用的结构化工作摘要。"""

    objective: str = Field(min_length=1, max_length=500)
    completed_questions: list[str] = Field(default_factory=list, max_length=16)
    pending_questions: list[str] = Field(default_factory=list, max_length=16)
    findings: list[ResearchFinding] = Field(default_factory=list, max_length=24)
    conflicts: list[str] = Field(default_factory=list, max_length=12)
    failed_sources: list[str] = Field(default_factory=list, max_length=12)
    next_actions: list[str] = Field(default_factory=list, max_length=12)
    evidence_ids: list[str] = Field(default_factory=list, max_length=64)


__all__ = [
    "MainAgentResponse",
    "MainAgentStatement",
    "MainAgentTable",
    "MainAgentTableRow",
    "ResearchContextSummary",
    "ResearchFinding",
]
