"""Main DeepAgent 的结构化输出契约。"""

from __future__ import annotations

from typing import Literal

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
    evidence_ids: list[str] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_cells(self) -> "MainAgentTableRow":
        """拒绝空单元格和异常长文本，避免借表格绕过输出边界。"""
        if any(not str(cell).strip() for cell in self.cells):
            raise ValueError("table_cell_must_not_be_empty")
        if any(len(str(cell)) > 500 for cell in self.cells):
            raise ValueError("table_cell_too_long")
        return self


class MainAgentTable(BaseModel):
    """可逐行验证 Evidence 的结构化表格。"""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(default="", max_length=120)
    columns: list[str] = Field(min_length=2, max_length=12)
    rows: list[MainAgentTableRow] = Field(min_length=1, max_length=50)

    @model_validator(mode="after")
    def validate_shape(self) -> "MainAgentTable":
        """所有数据行必须与表头保持相同列数。"""
        if any(not str(column).strip() for column in self.columns):
            raise ValueError("table_column_must_not_be_empty")
        if any(len(row.cells) != len(self.columns) for row in self.rows):
            raise ValueError("table_row_column_count_mismatch")
        return self


class MainAgentSection(BaseModel):
    """Grounded 回答中的结构化章节。"""

    model_config = ConfigDict(extra="forbid")

    heading: str = Field(min_length=1, max_length=80)
    statements: list[MainAgentStatement] = Field(default_factory=list)
    tables: list[MainAgentTable] = Field(default_factory=list)


class MainAgentResponse(BaseModel):
    """Main DeepAgent 的唯一允许输出格式。"""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["direct", "grounded", "clarify"]
    direct_answer: str = Field(default="", max_length=4000)
    clarification: str = Field(default="", max_length=1000)
    sections: list[MainAgentSection] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_mode_payload(self) -> "MainAgentResponse":
        """防止模型把无证据长文伪装成 Direct。"""
        if self.mode == "direct" and not self.direct_answer.strip():
            raise ValueError("direct_answer_required")
        if self.mode == "clarify" and not self.clarification.strip():
            raise ValueError("clarification_required")
        if self.mode == "grounded" and not any(
            section.statements or section.tables for section in self.sections
        ):
            raise ValueError("grounded_sections_required")
        return self


__all__ = [
    "MainAgentResponse",
    "MainAgentSection",
    "MainAgentStatement",
    "MainAgentTable",
    "MainAgentTableRow",
]
