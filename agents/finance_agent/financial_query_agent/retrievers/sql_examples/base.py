"""SQL 示例检索器抽象基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SqlExample:
    """用于 few-shot 注入的问答示例。"""

    category: str
    question: str
    sql: str
    notes: list[str] = field(default_factory=list)


class BaseSqlExampleRetriever(ABC):
    """定义 SQL 示例检索器最小接口。"""

    @abstractmethod
    def get_examples(self, query: str, *, k: int = 3) -> list[SqlExample]:
        """根据用户问题返回最相关的示例。"""

    @staticmethod
    def format_examples(examples: list[SqlExample]) -> str:
        if not examples:
            return "暂无可用示例，请严格依据 Schema 与用户问题生成安全 SQL。"

        blocks: list[str] = []
        for index, example in enumerate(examples, start=1):
            lines = [
                f"示例{index}（{example.category}）",
                f"问题：{example.question}",
                f"SQL：{example.sql.strip()}",
            ]
            if example.notes:
                lines.append(f"要点：{'；'.join(example.notes)}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)


__all__ = ["BaseSqlExampleRetriever", "SqlExample"]

