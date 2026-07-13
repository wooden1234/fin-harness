"""predefined 白名单查询能力。

包导入时只暴露无副作用的模型导出；execution/tool_selection
按需懒加载，避免与 FinAgentState / services 形成循环导入。
工作流编排见 workflows.predefined。
"""

from __future__ import annotations

from .intent import FinancialFactQuery, FinancialQueryIntent

__all__ = [
    "FinancialFactQuery",
    "FinancialQueryIntent",
    "PREDEFINED_SQL_DICT",
    "PredefinedTemplateRegistry",
    "PredefinedToolSelectionResult",
    "VALID_TEMPLATE_IDS",
    "build_predefined_sql_query",
    "execute_predefined_sql",
    "execute_predefined_sql_query",
    "predefined_sql",
    "select_predefined_tool",
    "template_catalog_text",
]


def __getattr__(name: str):
    if name in {
        "PREDEFINED_SQL_DICT",
        "PredefinedTemplateRegistry",
        "VALID_TEMPLATE_IDS",
        "template_catalog_text",
    }:
        from .whitelist import (
            PREDEFINED_SQL_DICT,
            PredefinedTemplateRegistry,
            VALID_TEMPLATE_IDS,
            template_catalog_text,
        )

        exports = {
            "PREDEFINED_SQL_DICT": PREDEFINED_SQL_DICT,
            "PredefinedTemplateRegistry": PredefinedTemplateRegistry,
            "VALID_TEMPLATE_IDS": VALID_TEMPLATE_IDS,
            "template_catalog_text": template_catalog_text,
        }
        return exports[name]

    if name in {"build_predefined_sql_query", "execute_predefined_sql", "execute_predefined_sql_query"}:
        from .execution import (
            build_predefined_sql_query,
            execute_predefined_sql,
            execute_predefined_sql_query,
        )

        exports = {
            "build_predefined_sql_query": build_predefined_sql_query,
            "execute_predefined_sql": execute_predefined_sql,
            "execute_predefined_sql_query": execute_predefined_sql_query,
        }
        return exports[name]

    if name in {"PredefinedToolSelectionResult", "predefined_sql", "select_predefined_tool"}:
        from .tool_selection import (
            PredefinedToolSelectionResult,
            predefined_sql,
            select_predefined_tool,
        )

        exports = {
            "PredefinedToolSelectionResult": PredefinedToolSelectionResult,
            "predefined_sql": predefined_sql,
            "select_predefined_tool": select_predefined_tool,
        }
        return exports[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
