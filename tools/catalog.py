"""工具目录加载：import 各模块以触发 ``register_tool``。"""

from __future__ import annotations

import importlib

# 新增工具时在此登记模块路径，启动/绑定时统一加载。
TOOL_MODULES: tuple[str, ...] = (
    "tools.weather",
    "tools.web_search",
)

_loaded = False


def load_all_tools(*, force: bool = False) -> list[str]:
    """加载目录中的工具模块；返回已加载模块名。"""
    global _loaded
    if _loaded and not force:
        return list(TOOL_MODULES)

    loaded: list[str] = []
    for module_name in TOOL_MODULES:
        importlib.import_module(module_name)
        loaded.append(module_name)
    _loaded = True
    return loaded


__all__ = ["TOOL_MODULES", "load_all_tools"]
