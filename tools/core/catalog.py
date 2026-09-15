"""工具目录加载：import 各模块以触发 ``register_tool``。"""

from __future__ import annotations

import importlib
import sys

# 新增工具时在此登记模块路径，启动/绑定时统一加载。
TOOL_MODULES: tuple[str, ...] = (
    "tools.weather",
    "tools.iwencai",
    "tools.web_search",
    "tools.knowledge",
    "tools.finance",
    "tools.calculation",
    "tools.analysis",
    "mcp.adapters.tyc",
)

_loaded = False


def load_all_tools(*, force: bool = False) -> list[str]:
    """加载目录中的工具模块；返回已加载模块名。"""
    global _loaded
    from tools.core.registry import list_registered_tools

    if _loaded and not force and list_registered_tools():
        return list(TOOL_MODULES)

    # 模块可能已因别处 import 进 sys.modules；注册表被清空后必须 reload
    # 才能再次执行模块级 register_tool。
    reload_existing = force or _loaded or not list_registered_tools()

    loaded: list[str] = []
    for module_name in TOOL_MODULES:
        existing = sys.modules.get(module_name)
        if existing is not None and reload_existing:
            importlib.reload(existing)
        else:
            importlib.import_module(module_name)
        loaded.append(module_name)
    _loaded = True

    if not list_registered_tools():
        raise RuntimeError("tool catalog is empty")
    return loaded


__all__ = ["TOOL_MODULES", "load_all_tools"]
