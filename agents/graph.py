"""V1 根图兼容门面。

实体实现位于 ``agent-v1/graph.py``。保留本模块是为了兼容历史脚本和外部
调用方；加载采用惰性 import，V2 不会因导入 ``agents`` 而构建 V1 图。
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


def _v1_graph_module():
    return import_module("agent-v1.graph")


def build_graph(*args: Any, **kwargs: Any):
    return _v1_graph_module().build_graph(*args, **kwargs)


def compile_graph(*args: Any, **kwargs: Any):
    return _v1_graph_module().compile_graph(*args, **kwargs)


def get_graph(*args: Any, **kwargs: Any):
    return _v1_graph_module().get_graph(*args, **kwargs)


def get_graph_with_memory(*args: Any, **kwargs: Any):
    return _v1_graph_module().get_graph_with_memory(*args, **kwargs)


def reset_graph_cache() -> None:
    _v1_graph_module().reset_graph_cache()


__all__ = [
    "build_graph",
    "compile_graph",
    "get_graph",
    "get_graph_with_memory",
    "reset_graph_cache",
]
