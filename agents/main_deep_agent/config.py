"""Main DeepAgent 的静态组装配置。"""

from __future__ import annotations

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
MAIN_SKILL_SOURCES = ("/skills/",)


def build_main_backend():
    """为 Skill 提供限制在当前包目录内的虚拟只读视图。"""
    from deepagents.backends.filesystem import FilesystemBackend

    return FilesystemBackend(root_dir=PACKAGE_DIR, virtual_mode=True)


__all__ = ["MAIN_SKILL_SOURCES", "PACKAGE_DIR", "build_main_backend"]
