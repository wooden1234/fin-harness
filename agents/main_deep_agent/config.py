"""Main DeepAgent 的静态组装配置。"""

from __future__ import annotations

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
MAIN_SKILLS_DIR = PACKAGE_DIR / "skills"
PROJECT_SKILLS_DIR = PROJECT_ROOT / "skills"

# 后源覆盖同名技能；Main 编排技能优先于工具绑定技能。
MAIN_SKILL_SOURCES = (
    ("/skills/tool/", "Tool Skills"),
    ("/skills/main/", "Main Skills"),
)


def build_main_backend():
    """只读挂载 Main 编排技能与项目工具技能，不物理搬家。

    - ``/skills/main/`` → ``agents/main_deep_agent/skills``
    - ``/skills/tool/`` → 仓库根目录 ``skills/``
    """
    from deepagents.backends.composite import CompositeBackend
    from deepagents.backends.filesystem import FilesystemBackend

    main_backend = FilesystemBackend(
        root_dir=MAIN_SKILLS_DIR,
        virtual_mode=True,
    )
    tool_backend = FilesystemBackend(
        root_dir=PROJECT_SKILLS_DIR,
        virtual_mode=True,
    )
    return CompositeBackend(
        default=main_backend,
        routes={
            "/skills/tool/": tool_backend,
            "/skills/main/": main_backend,
        },
    )


__all__ = [
    "MAIN_SKILL_SOURCES",
    "MAIN_SKILLS_DIR",
    "PACKAGE_DIR",
    "PROJECT_ROOT",
    "PROJECT_SKILLS_DIR",
    "build_main_backend",
]
