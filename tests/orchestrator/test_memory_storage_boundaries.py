from __future__ import annotations

import ast
from pathlib import Path


_MEMORY_MODULE_PREFIX = "app.services.memory"
_ALLOWED_MEMORY_MODULE = "app.services.memory.memory_loader"
_ALLOWED_AGGREGATE_NAMES = {
    "MemoryLoader",
    "MemoryProjection",
    "MemoryProjectionEntry",
}


def _forbidden_import(node: ast.AST) -> bool:
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        if module == _ALLOWED_MEMORY_MODULE:
            return False
        if module == _MEMORY_MODULE_PREFIX:
            return any(
                alias.name not in _ALLOWED_AGGREGATE_NAMES
                for alias in node.names
            )
        return module.startswith(f"{_MEMORY_MODULE_PREFIX}.")
    if isinstance(node, ast.Import):
        return any(
            alias.name.startswith(_MEMORY_MODULE_PREFIX)
            and alias.name != _ALLOWED_MEMORY_MODULE
            for alias in node.names
        )
    return False


def test_business_agents_do_not_import_memory_storage_directly():
    project_root = Path(__file__).resolve().parents[2]
    business_roots = (
        project_root / "agents" / "finance_agent",
        project_root / "agents" / "general_agent",
        project_root / "agents" / "main_deep_agent",
    )
    violations: list[str] = []

    for root in business_roots:
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if _forbidden_import(node):
                    violations.append(
                        f"{path.relative_to(project_root)}:{node.lineno}"
                    )

    assert violations == []
