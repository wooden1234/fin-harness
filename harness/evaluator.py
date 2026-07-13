"""在线和离线评测入口。"""

from __future__ import annotations

from typing import Any


def evaluate_run(result: dict[str, Any]) -> dict[str, Any]:
    """预留评测入口，后续接入 grounding、citation、compliance judge。"""
    return {"passed": True, "checks": [], "result_keys": sorted(result.keys())}
