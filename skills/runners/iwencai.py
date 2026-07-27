"""同花顺问财外部 Skill 的受限执行器。"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys
from typing import Any

from app.core.config import PROJECT_ROOT, settings

_SUPPORTED_SKILLS = {
    "hithink-astock-selector": "scripts/cli.py",
}


def _skill_root() -> Path:
    configured = Path(settings.IWENCAI_SKILL_ROOT)
    if not configured.is_absolute():
        configured = PROJECT_ROOT / configured
    return configured.resolve()


def _entrypoint(skill_id: str) -> Path:
    relative = _SUPPORTED_SKILLS.get(skill_id)
    if relative is None:
        raise ValueError(f"skill_not_allowed:{skill_id}")
    root = _skill_root()
    script = (root / relative).resolve()
    try:
        script.relative_to(root)
    except ValueError as exc:
        raise ValueError("skill_entrypoint_outside_root") from exc
    if not script.is_file():
        raise FileNotFoundError(f"skill_entrypoint_not_found:{skill_id}")
    return script


def _validate_positive_int(value: int, name: str) -> int:
    if value < 1:
        raise ValueError(f"{name}_must_be_positive")
    return value


async def run_installed_skill(
    skill_id: str,
    *,
    query: str,
    page: int = 1,
    limit: int = 10,
    call_type: str = "normal",
) -> dict[str, Any]:
    """执行白名单问财 Skill，并返回 JSON 结果。"""
    if not settings.IWENCAI_SKILL_RUNNER_ENABLED:
        return {"ok": False, "error": "skill_runner_disabled"}
    if not query.strip():
        return {"ok": False, "error": "query_must_not_be_empty"}
    if call_type not in {"normal", "retry"}:
        return {"ok": False, "error": "invalid_call_type"}

    try:
        page = _validate_positive_int(page, "page")
        limit = _validate_positive_int(limit, "limit")
        if limit > settings.IWENCAI_MAX_LIMIT:
            return {
                "ok": False,
                "error": f"limit_must_be_between_1_and_{settings.IWENCAI_MAX_LIMIT}",
            }
        script = _entrypoint(skill_id)
    except (ValueError, FileNotFoundError) as exc:
        return {"ok": False, "error": str(exc)}

    api_key = settings.IWENCAI_API_KEY.strip()
    if not api_key:
        return {
            "ok": False,
            "error": "iwencai_not_configured",
            "message": "请配置 IWENCAI_API_KEY 后再调用问财 Skill。",
        }

    # 只把 Skill 所需变量注入子进程，避免暴露父进程的完整环境。
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONUNBUFFERED": "1",
        "IWENCAI_API_KEY": api_key,
    }
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(script),
        "--query",
        query.strip(),
        "--page",
        str(page),
        "--limit",
        str(limit),
        "--call-type",
        call_type,
        "--timeout",
        str(int(settings.IWENCAI_TIMEOUT_SEC)),
        cwd=str(script.parent.parent),
        env=environment,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout, _stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=settings.IWENCAI_SKILL_RUNNER_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        process.kill()
        await process.communicate()
        return {"ok": False, "error": "skill_runner_timeout"}

    if len(stdout) > settings.IWENCAI_SKILL_RUNNER_MAX_OUTPUT_BYTES:
        return {"ok": False, "error": "skill_runner_output_too_large"}

    try:
        payload = json.loads(stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {"ok": False, "error": "skill_runner_invalid_json"}

    if process.returncode != 0:
        error = payload.get("error") if isinstance(payload, dict) else ""
        return {
            "ok": False,
            "error": "skill_runner_failed",
            "provider_error": str(error or "official_skill_failed"),
            "data": payload,
        }

    return {
        "ok": True,
        "provider": "iwencai",
        "skill_id": skill_id,
        "data": payload,
    }


__all__ = ["run_installed_skill"]
