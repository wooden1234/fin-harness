#!/usr/bin/env python3
"""Cursor hook：拦截危险 shell 命令。

支持 beforeShellExecution 与 preToolUse(Shell) 两种事件。
stdin: JSON；stdout: {"permission": "allow"|"deny"|"ask", ...}
exit 2 = deny（与 permission deny 等效）
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any


def _extract_command(payload: dict[str, Any]) -> str:
    command = payload.get("command")
    if isinstance(command, str) and command.strip():
        return command.strip()

    tool_input = payload.get("tool_input")
    if isinstance(tool_input, dict):
        nested = tool_input.get("command")
        if isinstance(nested, str) and nested.strip():
            return nested.strip()

    return ""


def _respond(
    permission: str,
    *,
    user_message: str = "",
    agent_message: str = "",
    exit_code: int = 0,
) -> None:
    body: dict[str, str] = {"permission": permission}
    if user_message:
        body["user_message"] = user_message
    if agent_message:
        body["agent_message"] = agent_message
    sys.stdout.write(json.dumps(body, ensure_ascii=False))
    raise SystemExit(exit_code)


def _check_rm_rf(command: str) -> None:
    if not re.search(r"\brm\b", command):
        return
    if not re.search(r"-[a-zA-Z]*[rf][a-zA-Z]*|-[a-zA-Z]*[rf]", command):
        return

    dangerous_targets = (
        r"(?:\s|^)/(?:\s|\*|$)",  # / or /*
        r"(?:\s|^)/\.\.(?:\s|$)",
        r"(?:\s|^)~(?:\s|$|/)",
        r"\$HOME\b",
        r"\$\{HOME\}",
        r"(?:\s|^)\*(?:\s|$)",  # rm -rf * at word boundary
    )
    for pattern in dangerous_targets:
        if re.search(pattern, command):
            _respond(
                "deny",
                user_message="已拦截：禁止对根目录、home 或通配路径执行 rm -rf。",
                agent_message=f"Hook blocked destructive rm: {command}",
                exit_code=2,
            )


def _check_git_force_push(command: str) -> None:
    if not re.search(r"\bgit\b", command) or not re.search(r"\bpush\b", command):
        return
    if not re.search(r"(?:^|\s)(?:-f|--force)\b", command):
        return

    if re.search(r"\b(?:main|master)\b", command):
        _respond(
            "deny",
            user_message="已拦截：禁止 force push 到 main/master。",
            agent_message=f"Hook blocked force push to protected branch: {command}",
            exit_code=2,
        )

    _respond(
        "ask",
        user_message="检测到 git force push，请确认是否真的要覆盖远程历史。",
        agent_message=f"Hook flagged force push for review: {command}",
    )


def _check_chmod_777(command: str) -> None:
    if re.search(r"\bchmod\b(?:\s+-R\s+|\s+)777\b", command):
        _respond(
            "deny",
            user_message="已拦截：禁止 chmod 777（过度开放权限）。",
            agent_message=f"Hook blocked chmod 777: {command}",
            exit_code=2,
        )


def _check_other_destructive(command: str) -> None:
    patterns = (
        (r"\bmkfs\b", "mkfs"),
        (r"\bdd\s+if=", "dd"),
        (r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:", "fork bomb"),
        (r"\bshutdown\b|\breboot\b|\bpoweroff\b", "system shutdown"),
        (r">\s*/dev/sd[a-z]", "write to block device"),
        (r"\bDROP\s+DATABASE\b", "DROP DATABASE"),
        (r"\bDROP\s+TABLE\b", "DROP TABLE"),
    )
    for pattern, label in patterns:
        if re.search(pattern, command, re.IGNORECASE):
            _respond(
                "deny",
                user_message=f"已拦截：检测到高危命令（{label}）。",
                agent_message=f"Hook blocked {label}: {command}",
                exit_code=2,
            )


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        _respond(
            "deny",
            user_message="Hook 无法解析命令输入，为安全起见已拒绝执行。",
            agent_message="Hook failed to parse stdin JSON.",
            exit_code=2,
        )

    command = _extract_command(payload)
    if not command:
        _respond("allow")

    _check_rm_rf(command)
    _check_git_force_push(command)
    _check_chmod_777(command)
    _check_other_destructive(command)

    _respond("allow")


if __name__ == "__main__":
    main()
