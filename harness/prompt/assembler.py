"""System prompt 组装与 request header。"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from typing import Any, Mapping, Sequence

from harness.prompt.sections import PromptSection, default_sections


def sha256_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def assemble_system(sections: Sequence[PromptSection] | None = None) -> str:
    items = sorted(sections or default_sections(), key=lambda item: (item.order, item.name))
    return "\n\n".join(item.text.strip() for item in items if item.text.strip())


def header_snapshot(
    *,
    system: str,
    tools: Sequence[Mapping[str, Any]],
    adapter_defaults: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "system": system,
        "tools": list(tools),
        "adapter_defaults": dict(adapter_defaults),
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "header_sha256": sha256_text(canonical),
        "system": system,
        "tools": list(tools),
        "adapter_defaults": dict(adapter_defaults),
    }
