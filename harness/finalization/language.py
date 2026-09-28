"""Final language compliance for user-visible answers."""

from __future__ import annotations

import re
from typing import Any, Mapping

_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_LATIN_WORD_RE = re.compile(r"\b[A-Za-z]{2,}\b")


def needs_language_rewrite(text: str, response_language: str | None) -> bool:
    """Use conservative thresholds so names and tickers do not trigger translation."""
    language = str(response_language or "").strip().lower()
    if not language:
        return False
    cjk_count = len(_CJK_RE.findall(text))
    latin_words = len(_LATIN_WORD_RE.findall(text))
    if language.startswith("en"):
        return cjk_count >= 4
    if language.startswith("zh"):
        return cjk_count == 0 and latin_words >= 8
    return False


async def enforce_response_language(
    text: str,
    *,
    preferences: Mapping[str, Any] | None,
    llm: Any,
) -> str:
    values = dict(preferences or {})
    language = str(values.get("response_language") or "").strip()
    if not needs_language_rewrite(text, language):
        return text
    output_format = str(values.get("preferred_output_format") or "markdown")
    system = (
        "You are a lossless response-language compliance editor. "
        f"Rewrite the answer in {language} and keep the output format as {output_format}. "
        "Preserve every fact, number, sign, date, unit, company name, citation, URL, "
        "Markdown structure, warning, and uncertainty. Do not add analysis or new facts. "
        "Return only the rewritten answer."
    )
    try:
        rewritten = str(await llm.complete(system=system, prompt=text) or "").strip()
    except Exception:  # noqa: BLE001
        return text
    return rewritten or text


__all__ = ["enforce_response_language", "needs_language_rewrite"]
