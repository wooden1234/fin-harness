from __future__ import annotations

import pytest

from harness.finalization.language import enforce_response_language, needs_language_rewrite


def test_language_mismatch_detection_is_conservative():
    assert needs_language_rewrite("西安当前天气为阴天。", "en-US") is True
    assert needs_language_rewrite("Xi'an: 19.5°C", "en-US") is False
    assert needs_language_rewrite("The current weather in Xi'an is cloudy and humid.", "zh-CN") is True


@pytest.mark.asyncio
async def test_language_gate_preserves_matching_answer_without_llm_call():
    class NeverCalled:
        async def complete(self, **_kwargs):
            raise AssertionError("language already matches")

    answer = "The current temperature is 19.5°C."
    assert await enforce_response_language(
        answer,
        preferences={"response_language": "en-US"},
        llm=NeverCalled(),
    ) == answer


@pytest.mark.asyncio
async def test_language_gate_rewrites_tool_style_chinese_answer_for_english_preference():
    translated = "## Current weather in Xi'an\n\nTemperature: 19.5°C"

    class Translator:
        def __init__(self):
            self.request = {}

        async def complete(self, **kwargs):
            self.request = kwargs
            return translated

    llm = Translator()
    result = await enforce_response_language(
        "## 西安当前天气\n\n气温：19.5°C",
        preferences={
            "response_language": "en-US",
            "preferred_output_format": "markdown",
        },
        llm=llm,
    )

    assert result == translated
    assert "en-US" in llm.request["system"]
    assert "19.5°C" in llm.request["prompt"]
