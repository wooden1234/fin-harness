"""DeepSeek 工具调用配置测试。"""

from agents import llm as llm_module


def test_deepseek_tool_agent_disables_thinking(monkeypatch) -> None:
    captured = {}

    class FakeChatDeepSeek:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(llm_module, "ChatDeepSeek", FakeChatDeepSeek)
    monkeypatch.setattr(llm_module.settings, "DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(llm_module.settings, "DEEPSEEK_THINKING_ENABLED", False)

    llm_module._build_deepseek_llm(temperature=0.0)

    assert captured["max_retries"] == 0
    assert captured["extra_body"] == {"thinking": {"type": "disabled"}}
