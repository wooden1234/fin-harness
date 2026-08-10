"""LLM 工厂配置测试。"""

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


def test_finance_llm_uses_openai_compatible_endpoint(monkeypatch) -> None:
    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(llm_module, "ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setattr(llm_module.settings, "FINANCE_LLM_API_KEY", "EMPTY")
    monkeypatch.setattr(
        llm_module.settings, "FINANCE_LLM_BASE_URL", "http://127.0.0.1:8001/v1"
    )
    monkeypatch.setattr(llm_module.settings, "FINANCE_LLM_MODEL", "finalign-awq")
    monkeypatch.setattr(llm_module.settings, "FINANCE_LLM_TEMPERATURE", 0.0)
    monkeypatch.setattr(llm_module.settings, "FINANCE_LLM_TIMEOUT_SEC", 120.0)
    monkeypatch.setattr(llm_module.settings, "FINANCE_LLM_ENABLE_THINKING", False)
    monkeypatch.setattr(
        llm_module,
        "_finance_llm_reachable",
        lambda **_kwargs: True,
    )
    llm_module._get_finance_llm_client.cache_clear()
    llm_module.reset_finance_llm_probe_cache()

    llm = llm_module.get_finance_llm()

    assert llm is not None
    assert captured["model"] == "finalign-awq"
    assert captured["base_url"] == "http://127.0.0.1:8001/v1"
    assert captured["api_key"] == "EMPTY"
    assert captured["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    llm_module._get_finance_llm_client.cache_clear()
    llm_module.reset_finance_llm_probe_cache()


def test_finance_llm_falls_back_to_faq_when_unconfigured(monkeypatch) -> None:
    sentinel = object()

    monkeypatch.setattr(llm_module.settings, "FINANCE_LLM_BASE_URL", "")
    monkeypatch.setattr(llm_module.settings, "FINANCE_LLM_MODEL", "")
    monkeypatch.setattr(llm_module, "get_faq_llm", lambda: sentinel)
    llm_module.reset_finance_llm_probe_cache()

    assert llm_module.get_finance_llm() is sentinel
    llm_module.reset_finance_llm_probe_cache()


def test_finance_llm_falls_back_to_deepseek_when_vllm_down(monkeypatch) -> None:
    sentinel = object()

    monkeypatch.setattr(
        llm_module.settings, "FINANCE_LLM_BASE_URL", "http://127.0.0.1:8001/v1"
    )
    monkeypatch.setattr(llm_module.settings, "FINANCE_LLM_MODEL", "finalign-awq")
    monkeypatch.setattr(
        llm_module,
        "_finance_llm_reachable",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(llm_module, "get_faq_llm", lambda: sentinel)
    llm_module.reset_finance_llm_probe_cache()

    assert llm_module.get_finance_llm() is sentinel
    llm_module.reset_finance_llm_probe_cache()


def test_finance_llm_probe_uses_models_endpoint(monkeypatch) -> None:
    calls: list[dict] = []

    class FakeResponse:
        status_code = 200

    def fake_get(url, **kwargs):
        calls.append({"url": url, **kwargs})
        return FakeResponse()

    monkeypatch.setattr(llm_module.httpx, "get", fake_get)
    monkeypatch.setattr(llm_module.settings, "FINANCE_LLM_PROBE_TIMEOUT_SEC", 1.5)
    monkeypatch.setattr(llm_module.settings, "FINANCE_LLM_PROBE_TTL_SEC", 0.0)
    llm_module.reset_finance_llm_probe_cache()

    assert llm_module._finance_llm_reachable(
        base_url="http://127.0.0.1:8001/v1",
        api_key="EMPTY",
    )
    assert calls[0]["url"] == "http://127.0.0.1:8001/v1/models"
    assert calls[0]["headers"]["Authorization"] == "Bearer EMPTY"
    llm_module.reset_finance_llm_probe_cache()
