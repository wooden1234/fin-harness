from harness.prompt.assembler import assemble_system
from harness.prompt.sections import default_sections, preference_section, preference_sections


def test_preference_section_renders_long_term_and_marks_override():
    section = preference_section({"response_language": "zh-CN"})
    assert section is not None
    assert section.name == "user_preferences"
    assert section.order == 50
    assert "[用户长期偏好]" in section.text
    assert "- response_language=zh-CN" in section.text
    assert "当前轮用户要求优先" in section.text
    assert "覆盖身份段的默认中文" in section.text


def test_preference_sections_keep_turn_overrides_after_long_term():
    sections = preference_sections(
        {"response_language": "zh-CN"},
        {"response_language": "en-US"},
    )
    assert [item.name for item in sections] == ["user_preferences", "turn_overrides"]
    assert sections[0].order == 50
    assert sections[1].order == 60
    assert "[用户长期偏好]" in sections[0].text
    assert "[本轮临时要求]" in sections[1].text
    assert "- response_language=en-US" in sections[1].text
    assert "覆盖冲突的长期偏好" in sections[1].text


def test_preference_section_omits_empty_values():
    assert preference_section({}, {}) is None
    assert preference_section(None, None) is None
    assert preference_section({"response_language": None}) is None
    assert preference_sections({}, {"response_language": None}) == ()


def test_assemble_system_places_preferences_after_stable_prefix():
    sections = (
        *default_sections(),
        *preference_sections(
            {"default_market": "US"},
            {"response_language": "en-US"},
        ),
    )
    text = assemble_system(sections)
    assert text.index("投资有风险") < text.index("memory_write")
    assert text.index("memory_write") < text.index("可用 skill")
    assert text.index("可用 skill") < text.index("[用户长期偏好]")
    assert text.index("[用户长期偏好]") < text.index("[本轮临时要求]")
    assert "default_market=US" in text
    assert "response_language=en-US" in text
