from harness.prompt.assembler import assemble_system
from harness.prompt.sections import default_sections, preference_section


def test_preference_section_renders_long_term_and_marks_override():
    section = preference_section({"response_language": "zh-CN"})
    assert section is not None
    assert section.name == "user_preferences"
    assert section.order == 25
    assert "[用户长期偏好]" in section.text
    assert "- response_language=zh-CN" in section.text
    assert "当前轮用户要求优先" in section.text
    assert "覆盖身份段的默认中文" in section.text


def test_preference_section_appends_turn_overrides():
    section = preference_section(
        {"response_language": "zh-CN"},
        {"response_language": "en-US"},
    )
    assert section is not None
    assert "[用户长期偏好]" in section.text
    assert "[本轮临时要求]" in section.text
    assert "- response_language=en-US" in section.text
    assert "覆盖冲突的长期偏好" in section.text


def test_preference_section_omits_empty_values():
    assert preference_section({}, {}) is None
    assert preference_section(None, None) is None
    assert preference_section({"response_language": None}) is None


def test_assemble_system_places_preferences_between_compliance_and_tools():
    pref = preference_section({"default_market": "US"})
    text = assemble_system((*default_sections(), pref))
    assert text.index("投资有风险") < text.index("[用户长期偏好]")
    assert text.index("[用户长期偏好]") < text.index("memory_write")
    assert "default_market=US" in text
