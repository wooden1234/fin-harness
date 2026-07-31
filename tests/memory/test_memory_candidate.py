from app.services.memory.memory_command import (
    extract_preference_rule,
    parse_memory_command,
)


def test_normal_preference_matches_rule():
    assert extract_preference_rule("我喜欢表格展示") == (
        "preferred_output_format",
        "table",
    )


def test_explicit_command_does_not_match_normal_preference_rule():
    text = "请记住以后用中文回答"
    assert parse_memory_command(text) == ("response_language", "zh-CN")
    assert extract_preference_rule(text) is None
