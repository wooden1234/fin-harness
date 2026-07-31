from app.services.memory.memory_conflict import values_conflict
from app.services.memory.memory_command import extract_preference_rule


def test_different_values_are_conflicts():
    assert values_conflict("zh-CN", "en-US")
    assert not values_conflict("zh-CN", "zh-CN")


def test_rule_candidate_does_not_treat_explicit_command_as_conflict_candidate():
    assert extract_preference_rule("请记住以后用中文回答") is None
