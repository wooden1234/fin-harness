from app.services.memory.memory_command import (
    extract_turn_preferences,
    parse_memory_command,
    parse_memory_rule_action,
)
from app.services.memory.memory_policy import validate_preference


def test_only_explicit_command_is_parsed():
    assert parse_memory_command("请记住以后用中文回答") == (
        "response_language",
        "zh-CN",
    )
    assert parse_memory_command("我今天想看美股") is None


def test_whitelist_rejects_unknown_key_and_value():
    validate_preference("default_market", "US")
    try:
        validate_preference("holdings", "600000")
    except ValueError:
        pass
    else:
        raise AssertionError("敏感/未知偏好 key 不应通过白名单")

    try:
        validate_preference("default_market", "UNKNOWN")
    except ValueError:
        pass
    else:
        raise AssertionError("枚举之外的 value 不应通过白名单")


def test_sync_actions_are_resolved_only_by_rules():
    remember = parse_memory_rule_action("请记住以后用英文回答")
    update = parse_memory_rule_action("把回答语言改成中文")
    delete = parse_memory_rule_action("忘记我的回答语言偏好")

    assert (remember.kind, remember.memory_key, remember.value) == (
        "remember",
        "response_language",
        "en-US",
    )
    assert (update.kind, update.memory_key, update.value) == (
        "update",
        "response_language",
        "zh-CN",
    )
    assert (delete.kind, delete.memory_key, delete.value) == (
        "delete",
        "response_language",
        None,
    )


def test_unresolved_sync_action_does_not_fall_back_to_llm():
    action = parse_memory_rule_action("请记住我的特殊风格")
    conflicting_update = parse_memory_rule_action("把输出格式改成英文")
    assert action.kind == "remember"
    assert action.resolved is False
    assert conflicting_update.kind == "update"
    assert conflicting_update.resolved is False


def test_temporary_sensitive_implicit_and_business_are_separated():
    assert parse_memory_rule_action("这次请用英文回答").kind == "temporary"
    assert parse_memory_rule_action("请记住我的手机号 13800138000").kind == "sensitive"
    assert (
        parse_memory_rule_action("请记住 token=abc123456789").kind
        == "sensitive"
    )
    assert parse_memory_rule_action("我喜欢表格展示").kind == "implicit"
    assert parse_memory_rule_action("贵州茅台的股价是多少").kind == "ordinary"


def test_temporary_preferences_are_structured_for_current_turn_only():
    assert extract_turn_preferences("这次请用英文并用表格回答") == {
        "response_language": "en-US",
        "preferred_output_format": "table",
    }
    assert extract_turn_preferences("以后默认用英文回答") == {}


def test_ambiguous_management_requires_memory_context():
    assert parse_memory_rule_action("删除之前那个记忆").kind == "delete"
    assert parse_memory_rule_action("修改一下我的偏好").kind == "update"
    assert parse_memory_rule_action("删除本地文件").kind == "ordinary"
