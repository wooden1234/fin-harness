from app.services.memory.memory_compliance import scan_memory


class Record:
    id = "m-1"
    display_text = "用户偏好中文"
    search_text = "response_language zh-CN"


def test_compliance_scan_does_not_flag_normal_preference():
    result = scan_memory(Record())
    assert result["risk"] == "none"
    assert result["matched_rules"] == []


def test_compliance_scan_flags_sensitive_content_without_returning_raw_text():
    Record.display_text = "用户持仓 100 股"
    result = scan_memory(Record())
    assert result["risk"] == "high"
    assert "asset_or_holding" in result["matched_rules"]
    assert result["redacted_preview"] == "[已脱敏]"
