"""text_to_sql 纠错环防无限循环单元测试。"""

import pytest

from agents.finance_agent.financial_query_agent.text_to_sql.components.retry_guard import (
    resolve_retry_action,
    sql_fingerprint,
)


def _state(**kwargs):
    base = {
        "question": "宁德时代 2024 年营业收入是多少",
        "attempts": 1,
        "max_attempts": 3,
        "seen_sql_hashes": [],
        "last_error_type": "",
        "repeat_error_count": 0,
        "sql": "SELECT 1",
        "sql_params": {"limit": 5},
    }
    base.update(kwargs)
    return base


def test_sql_fingerprint_stable_for_same_sql():
    fp1 = sql_fingerprint("SELECT 1", {"limit": 5})
    fp2 = sql_fingerprint("SELECT 1", {"limit": 5})
    fp3 = sql_fingerprint("SELECT 2", {"limit": 5})

    assert fp1 == fp2
    assert fp1 != fp3


def test_resolve_retry_action_records_seen_sql_before_correction():
    state = _state()
    action = resolve_retry_action(
        state,
        error_type="result_empty",
        error="0 行",
        sql="SELECT 1",
        params={"limit": 5},
        terminal_on_abort="execution_error_output",
    )

    assert action["next_step"] == "correct_sql"
    assert len(action["seen_sql_hashes"]) == 1
    assert action["repeat_error_count"] == 1
    assert action["last_error_type"] == "result_empty"


def test_resolve_retry_action_aborts_on_duplicate_sql():
    fingerprint = sql_fingerprint("SELECT 1", {"limit": 5})
    state = _state(seen_sql_hashes=[fingerprint], repeat_error_count=1, last_error_type="result_empty")

    action = resolve_retry_action(
        state,
        error_type="result_empty",
        error="0 行",
        sql="SELECT 1",
        params={"limit": 5},
        terminal_on_abort="execution_error_output",
    )

    assert action["next_step"] == "execution_error_output"
    assert "重复 SQL" in action["validation_error"]


def test_resolve_retry_action_aborts_on_repeat_same_error(monkeypatch):
    monkeypatch.setattr(
        "agents.finance_agent.financial_query_agent.text_to_sql.components.retry_guard.settings.FINANCIAL_SQL_MAX_REPEAT_SAME_ERROR",
        2,
    )
    state = _state(repeat_error_count=1, last_error_type="result_empty")

    action = resolve_retry_action(
        state,
        error_type="result_empty",
        error="0 行",
        sql="SELECT 2",
        params={"limit": 5},
        terminal_on_abort="execution_error_output",
    )

    assert action["next_step"] == "execution_error_output"
    assert action["repeat_error_count"] == 2
    assert "连续相同错误" in action["validation_error"]


def test_resolve_retry_action_aborts_when_attempts_exhausted():
    state = _state(attempts=3, max_attempts=3)

    action = resolve_retry_action(
        state,
        error_type="runtime",
        error="db down",
        sql="SELECT 1",
        params={"limit": 5},
        terminal_on_abort="execution_error_output",
    )

    assert action["next_step"] == "execution_error_output"
