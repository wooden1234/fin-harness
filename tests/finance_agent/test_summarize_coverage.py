"""Summarize 节点：coverage 感知汇总。"""

from __future__ import annotations

import pytest

from app.agents.finance_agent.summarize.node import _pick_best_per_subtask


def test_pick_best_prefers_web_over_faq_uncovered():
    results = [
        {
            "sub_task_id": "t1",
            "type": "faq",
            "coverage": "uncovered",
            "context": "（未找到相关知识库条目）",
        },
        {
            "sub_task_id": "t1",
            "type": "web_search",
            "coverage": "covered",
            "context": "[联网搜索] 信用卡年费说明...",
        },
    ]
    picked = _pick_best_per_subtask(results)
    assert len(picked) == 1
    assert picked[0]["type"] == "web_search"
