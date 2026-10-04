from __future__ import annotations

from agents.finance_agent.pdf_agent.query_rewrite.answer_mismatch.prompt import (
    PDF_ANSWER_MISMATCH_PROMPT,
)
from agents.finance_agent.pdf_agent.query_rewrite.answer_mismatch.node import (
    build_rewrite_contract,
)
from agents.finance_agent.pdf_agent.query_rewrite.hyde.prompt import PDF_HYDE_PROMPT
from agents.finance_agent.pdf_agent.query_rewrite.step_back.prompt import PDF_STEP_BACK_PROMPT
from retrieval.eval.build_eval_sets import rewrite_cases
from scripts.eval_hyde_stepback import (
    ARMS,
    _select_arms,
    anchor_keep,
    build_parser,
    entity_keep,
    forbidden_leakage,
    forbidden_terms,
)


def test_rewrite_prompts_contain_precision_constraints():
    assert "最多补一个结构锚点" in PDF_STEP_BACK_PROMPT
    assert "8 到 24 个词" in PDF_STEP_BACK_PROMPT
    assert "规范别名只能追加" in PDF_HYDE_PROMPT
    assert "指标/事实问题 60～100 字" in PDF_HYDE_PROMPT
    assert "排除项" in PDF_HYDE_PROMPT
    assert "排除项" in PDF_STEP_BACK_PROMPT
    assert "排除项" in PDF_ANSWER_MISMATCH_PROMPT


def test_answer_mismatch_cases_select_dedicated_arms():
    selected = _select_arms(
        {"bucket": "answer_mismatch", "expect_strategy": "answer_mismatch"},
        set(ARMS),
        "expected",
    )
    assert selected == ["original", "answer_mismatch_only", "fused_answer_mismatch"]


def test_control_cases_do_not_select_answer_mismatch():
    selected = _select_arms(
        {"bucket": "control", "expect_strategy": "none"},
        set(ARMS),
        "expected",
    )
    assert "answer_mismatch_only" not in selected
    assert "fused_answer_mismatch" not in selected


def test_forbidden_terms_are_measured_separately_from_entity_keep():
    query = "寒武纪2024年营业收入，不是龙芯中科"
    assert forbidden_terms(query) == ["龙芯中科"]
    assert forbidden_leakage(query, "寒武纪 2024 年营业收入 年度报告") == 0.0
    assert forbidden_leakage(query, "寒武纪与龙芯中科营业收入对比") == 1.0
    assert forbidden_terms("寒武纪2024年营业收入，不要答成龙芯") == ["龙芯"]
    assert forbidden_leakage(
        "宁德时代2025年营业收入，不要用腾讯收入",
        "宁德时代与腾讯营业收入对比",
    ) == 1.0


def test_eval_cli_can_filter_answer_mismatch_bucket_and_case_ids():
    args = build_parser().parse_args(
        [
            "run",
            "--bucket",
            "answer_mismatch",
            "--case-id",
            "mm-001",
            "mm-006",
        ]
    )

    assert args.bucket == "answer_mismatch"
    assert args.case_id == ["mm-001", "mm-006"]


def test_all_mismatch_safe_templates_satisfy_query_contract_metrics():
    cases = [case for case in rewrite_cases() if case["bucket"] == "answer_mismatch"]

    assert len(cases) == 6
    for case in cases:
        contract = build_rewrite_contract(case["query"])
        rewritten = " ".join(
            item for item in (contract.base_query, contract.anchor) if item
        )
        assert entity_keep(
            case["query"], rewritten, case["must_keep_terms"]
        ) == 1.0
        assert anchor_keep(rewritten, case["expected_anchors"]) == 1.0
        assert forbidden_leakage(case["query"], rewritten) == 0.0
