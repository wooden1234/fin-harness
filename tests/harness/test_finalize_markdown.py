from harness.finalization.submit import finalize_markdown, strip_source_attribution


def test_strip_source_attribution_removes_iwencai_footer():
    text = (
        "沪深A股总市值超300亿公司筛选结果\n\n"
        "符合条件总数：646 只。\n\n"
        "数据来源：同花顺问财（evidence_id: iwencai.industry.query:98c742ee4da3、"
        "iwencai.industry.query:886ccb99c16d）"
    )
    cleaned = strip_source_attribution(text)
    assert "数据来源" not in cleaned
    assert "evidence_id" not in cleaned
    assert "646 只" in cleaned


def test_finalize_markdown_drops_source_line_and_keeps_answer():
    published = finalize_markdown(
        "贵州茅台2024年营业收入 1741.44 亿。\n数据来源：同花顺问财"
    )
    assert published == "贵州茅台2024年营业收入 1741.44 亿。"
