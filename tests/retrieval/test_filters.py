from retrieval.core.filters import (
    FILTER_RULES,
    KNOWLEDGE_BASE_PROFILES,
    MatchMode,
    has_strict_filters,
    infer_pdf_field_filters,
    infer_pdf_metadata_filters,
    filters_for_category,
    metadata_matches,
    routable_kb_ids,
    rules_for_categories,
    supported_filter_keys,
)


def test_kb_profiles_exist_for_pdf_collections():
    assert {
        "annual_reports",
        "research_reports",
        "industry_whitepapers",
        "policy",
        "macro_research",
    } <= set(KNOWLEDGE_BASE_PROFILES)


def test_routable_kb_ids_from_contract():
    assert "faq" not in routable_kb_ids()
    assert "annual_reports" in routable_kb_ids()


def test_annual_reports_supports_year_ticker_company():
    keys = supported_filter_keys(["annual_reports"])
    assert {"year", "ticker", "company", "doc_id", "category"} <= keys
    assert "issuer" not in keys


def test_policy_supports_issuer_not_ticker():
    keys = supported_filter_keys(["policy"])
    assert {"year", "issuer", "doc_id"} <= keys
    assert "ticker" not in keys


def test_research_reports_do_not_support_ticker_hard_filter():
    keys = supported_filter_keys(["research_reports"])
    assert {"year", "issuer", "doc_id"} <= keys
    assert "ticker" not in keys


def test_filters_for_category_drops_unsupported_ticker():
    filters = filters_for_category(
        {"category": ["annual_reports", "research_reports"], "ticker": "CATL", "year": 2026},
        "research_reports",
    )
    assert filters == {"category": "research_reports", "year": 2026}


def test_rules_for_categories_scopes_exact_keys():
    annual = {r.filter_key for r in rules_for_categories(["annual_reports"]) if r.mode is MatchMode.EXACT}
    policy = {r.filter_key for r in rules_for_categories(["policy"]) if r.mode is MatchMode.EXACT}
    assert "ticker" in annual
    assert "ticker" not in policy


def test_filter_rules_union_covers_common_keys():
    exact_keys = {rule.filter_key for rule in FILTER_RULES if rule.mode is MatchMode.EXACT}
    fuzzy_keys = {rule.filter_key for rule in FILTER_RULES if rule.mode is MatchMode.FUZZY}
    assert {"category", "year", "ticker", "doc_id"} <= exact_keys
    assert {"company", "source", "issuer"} <= fuzzy_keys


def test_infer_with_explicit_kb():
    filters = infer_pdf_metadata_filters(
        "宁德时代2025年报营业收入",
        knowledge_bases=["annual_reports"],
    )
    assert filters["year"] == 2025
    assert filters["category"] == "annual_reports"
    assert "company" not in filters
    assert has_strict_filters(filters)


def test_infer_without_kb_skips_category_and_year():
    filters = infer_pdf_metadata_filters("宁德时代2025年报营业收入")
    assert "category" not in filters
    assert "year" not in filters


def test_infer_respects_kb_contract():
    filters = infer_pdf_metadata_filters("2024产业政策 300750", knowledge_bases=["policy"])
    assert filters["category"] == "policy"
    assert "year" not in filters
    assert "ticker" not in filters


def test_field_filters_do_not_set_category():
    fields = infer_pdf_field_filters(
        "宁德时代2024年年报披露的营业收入",
        knowledge_bases=["annual_reports"],
    )
    assert "category" not in fields
    assert fields["year"] == 2024
    assert "ticker" not in fields


def test_field_filters_empty_when_no_match():
    fields = infer_pdf_field_filters("白皮书讲了什么", knowledge_bases=["industry_whitepapers"])
    assert fields == {}


def test_field_filters_no_year_without_routed_kb():
    fields = infer_pdf_field_filters("宁德时代2024年年报披露的营业收入")
    assert "year" not in fields


def test_field_filters_skip_planning_year_for_policy():
    fields = infer_pdf_field_filters("2024规划要点", knowledge_bases=["policy"])
    assert "year" not in fields


def test_field_filters_no_year_on_five_year_plan_without_digits():
    fields = infer_pdf_field_filters("十五五规划产业目标", knowledge_bases=["policy"])
    assert "year" not in fields


def test_has_strict_filters_false_without_exact_keys():
    assert not has_strict_filters({"company": "CATL"})
    assert not has_strict_filters(None)


def test_metadata_rejects_wrong_fiscal_year():
    metadata = {"fiscal_year": 2024, "title": "宁德时代2025年度报告"}
    assert not metadata_matches(metadata, {"year": 2025})


def test_metadata_accepts_matching_fiscal_year():
    metadata = {"fiscal_year": 2025, "title": "宁德时代2024年度报告"}
    assert metadata_matches(metadata, {"year": 2025})


def test_metadata_rejects_title_year_without_fiscal_year():
    metadata = {"title": "宁德时代2025年度报告", "doc_id": "PDF-AR-2025-CATL"}
    assert not metadata_matches(metadata, {"year": 2025})


def test_metadata_accepts_indexed_year_field():
    metadata = {"year": 2025, "title": "宁德时代2024年度报告"}
    assert metadata_matches(metadata, {"year": 2025})


def test_metadata_accepts_effective_date_year():
    metadata = {"effective_date": "2025-03-31", "title": "宁德时代2024年度报告"}
    assert metadata_matches(metadata, {"year": 2025})


def test_metadata_doc_id_is_exact_not_substring():
    metadata = {"doc_id": "PDF-AR-2024-CATL"}
    assert not metadata_matches(metadata, {"doc_id": "PDF-AR-2024"})
    assert metadata_matches(metadata, {"doc_id": "PDF-AR-2024-CATL"})


def test_metadata_ticker_exact():
    metadata = {"ticker": "300750", "title": "某公司2024年度报告"}
    assert metadata_matches(metadata, {"ticker": "300750"})
    assert not metadata_matches(metadata, {"ticker": "688256"})


def test_company_matches_literal_without_alias_table():
    metadata = {"company": "CATL", "title": "宁德时代2024年度报告", "ticker": "300750"}
    assert metadata_matches(metadata, {"company": "CATL"})
    assert metadata_matches(metadata, {"company": "宁德时代"})
    assert not metadata_matches(metadata, {"company": "寒武纪"})
