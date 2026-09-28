import pytest

from capabilities.knowledge import catalog_pdf, search_faq, search_pdf, search_web


@pytest.mark.asyncio
async def test_search_faq_uses_local_retriever_and_stamps_evidence(monkeypatch):
    seen = {}

    async def fake_search(query, *, domain, top_k=3):
        seen["query"] = query
        seen["domain"] = domain
        seen["top_k"] = top_k
        return []

    monkeypatch.setattr("tools.knowledge.search_faq_knowledge", fake_search)
    result = await search_faq(
        "什么是 T+1 交易制度？",
        domain="capital_market",
        top_k=3,
        research_question_id="q1",
    )
    assert seen == {
        "query": "什么是 T+1 交易制度？",
        "domain": "capital_market",
        "top_k": 3,
    }
    assert result["ok"] is True
    assert result["source_tool"] == "knowledge.faq.search"
    assert str(result["evidence_id"]).startswith("knowledge.faq.search:")
    assert result["evidence"] == []


@pytest.mark.asyncio
async def test_pdf_and_web_adapters_stamp_their_tool_ids(monkeypatch):
    monkeypatch.setattr(
        "tools.knowledge.catalog_pdf_documents",
        lambda **_kwargs: [{"doc_id": "doc-1"}],
    )

    async def fake_pdf(*_args, **_kwargs):
        return []

    async def fake_web(query, *, scope="allowlist"):
        return {"answer": query, "results": [], "configured": True}

    monkeypatch.setattr("tools.knowledge.search_pdf_knowledge", fake_pdf)
    monkeypatch.setattr("tools.web_search.fetch_web_search", fake_web)

    catalog = await catalog_pdf(query="寒武纪 年报", categories=["annual_reports"])
    pdf = await search_pdf(
        query="研发投入",
        categories=["annual_reports"],
        doc_ids=["doc-1"],
    )
    web = await search_web("证监会 程序化交易")

    assert catalog["source_tool"] == "knowledge.pdf.catalog"
    assert catalog["count"] == 1
    assert pdf["source_tool"] == "knowledge.pdf.search"
    assert pdf["evidence"] == []
    assert web["source_tool"] == "web.search"
    assert web["answer"] == "证监会 程序化交易"
