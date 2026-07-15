import asyncio
from functools import lru_cache
from typing import Literal

from fastapi import APIRouter, Depends, Query
from app.core.security import get_current_user
from app.models.user import User
from retrieval import get_retriever
from retrieval.core.filters import compact_filters
from app.schemas.rag import RagHitItem, RagSearchResponse

PdfCategory = Literal[
    "macro_research",
    "annual_reports",
    "research_reports",
    "industry_whitepapers",
    "policy",
]

router = APIRouter(prefix="/rag", tags=["rag"])


@lru_cache(maxsize=16)
def _get_retriever(categories_key: str, hybrid: bool):
    """进程内复用；categories_key 为逗号分隔的 category 或 '__all__'。"""
    if categories_key == "__all__":
        return get_retriever(top_k=3, similarity_threshold=None, hybrid=hybrid)
    categories = [c.strip() for c in categories_key.split(",") if c.strip()]
    return get_retriever(
        categories=categories,
        top_k=3,
        similarity_threshold=None,
        hybrid=hybrid,
    )


@router.post("/search", response_model=RagSearchResponse)
async def search_rag(
    query: str = Query(..., description="搜索查询"),
    categories: list[PdfCategory | Literal["faq"]] | None = Query(
        None,
        description="限定检索集合；不传则搜索全部（FAQ + 五类 PDF）",
    ),
    company: str | None = Query(None, description="公司过滤，如 CATL / 宁德时代 / 688256"),
    year: int | None = Query(None, description="年份过滤，如 2024"),
    source: str | None = Query(None, description="来源文件名/标题/doc_id 过滤"),
    hybrid: bool = Query(True, description="是否启用向量 + BM25 混合检索"),
    current_user: User = Depends(get_current_user),
):
    key = "__all__" if not categories else ",".join(sorted(set(categories)))
    metadata_filters = compact_filters(
        {
            "category": list(categories) if categories else None,
            "company": company,
            "year": year,
            "source": source,
        }
    )
    retriever = _get_retriever(key, hybrid)
    hits = await asyncio.to_thread(
        retriever.search,
        query,
        top_k=3,
        metadata_filters=metadata_filters,
    )
    hits = [
        RagHitItem(
            text=h.text,
            score=h.score,
            metadata=h.metadata,
            node_id=h.node_id,
        )
        for h in hits
    ]
    return RagSearchResponse(query=query, top_k=len(hits), hits=hits)
