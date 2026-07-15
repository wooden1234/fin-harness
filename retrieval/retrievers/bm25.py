"""BM25 utilities for hybrid retrieval."""

from __future__ import annotations

import re
from typing import Iterable

import jieba
from rank_bm25 import BM25Okapi

_WORD_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._%+-]*")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """Tokenize mixed Chinese/English financial text for BM25."""
    lowered = text.lower()
    tokens: list[str] = []
    tokens.extend(_WORD_RE.findall(lowered))
    for seq in _CJK_RE.findall(lowered):
        tokens.extend(tok.strip() for tok in jieba.cut(seq) if tok.strip())
    return tokens


def bm25_scores(
    query: str,
    documents: Iterable[str],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    corpus_tokens = [tokenize(doc) for doc in documents]
    query_tokens = tokenize(query)
    if not corpus_tokens or not query_tokens:
        return [0.0 for _ in corpus_tokens]

    bm25 = BM25Okapi(corpus_tokens, k1=k1, b=b)
    return [float(score) for score in bm25.get_scores(query_tokens)]
