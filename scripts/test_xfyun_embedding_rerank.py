"""讯飞 MaaS Embedding / Rerank API 连通性测试。

用法:
    python scripts/test_xfyun_embedding_rerank.py
    python scripts/test_xfyun_embedding_rerank.py --embedding-only
    python scripts/test_xfyun_embedding_rerank.py --rerank-only
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT / "app" / "backend"
for path in (str(BACKEND_DIR), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)
load_dotenv(ROOT / ".env", override=True)

from app.core.config import settings  # noqa: E402
from retrieval.clients.embeddings import (  # noqa: E402
    _resolve_embedding_credentials,
    embedding_provider,
    get_embed_model,
)
from retrieval.clients.rerank_client import (  # noqa: E402
    rerank_documents,
    rerank_enabled,
    rerank_provider,
)


def _l2_norm(values: list[float]) -> float:
    return math.sqrt(sum(v * v for v in values))


def _validate_api_key(api_key: str) -> None:
    if ":" not in api_key:
        raise RuntimeError("API Key 格式应为 app_id:api_secret")
    secret = api_key.split(":", 1)[1]
    if len(secret) > 36:
        print(
            "警告: API Key secret 部分过长，可能是复制时把换行后的内容也拼进来了。"
            "请只保留控制台第一行（通常 secret 为 32 字符）。"
        )


def test_embedding() -> None:
    provider = embedding_provider()
    api_key, api_base, model = _resolve_embedding_credentials()
    dim = settings.EMBEDDING_DIM
    _validate_api_key(api_key)

    print("=== Embedding ===")
    print(f"provider : {provider}")
    print(f"model    : {model}")
    print(f"base_url : {api_base}")
    print(f"dim      : {dim}")

    text = "什么是 T+1 交易制度？"
    url = f"{api_base.rstrip('/')}/embeddings"
    response = httpx.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "input": text,
            "dimensions": dim,
        },
        timeout=60.0,
    )
    print(f"HTTP     : {response.status_code}")
    if response.status_code >= 400:
        print(f"body     : {response.text[:500]}")
        response.raise_for_status()

    payload = response.json()
    data = payload.get("data") or []
    if not data:
        raise RuntimeError(f"embedding 响应缺少 data 字段: {payload}")

    vector = data[0].get("embedding") or []
    if not isinstance(vector, list) or not vector:
        raise RuntimeError(f"embedding 向量为空: {payload}")

    norm = _l2_norm(vector)
    print(f"vector   : len={len(vector)}, l2_norm={norm:.4f}, head={vector[:3]}")
    if len(vector) != dim:
        raise RuntimeError(f"向量维度 {len(vector)} != EMBEDDING_DIM {dim}")

    # 走项目封装再测一次
    get_embed_model.cache_clear()
    embed_model = get_embed_model()
    wrapped = embed_model.get_text_embedding(text)
    print(f"wrapper  : len={len(wrapped)}, l2_norm={_l2_norm(wrapped):.4f}")
    print("embedding: OK")


def test_rerank() -> None:
    print("\n=== Rerank ===")
    print(f"enabled  : {rerank_enabled()}")
    print(f"provider : {rerank_provider()}")
    print(f"model    : {settings.RERANK_MODEL}")
    print(f"base_url : {settings.RERANK_BASE_URL}")

    query = "T+1 交易制度是什么？"
    documents = [
        "T+1 是指当日买入的股票，下一个交易日才能卖出。",
        "A 股实行 T+1 交收制度，投资者当天买入的证券需次日方可卖出。",
        "Python 是一种高级编程语言，广泛用于数据分析。",
    ]

    results = rerank_documents(query=query, documents=documents, top_n=2)
    if not results:
        raise RuntimeError("rerank 返回空结果")

    print("results  :")
    for item in results:
        doc = documents[item.index]
        preview = doc[:40] + ("..." if len(doc) > 40 else "")
        print(f"  idx={item.index} score={item.score:.4f} doc={preview!r}")

    best = results[0]
    if best.index not in {0, 1}:
        raise RuntimeError(
            f"rerank 排序异常，最高分 idx={best.index}，预期应为 0 或 1"
        )
    print("rerank   : OK")


def main() -> int:
    parser = argparse.ArgumentParser(description="测试讯飞 MaaS embedding / rerank API")
    parser.add_argument("--embedding-only", action="store_true")
    parser.add_argument("--rerank-only", action="store_true")
    args = parser.parse_args()

    run_embedding = not args.rerank_only
    run_rerank = not args.embedding_only

    failed = False
    if run_embedding:
        try:
            test_embedding()
        except Exception as exc:
            failed = True
            print(f"embedding: FAIL — {exc}")

    if run_rerank:
        try:
            test_rerank()
        except Exception as exc:
            failed = True
            print(f"rerank   : FAIL — {exc}")

    if failed:
        print("\n至少一项 API 测试失败，请检查 .env 中的密钥、model、base_url。")
        return 1

    print("\n全部 API 测试通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
