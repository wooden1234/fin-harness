"""本地 RPC demo：财务事实查询跑在另一个进程里。

Agent 进程只发起一次调用。服务进程执行查询并返回原来的字典。
evidence_id 仍由调用方在本进程盖上，不放进 RPC 服务。

运行：

    python scripts/demo_finance_rpc.py
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


METHOD = "finance.fact.lookup"
TOOL_ID = "knowledge.fact.lookup"


def fetch_financial_fact(arguments: dict[str, Any]) -> dict[str, Any]:
    """服务进程里的查询。demo 用固定结果代替数据库。"""
    companies = list(arguments.get("companies") or [])
    metrics = list(arguments.get("metrics") or [])
    if not companies or not metrics:
        return {
            "ok": False,
            "error": "fact_query_requires_clarification",
            "message": "结构化查询缺少明确公司、指标或期间，不能安全执行。",
        }
    company = str(companies[0])
    metric = str(metrics[0])
    return {
        "ok": True,
        "route": "local_fact",
        "answer": f"{company}{metric}为 100 亿元。",
        "facts": [
            {
                "company": company,
                "period_year": 2024,
                "metric": metric,
                "value": "100",
                "unit": "亿元",
                "currency": "CNY",
            }
        ],
        "citations": [],
    }


def stamp_evidence(payload: dict[str, Any]) -> dict[str, Any]:
    """调用方本进程盖证据编号。服务端返回值里没有这一项。"""
    from uuid import uuid4

    out = dict(payload)
    out.setdefault("ok", True)
    out.setdefault("evidence_id", f"{TOOL_ID}:{uuid4().hex[:12]}")
    out.setdefault("source_tool", TOOL_ID)
    return out


class RpcHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        request_id = body.get("id")
        if body.get("method") != METHOD:
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": "method not found"},
            }
        else:
            result = fetch_financial_fact(dict(body.get("params") or {}))
            response = {"jsonrpc": "2.0", "id": request_id, "result": result}
        encoded = json.dumps(response, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: Any) -> None:
        return


def serve(port: int) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", port), RpcHandler)
    print(f"RPC 服务已监听 127.0.0.1:{port}，方法 {METHOD}", flush=True)
    server.serve_forever()


def lookup_fact(port: int, arguments: dict[str, Any]) -> dict[str, Any]:
    """Agent 侧的 capabilities.finance.lookup_fact。"""
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": METHOD,
        "params": arguments,
    }
    raw = json.dumps(request, ensure_ascii=False).encode("utf-8")
    http_request = urllib.request.Request(
        f"http://127.0.0.1:{port}/rpc",
        data=raw,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(http_request, timeout=3) as response:
        body = json.loads(response.read().decode("utf-8"))
    if "error" in body:
        return {"ok": False, "error": "tool_execution_failed", "message": body["error"]["message"]}
    result = body["result"]
    if not isinstance(result, dict):
        return {"ok": False, "error": "tool_contract_error"}
    return stamp_evidence(result)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        serve(int(sys.argv[2]))
        return

    port = free_port()
    server = subprocess.Popen(
        [sys.executable, __file__, "serve", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        ready = server.stdout.readline() if server.stdout is not None else ""
        print(ready.strip())
        arguments = {
            "question": "腾讯2024年营收是多少",
            "companies": ["腾讯"],
            "metrics": ["营收"],
            "years": [2024],
            "operation": "lookup",
            "top_k": 5,
        }
        print("调用方发出：")
        print(json.dumps({"method": METHOD, "params": arguments}, ensure_ascii=False, indent=2))
        result = lookup_fact(port, arguments)
        print("调用方收到并盖上 evidence_id：")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except urllib.error.URLError as exc:
        print(f"RPC 调用失败：{exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    finally:
        server.terminate()
        server.wait(timeout=5)


if __name__ == "__main__":
    main()
