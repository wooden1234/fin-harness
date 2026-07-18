"""MinerU Open API 客户端。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

POLL_INTERVAL_SEC = 5
POLL_TIMEOUT_SEC = 1800
TERMINAL_STATES = frozenset({"done", "failed"})


class MinerUClient:
    """MinerU Open API v4 客户端（本地文件批量上传）。"""

    def __init__(
        self,
        token: str,
        base_url: str = "https://mineru.net/api/v4",
        model_version: str = "vlm",
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_version = model_version
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "*/*",
        }
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> MinerUClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        resp = self._client.post(
            f"{self.base_url}{path}",
            headers=self._headers,
            json=payload,
        )
        resp.raise_for_status()
        body = resp.json()
        if body.get("code") != 0:
            raise RuntimeError(f"MinerU API 错误: {body.get('msg')} (code={body.get('code')})")
        return body["data"]

    def _get(self, path: str) -> dict[str, Any]:
        resp = self._client.get(f"{self.base_url}{path}", headers=self._headers)
        resp.raise_for_status()
        body = resp.json()
        if body.get("code") != 0:
            raise RuntimeError(f"MinerU API 错误: {body.get('msg')} (code={body.get('code')})")
        return body["data"]

    def submit_local_batch(self, jobs: list[Any]) -> tuple[str, list[str]]:
        """申请上传链接并 PUT 上传本地 PDF，返回 (batch_id, data_ids)。"""
        payload: dict[str, Any] = {
            "files": [
                {
                    "name": job.upload_name,
                    "data_id": job.data_id,
                    "is_ocr": job.is_ocr,
                }
                for job in jobs
            ],
            "model_version": self.model_version,
            "enable_formula": True,
            "enable_table": True,
            "language": "ch",
        }
        data = self._post("/file-urls/batch", payload)
        batch_id = data["batch_id"]
        upload_urls: list[str] = data["file_urls"]

        if len(upload_urls) != len(jobs):
            raise RuntimeError(
                f"上传链接数量不匹配: jobs={len(jobs)}, urls={len(upload_urls)}"
            )

        for job, upload_url in zip(jobs, upload_urls):
            content = Path(job.pdf_path).read_bytes()
            upload_resp = self._client.put(upload_url, content=content)
            if upload_resp.status_code != 200:
                raise RuntimeError(
                    f"上传失败 {Path(job.pdf_path).name}: HTTP {upload_resp.status_code}"
                )
            logger.info(f"已上传: {job.category}/{job.upload_name}")

        return batch_id, [job.data_id for job in jobs]

    def poll_batch_results(
        self,
        batch_id: str,
        *,
        poll_interval: float = POLL_INTERVAL_SEC,
        timeout: float = POLL_TIMEOUT_SEC,
    ) -> list[dict[str, Any]]:
        """轮询批量任务，直到全部进入 done / failed。"""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            data = self._get(f"/extract-results/batch/{batch_id}")
            results: list[dict[str, Any]] = data.get("extract_result") or []
            if not results:
                time.sleep(poll_interval)
                continue

            pending = [r for r in results if r.get("state") not in TERMINAL_STATES]
            for item in results:
                state = item.get("state", "unknown")
                name = item.get("file_name", "?")
                if state == "running":
                    progress = item.get("extract_progress") or {}
                    logger.info(
                        f"解析中 {name}: {progress.get('extracted_pages', '?')}/"
                        f"{progress.get('total_pages', '?')} 页"
                    )
                elif state in {"pending", "waiting-file", "uploading", "converting"}:
                    logger.info(f"排队/处理中 {name}: {state}")

            if not pending:
                return results

            time.sleep(poll_interval)

        raise TimeoutError(f"批量任务超时 batch_id={batch_id}")

    def download_zip(self, url: str, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with self._client.stream("GET", url, follow_redirects=True) as resp:
            resp.raise_for_status()
            with dest.open("wb") as f:
                for chunk in resp.iter_bytes():
                    f.write(chunk)
        logger.info(f"已保存 zip: {dest}")
        return dest
