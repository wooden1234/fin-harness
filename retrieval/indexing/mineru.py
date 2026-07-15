"""MinerU 精准解析：本地 PDF 批量上传 → 轮询 → 下载 zip 结果。"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import httpx
from loguru import logger

from app.core.config import settings

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT_DIR / "knowledge" / "raw"
PARSED_DIR = ROOT_DIR / "knowledge" / "parsed"
MANIFEST_PATH = PARSED_DIR / "extract_manifest.json"

DEFAULT_CATEGORIES = (
    "industry_whitepapers",
    "research_reports",
    "policy",
    "macro_research",
)

BATCH_SIZE = 50
POLL_INTERVAL_SEC = 5
POLL_TIMEOUT_SEC = 1800
MAX_PAGES_PER_JOB = 200
TERMINAL_STATES = frozenset({"done", "failed"})


@dataclass
class PdfJob:
    pdf_path: Path
    source_path: Path
    category: str
    data_id: str
    is_ocr: bool
    page_range: tuple[int, int] | None = None

    @property
    def zip_path(self) -> Path:
        stem = self.source_path.stem
        if self.page_range:
            start, end = self.page_range
            name = f"{stem}_p{start:03d}-{end:03d}.zip"
        else:
            name = f"{stem}.zip"
        return PARSED_DIR / self.category / name

    @property
    def upload_name(self) -> str:
        if self.page_range:
            start, end = self.page_range
            return f"{self.source_path.stem}_p{start:03d}-{end:03d}.pdf"
        return self.source_path.name


@dataclass
class ExtractRecord:
    pdf: str
    category: str
    data_id: str
    state: str
    zip_path: str | None = None
    batch_id: str | None = None
    full_zip_url: str | None = None
    err_msg: str | None = None
    page_range: str | None = None
    updated_at: str | None = None


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

    def submit_local_batch(self, jobs: list[PdfJob]) -> tuple[str, list[str]]:
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
            content = job.pdf_path.read_bytes()
            upload_resp = self._client.put(upload_url, content=content)
            if upload_resp.status_code != 200:
                raise RuntimeError(
                    f"上传失败 {job.pdf_path.name}: HTTP {upload_resp.status_code}"
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


def _resolve_token() -> str:
    token = settings.MINERU_API_KEY or settings.MINERU_TOKEN
    if not token:
        raise RuntimeError("未配置 MINERU_API_KEY 或 MINERU_TOKEN")
    return token


def _sanitize_data_id(stem: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]", "-", stem)
    return value[:128] or "pdf"


def _max_pages_per_job() -> int:
    return settings.MINERU_MAX_PAGES or MAX_PAGES_PER_JOB


def get_pdf_page_count(pdf_path: Path) -> int:
    from pypdf import PdfReader

    return len(PdfReader(str(pdf_path)).pages)


def split_pdf_ranges(total_pages: int, max_pages: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = 1
    while start <= total_pages:
        end = min(start + max_pages - 1, total_pages)
        ranges.append((start, end))
        start = end + 1
    return ranges


def split_pdf_file(pdf_path: Path, start: int, end: int, dest: Path) -> Path:
    from pypdf import PdfReader, PdfWriter

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest

    reader = PdfReader(str(pdf_path))
    writer = PdfWriter()
    for page_idx in range(start - 1, end):
        writer.add_page(reader.pages[page_idx])
    with dest.open("wb") as f:
        writer.write(f)
    logger.info(f"已拆分 PDF: {pdf_path.name} -> {dest.name} ({start}-{end} 页)")
    return dest


def expand_jobs_with_page_splits(jobs: list[PdfJob]) -> list[PdfJob]:
    """超过 MinerU 页数上限时，本地拆分为多个 PDF 再分别提交。"""
    max_pages = _max_pages_per_job()
    split_root = PARSED_DIR / "_splits"
    expanded: list[PdfJob] = []

    for job in jobs:
        total_pages = get_pdf_page_count(job.source_path)
        if total_pages <= max_pages:
            expanded.append(job)
            continue

        logger.info(
            f"{job.source_path.name}: {total_pages} 页，超过 {max_pages} 页上限，自动拆分"
        )
        for start, end in split_pdf_ranges(total_pages, max_pages):
            split_path = (
                split_root
                / job.category
                / f"{job.source_path.stem}_p{start:03d}-{end:03d}.pdf"
            )
            split_pdf_file(job.source_path, start, end, split_path)
            expanded.append(
                PdfJob(
                    pdf_path=split_path,
                    source_path=job.source_path,
                    category=job.category,
                    data_id=_sanitize_data_id(f"{job.source_path.stem}-p{start}-{end}"),
                    is_ocr=job.is_ocr,
                    page_range=(start, end),
                )
            )
    return expanded


def discover_pdf_jobs(
    categories: Iterable[str] = DEFAULT_CATEGORIES,
    *,
    raw_dir: Path = RAW_DIR,
) -> list[PdfJob]:
    jobs: list[PdfJob] = []
    for category in categories:
        category_dir = raw_dir / category
        if not category_dir.is_dir():
            logger.warning(f"目录不存在，跳过: {category_dir}")
            continue
        is_ocr = category == "policy"
        for pdf_path in sorted(category_dir.glob("*.pdf")):
            jobs.append(
                PdfJob(
                    pdf_path=pdf_path,
                    source_path=pdf_path,
                    category=category,
                    data_id=_sanitize_data_id(pdf_path.stem),
                    is_ocr=is_ocr,
                )
            )
    return expand_jobs_with_page_splits(jobs)


def _load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        return {"files": {}}
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _save_manifest(manifest: dict[str, Any]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _record_key(job: PdfJob) -> str:
    key = f"{job.category}/{job.source_path.name}"
    if job.page_range:
        start, end = job.page_range
        key += f"#p{start:03d}-{end:03d}"
    return key


def _page_range_label(job: PdfJob) -> str | None:
    if not job.page_range:
        return None
    start, end = job.page_range
    return f"{start:03d}-{end:03d}"


def _update_manifest(manifest: dict[str, Any], record: ExtractRecord) -> None:
    manifest.setdefault("files", {})[record.pdf] = asdict(record)
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    _save_manifest(manifest)


def extract_pdfs(
    categories: Iterable[str] = DEFAULT_CATEGORIES,
    *,
    raw_dir: Path = RAW_DIR,
    force: bool = False,
    poll_interval: float = POLL_INTERVAL_SEC,
    poll_timeout: float = POLL_TIMEOUT_SEC,
) -> list[ExtractRecord]:
    """扫描指定目录 PDF，调用 MinerU 解析并保存 zip。"""
    token = _resolve_token()
    base_url = settings.MINERU_BASE_URL
    model_version = settings.MINERU_MODEL_VERSION

    all_jobs = discover_pdf_jobs(categories, raw_dir=raw_dir)
    if not all_jobs:
        logger.warning("未找到任何 PDF 文件")
        return []

    manifest = _load_manifest()
    pending_jobs: list[PdfJob] = []
    records: list[ExtractRecord] = []

    for job in all_jobs:
        key = _record_key(job)
        if job.zip_path.exists() and not force:
            logger.info(f"已存在 zip，跳过: {job.zip_path}")
            cached = manifest.get("files", {}).get(key)
            if cached:
                records.append(ExtractRecord(**cached))
            else:
                records.append(
                    ExtractRecord(
                        pdf=key,
                        category=job.category,
                        data_id=job.data_id,
                        state="done",
                        zip_path=str(job.zip_path.relative_to(ROOT_DIR)),
                        updated_at=datetime.now(timezone.utc).isoformat(),
                    )
                )
            continue
        pending_jobs.append(job)

    if not pending_jobs:
        logger.info("所有 PDF 均已解析，无需提交新任务")
        return records

    with MinerUClient(token, base_url=base_url, model_version=model_version) as client:
        for offset in range(0, len(pending_jobs), BATCH_SIZE):
            batch_jobs = pending_jobs[offset : offset + BATCH_SIZE]
            logger.info(
                f"提交批次 {offset // BATCH_SIZE + 1}: "
                f"{len(batch_jobs)} 个文件"
            )
            batch_id, _ = client.submit_local_batch(batch_jobs)
            results = client.poll_batch_results(
                batch_id,
                poll_interval=poll_interval,
                timeout=poll_timeout,
            )

            by_name = {item.get("file_name"): item for item in results}
            for job in batch_jobs:
                key = _record_key(job)
                item = by_name.get(job.upload_name, {})
                state = item.get("state", "unknown")
                now = datetime.now(timezone.utc).isoformat()
                record = ExtractRecord(
                    pdf=key,
                    category=job.category,
                    data_id=job.data_id,
                    state=state,
                    batch_id=batch_id,
                    full_zip_url=item.get("full_zip_url"),
                    err_msg=item.get("err_msg") or None,
                    page_range=_page_range_label(job),
                    updated_at=now,
                )

                if state == "done" and record.full_zip_url:
                    client.download_zip(record.full_zip_url, job.zip_path)
                    record.zip_path = str(job.zip_path.relative_to(ROOT_DIR))
                elif state == "failed":
                    logger.error(
                        f"解析失败 {job.upload_name}: {record.err_msg or 'unknown'}"
                    )

                _update_manifest(manifest, record)
                records.append(record)

    done = sum(1 for r in records if r.state == "done")
    failed = sum(1 for r in records if r.state == "failed")
    logger.info(f"完成: success={done}, failed={failed}, total={len(records)}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MinerU 精准解析：提取 knowledge/raw 下 PDF 并保存 zip"
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(DEFAULT_CATEGORIES),
        help="要处理的子目录（默认四个 PDF 类别）",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help="PDF 根目录，默认 knowledge/raw",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="忽略已有 zip，重新提交解析",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=POLL_INTERVAL_SEC,
        help="轮询间隔（秒）",
    )
    parser.add_argument(
        "--poll-timeout",
        type=float,
        default=POLL_TIMEOUT_SEC,
        help="单批次最长等待时间（秒）",
    )
    args = parser.parse_args()

    extract_pdfs(
        args.categories,
        raw_dir=args.raw_dir,
        force=args.force,
        poll_interval=args.poll_interval,
        poll_timeout=args.poll_timeout,
    )


if __name__ == "__main__":
    main()
