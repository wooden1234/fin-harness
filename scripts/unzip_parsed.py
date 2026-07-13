"""解压 knowledge/parsed 下 MinerU 输出的 zip 包。

用法:
    python scripts/unzip_parsed.py
    python scripts/unzip_parsed.py --categories annual_reports industry_whitepapers
    python scripts/unzip_parsed.py --force
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.core.logger import get_logger

logger = get_logger(service="unzip_parsed")

DEFAULT_PARSED_DIR = ROOT_DIR / "knowledge" / "parsed"
SKIP_DIRS = {"_splits"}


def discover_zip_files(
    parsed_dir: Path,
    categories: list[str] | None = None,
) -> list[Path]:
    roots: list[Path]
    if categories:
        roots = [parsed_dir / c for c in categories]
    else:
        roots = [parsed_dir]

    zips: list[Path] = []
    for root in roots:
        if not root.is_dir():
            logger.warning(f"目录不存在，跳过: {root}")
            continue
        for zip_path in sorted(root.rglob("*.zip")):
            if any(part in SKIP_DIRS for part in zip_path.parts):
                continue
            zips.append(zip_path)
    return zips


def extract_zip(zip_path: Path, *, force: bool = False) -> Path:
    dest_dir = zip_path.with_suffix("")
    if dest_dir.exists() and not force:
        if any(dest_dir.iterdir()):
            logger.info(f"已解压，跳过: {dest_dir.relative_to(ROOT_DIR)}")
            return dest_dir

    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)

    rel = dest_dir.relative_to(ROOT_DIR)
    logger.info(f"已解压: {zip_path.name} -> {rel}")
    return dest_dir


def unzip_parsed(
    parsed_dir: Path = DEFAULT_PARSED_DIR,
    *,
    categories: list[str] | None = None,
    force: bool = False,
) -> list[Path]:
    zip_files = discover_zip_files(parsed_dir, categories)
    if not zip_files:
        logger.warning(f"未找到 zip 文件: {parsed_dir}")
        return []

    extracted: list[Path] = []
    for zip_path in zip_files:
        extracted.append(extract_zip(zip_path, force=force))

    logger.info(f"完成: 处理 {len(zip_files)} 个 zip")
    return extracted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="解压 knowledge/parsed 下的 MinerU zip 到同名目录"
    )
    parser.add_argument(
        "--parsed-dir",
        type=Path,
        default=DEFAULT_PARSED_DIR,
        help="zip 根目录，默认 knowledge/parsed",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="仅解压指定子目录，如 annual_reports policy",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="目标目录已存在时也重新解压",
    )
    args = parser.parse_args()

    unzip_parsed(
        args.parsed_dir,
        categories=args.categories,
        force=args.force,
    )


if __name__ == "__main__":
    main()
