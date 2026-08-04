"""阿里云 OSS 客户端（聊天图片上传）。"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(service="oss")


class OssNotConfiguredError(RuntimeError):
    """缺少 OSS 必要配置。"""


def oss_configured() -> bool:
    return bool(
        settings.OSS_ACCESS_KEY_ID
        and settings.OSS_ACCESS_KEY_SECRET
        and settings.OSS_ENDPOINT
        and settings.OSS_BUCKET
    )


@lru_cache(maxsize=1)
def _bucket() -> Any:
    if not oss_configured():
        raise OssNotConfiguredError(
            "未配置 OSS_ACCESS_KEY_ID / OSS_ACCESS_KEY_SECRET / "
            "OSS_ENDPOINT / OSS_BUCKET"
        )
    import oss2

    auth = oss2.Auth(settings.OSS_ACCESS_KEY_ID, settings.OSS_ACCESS_KEY_SECRET)
    endpoint = settings.OSS_ENDPOINT.strip()
    if not endpoint.startswith("http://") and not endpoint.startswith("https://"):
        endpoint = f"https://{endpoint}"
    return oss2.Bucket(auth, endpoint, settings.OSS_BUCKET)


def put_object(*, object_key: str, data: bytes, content_type: str) -> None:
    bucket = _bucket()
    headers = {"Content-Type": content_type} if content_type else None
    result = bucket.put_object(object_key, data, headers=headers)
    status = int(getattr(result, "status", 0) or 0)
    if status not in {200, 203}:
        raise RuntimeError(f"oss_put_failed:status={status}")


def get_object_bytes(object_key: str) -> bytes:
    """用 AK 鉴权下载对象（供 Vision 转 base64，避免公网/签名拉取超时）。"""
    bucket = _bucket()
    result = bucket.get_object(object_key)
    data = result.read()
    if not isinstance(data, (bytes, bytearray)):
        data = bytes(data)
    return bytes(data)


def signed_object_url(object_key: str, *, expires_sec: int | None = None) -> str:
    """生成带签名的临时 GET URL（私有桶可被 DashScope 拉取）。"""
    bucket = _bucket()
    expires = max(
        60,
        int(
            expires_sec
            if expires_sec is not None
            else (settings.OSS_SIGN_URL_EXPIRES_SEC or 3600)
        ),
    )
    url = bucket.sign_url("GET", object_key, expires)
    # oss2 偶发签出 http，统一成 https 便于外网模型拉取
    if url.startswith("http://"):
        url = "https://" + url[len("http://") :]
    return url


def object_url(object_key: str) -> str:
    """公网前缀（仅当桶可读时）；否则回退签名 URL。"""
    base = str(settings.OSS_PUBLIC_BASE_URL or "").strip().rstrip("/")
    if base:
        if not base.startswith("http://") and not base.startswith("https://"):
            base = f"https://{base}"
        return f"{base}/{object_key.lstrip('/')}"
    return signed_object_url(object_key)


__all__ = [
    "OssNotConfiguredError",
    "get_object_bytes",
    "object_url",
    "oss_configured",
    "put_object",
    "signed_object_url",
]
