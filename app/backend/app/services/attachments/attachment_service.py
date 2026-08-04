"""聊天附件：校验、上传 OSS、元数据缓存。"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any

from fastapi import HTTPException, UploadFile

from app.core.config import settings
from app.core.logger import get_logger
from app.core.redis_client import RedisUnavailableError, get_redis_client
from app.core.redis_keys import redis_keys
from app.services.storage.oss_client import (
    OssNotConfiguredError,
    get_object_bytes,
    object_url,
    oss_configured,
    put_object,
    signed_object_url,
)

logger = get_logger(service="attachments")

_ALLOWED_CONTENT_TYPES = frozenset(
    {"image/jpeg", "image/png", "image/webp"}
)
_EXT_BY_TYPE = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
}
# Redis 不可用时的进程内降级缓存（仅开发/测试）
_MEMORY_STORE: dict[str, tuple[float, dict[str, Any]]] = {}


@dataclass(frozen=True, slots=True)
class AttachmentMeta:
    attachment_id: str
    object_key: str
    content_type: str
    size: int
    user_id: int
    tenant_id: str
    created_at: float

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "AttachmentMeta":
        data = json.loads(raw)
        return cls(
            attachment_id=str(data["attachment_id"]),
            object_key=str(data["object_key"]),
            content_type=str(data["content_type"]),
            size=int(data["size"]),
            user_id=int(data["user_id"]),
            tenant_id=str(data.get("tenant_id") or "default"),
            created_at=float(data.get("created_at") or 0),
        )


def _meta_key(attachment_id: str):
    return redis_keys.build("attachment", attachment_id)


def _normalize_content_type(upload: UploadFile) -> str:
    content_type = str(upload.content_type or "").split(";")[0].strip().lower()
    if content_type == "image/jpg":
        content_type = "image/jpeg"
    return content_type


async def _store_meta(meta: AttachmentMeta) -> None:
    ttl = max(60, int(settings.ATTACHMENT_TTL_SEC or 86_400))
    payload = meta.to_json()
    client = get_redis_client()
    if client is not None:
        try:
            await client.set(_meta_key(meta.attachment_id), payload, ttl_seconds=ttl)
            return
        except RedisUnavailableError:
            logger.warning("attachment meta redis unavailable, using memory store")
    expires_at = time.time() + ttl
    _MEMORY_STORE[meta.attachment_id] = (expires_at, json.loads(payload))


async def get_attachment_meta(attachment_id: str) -> AttachmentMeta | None:
    attachment_id = str(attachment_id or "").strip()
    if not attachment_id:
        return None
    client = get_redis_client()
    if client is not None:
        try:
            raw = await client.get(_meta_key(attachment_id))
            if raw:
                return AttachmentMeta.from_json(raw)
        except RedisUnavailableError:
            logger.warning("attachment meta redis read unavailable")
    entry = _MEMORY_STORE.get(attachment_id)
    if entry is None:
        return None
    expires_at, data = entry
    if expires_at < time.time():
        _MEMORY_STORE.pop(attachment_id, None)
        return None
    return AttachmentMeta.from_json(json.dumps(data))


async def require_owned_attachment(
    attachment_id: str,
    *,
    user_id: int,
    tenant_id: str,
) -> AttachmentMeta:
    meta = await get_attachment_meta(attachment_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="附件不存在或已过期")
    if meta.user_id != user_id or meta.tenant_id != tenant_id:
        raise HTTPException(status_code=403, detail="无权访问该附件")
    return meta


async def upload_image_attachment(
    upload: UploadFile,
    *,
    user_id: int,
    tenant_id: str = "default",
) -> AttachmentMeta:
    if not oss_configured():
        raise HTTPException(
            status_code=503,
            detail="图片上传未配置：请设置 OSS_ACCESS_KEY_ID / "
            "OSS_ACCESS_KEY_SECRET / OSS_ENDPOINT / OSS_BUCKET",
        )
    content_type = _normalize_content_type(upload)
    if content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail="仅支持 JPEG / PNG / WebP 图片",
        )
    max_bytes = max(1024, int(settings.ATTACHMENT_MAX_BYTES or 5_242_880))
    data = await upload.read(max_bytes + 1)
    if not data:
        raise HTTPException(status_code=400, detail="空文件")
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"图片超过大小限制（最大 {max_bytes} 字节）",
        )

    attachment_id = str(uuid.uuid4())
    ext = _EXT_BY_TYPE[content_type]
    prefix = str(settings.OSS_PREFIX or "chat-images/").strip("/")
    object_key = (
        f"{prefix}/tenant-{tenant_id}/user-{user_id}/{attachment_id}.{ext}"
    )
    try:
        put_object(object_key=object_key, data=data, content_type=content_type)
    except OssNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("oss upload failed")
        raise HTTPException(status_code=502, detail="上传到对象存储失败") from exc

    meta = AttachmentMeta(
        attachment_id=attachment_id,
        object_key=object_key,
        content_type=content_type,
        size=len(data),
        user_id=user_id,
        tenant_id=tenant_id,
        created_at=time.time(),
    )
    await _store_meta(meta)
    return meta


def attachment_access_url(meta: AttachmentMeta) -> str:
    """供 Vision 等服务端拉取：私有桶必须用签名 URL（公网直链常 403）。"""
    return signed_object_url(meta.object_key)


def download_attachment_bytes(meta: AttachmentMeta) -> bytes:
    """服务端鉴权下载附件字节。"""
    return get_object_bytes(meta.object_key)


def attachment_public_url(meta: AttachmentMeta) -> str:
    """对外展示用 URL（依赖桶可读或回退签名）。"""
    return object_url(meta.object_key)


__all__ = [
    "AttachmentMeta",
    "attachment_access_url",
    "attachment_public_url",
    "download_attachment_bytes",
    "get_attachment_meta",
    "require_owned_attachment",
    "upload_image_attachment",
]
