"""Redis Key 的统一构造与校验。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re

from app.core.config import settings

_SEGMENT_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True, slots=True)
class RedisKey:
    """经过统一构造器校验的 Redis Key。"""

    value: str

    def __str__(self) -> str:
        return self.value


class RedisKeyBuilder:
    """统一添加项目、环境与租户作用域，避免业务代码手工拼接 Key。"""

    def __init__(self, *, prefix: str, environment: str) -> None:
        self._prefix = self._validate_segment(prefix, name="prefix")
        self._environment = self._validate_segment(environment, name="environment")

    @staticmethod
    def _validate_segment(value: object, *, name: str = "segment") -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError(f"Redis Key {name} 不能为空")
        if len(normalized) > 128:
            raise ValueError(f"Redis Key {name} 长度不能超过 128")
        if not _SEGMENT_PATTERN.fullmatch(normalized):
            raise ValueError(
                f"Redis Key {name} 只能包含字母、数字、点、下划线和短横线"
            )
        return normalized

    def build(self, domain: str, *segments: object) -> RedisKey:
        parts = [
            self._prefix,
            self._environment,
            self._validate_segment(domain, name="domain"),
            *(
                self._validate_segment(segment)
                for segment in segments
            ),
        ]
        return RedisKey(":".join(parts))

    def tenant(
        self,
        domain: str,
        tenant_id: object,
        *segments: object,
    ) -> RedisKey:
        return self.build(domain, "tenant", tenant_id, *segments)

    def user(
        self,
        domain: str,
        tenant_id: object,
        user_id: object,
        *segments: object,
    ) -> RedisKey:
        return self.build(
            domain,
            "tenant",
            tenant_id,
            "user",
            user_id,
            *segments,
        )

    @staticmethod
    def digest(value: str) -> str:
        """长文本或敏感输入只能以摘要形式进入 Key。"""
        return hashlib.sha256(value.encode("utf-8")).hexdigest()


redis_keys = RedisKeyBuilder(
    prefix=settings.REDIS_KEY_PREFIX,
    environment=settings.APP_ENV,
)


__all__ = ["RedisKey", "RedisKeyBuilder", "redis_keys"]
