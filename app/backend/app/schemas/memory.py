"""长期记忆 API schema。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator

from app.services.memory.memory_command import parse_memory_command


class MemoryCreate(BaseModel):
    command: str | None = Field(default=None, max_length=500)
    memory_key: str | None = Field(default=None, max_length=64)
    value: Any | None = None
    display_text: str | None = Field(default=None, max_length=500)
    ttl_days: int | None = Field(default=None, ge=1, le=3650)
    provenance: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_command_or_value(self) -> "MemoryCreate":
        if self.command:
            parsed = parse_memory_command(self.command)
            if parsed is None:
                raise ValueError("未识别到明确的可保存偏好")
            if self.memory_key is None:
                self.memory_key, self.value = parsed
        if not self.memory_key or self.value is None:
            raise ValueError("请提供 command，或同时提供 memory_key 和 value")
        return self


class MemoryUpdate(BaseModel):
    value: Any | None = None
    display_text: str | None = Field(default=None, max_length=500)
    ttl_days: int | None = Field(default=None, ge=1, le=3650)
    expected_version: int | None = Field(default=None, ge=1)


class EpisodicMemoryCreate(BaseModel):
    event_key: str = Field(..., min_length=1, max_length=128)
    value: dict[str, Any]
    display_text: str = Field(..., min_length=1, max_length=1000)
    expires_at: datetime | None = None
    source_conversation_id: int | None = None
    source_message_id: int | None = None
    source_run_id: str | None = Field(default=None, max_length=36)


class MemoryResponse(BaseModel):
    id: str
    memory_type: str
    memory_key: str
    value: Any
    display_text: str
    consent_status: str
    confidence: float
    status: str
    version: int
    expires_at: datetime | None
    created_at: datetime | None
    updated_at: datetime | None

    @classmethod
    def from_record(cls, record: Any) -> "MemoryResponse":
        return cls(
            id=record.id,
            memory_type=record.memory_type,
            memory_key=record.memory_key,
            value=(record.value_json or {}).get("value"),
            display_text=record.display_text,
            consent_status=record.consent_status,
            confidence=float(record.confidence or 0),
            status=record.status,
            version=record.version,
            expires_at=record.expires_at,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )
