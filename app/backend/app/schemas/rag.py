from typing import Any
from pydantic import BaseModel, Field


class RagHitItem(BaseModel):
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    node_id: str | None = None


class RagSearchResponse(BaseModel):
    query: str
    top_k: int
    hits: list[RagHitItem]