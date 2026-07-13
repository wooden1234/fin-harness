from pydantic import BaseModel, Field

from app.shared import RiskLevel


class RiskAssessment(BaseModel):
    """风险评估结果（独立模型，不污染 Router）"""
    risk_level: RiskLevel = Field(description="L1-L4 风险等级")
    reason: str = Field(description="判定理由，1-2 句中文")
    needs_human: bool = Field(default=False, description="是否需要转人工")
