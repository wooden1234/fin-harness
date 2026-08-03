"""Main DeepAgent 单主路径入口。"""

from agents.main_deep_agent.node import main_deep_agent_node
from agents.main_deep_agent.quality import main_evidence_quality_gate

__all__ = ["main_deep_agent_node", "main_evidence_quality_gate"]
