"""Harness 统一错误类型。"""


class HarnessError(Exception):
    """Harness 基础异常。"""


class PolicyDeniedError(HarnessError):
    """策略拒绝本次运行。"""


class RegistryLookupError(HarnessError):
    """注册表中没有找到目标能力。"""
