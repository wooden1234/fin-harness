"""运行指标和 trace 记录。"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter


@dataclass(slots=True)
class Timer:
    """简单耗时计时器。"""

    started_at: float

    @classmethod
    def start(cls) -> "Timer":
        return cls(started_at=perf_counter())

    def elapsed_ms(self) -> float:
        return (perf_counter() - self.started_at) * 1000
