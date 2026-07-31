"""pytest 公共配置：加载项目根目录 .env。"""

from pathlib import Path
import sys

import pytest
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "app" / "backend"
DEPS_DIR = ROOT_DIR / ".deps"
python_abi = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
# `.deps` 可能由另一个 Python 小版本构建，错误注入会遮蔽 Conda 中可用的扩展包。
deps_binaries = list(DEPS_DIR.rglob("*.so")) if DEPS_DIR.exists() else []
compatible_deps = not deps_binaries or any(
    python_abi in path.name for path in deps_binaries
)
paths = [str(BACKEND_DIR), str(ROOT_DIR)]
if compatible_deps:
    paths.append(str(DEPS_DIR))
for path in paths:
    if path not in sys.path:
        sys.path.insert(0, path)

load_dotenv(ROOT_DIR / ".env", override=False)


def _has_embedding_api_key() -> bool:
    import os

    return bool(
        os.getenv("EMBEDDING_API_KEY")
        or os.getenv("QWEN_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
    )


requires_embedding_key = pytest.mark.skipif(
    not _has_embedding_api_key(),
    reason="未配置 EMBEDDING_API_KEY / QWEN_API_KEY / DASHSCOPE_API_KEY，跳过 Embedding 集成测试",
)


def _has_llm_api_key() -> bool:
    import os

    return bool(os.getenv("DEEPSEEK_API_KEY"))


requires_llm_key = pytest.mark.skipif(
    not _has_llm_api_key(),
    reason="未配置 DEEPSEEK_API_KEY，跳过 Supervisor LLM 集成测试",
)
