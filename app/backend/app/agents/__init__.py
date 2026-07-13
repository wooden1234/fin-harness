"""Shim：``app.agents`` 指向项目根 ``agents`` 包，兼容历史 import 路径。"""

from __future__ import annotations

import importlib
import sys

_agents = importlib.import_module("agents")
sys.modules[__name__] = _agents
