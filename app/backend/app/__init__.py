"""Backend application package."""

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent.parent

_root = str(PROJECT_ROOT)
if _root not in sys.path:
    sys.path.insert(0, _root)
