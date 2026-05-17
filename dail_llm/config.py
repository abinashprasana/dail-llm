"""
Backward-compatibility shim.
The canonical configuration has moved to config.py at the project root.
All new code should use: from config import ...
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import *  # noqa: F401, F403
