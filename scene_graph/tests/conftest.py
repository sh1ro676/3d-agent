"""让测试能从项目根 import `scene_graph` / `tools`。

不装包（没有 `pip install -e .`），所以手写路径。项目根的判定用 `parents[2]`：
    <root>/scene_graph/tests/conftest.py  →  parents[0]=tests, [1]=scene_graph, [2]=<root>
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
