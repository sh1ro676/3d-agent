"""让测试能从项目根 import `evaluation`。

与 `scene_graph/tests/conftest.py` 同一套路（不装包，手写路径）。
路径判定用 `parents[2]`：
    <root>/evaluation/tests/conftest.py  →  parents[0]=tests, [1]=evaluation, [2]=<root>

口径层是**零 torch、零 GPU、零联网**的：这里不需要任何模型目录或缓存路径，
要保证的恰恰是「不碰它们也能跑」。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
