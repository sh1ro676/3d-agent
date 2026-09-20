"""让测试能从项目根 import `evaluation`。

与 `scene_graph/tests/conftest.py` 同一套路（不装包，手写路径）。
路径判定用 `parents[2]`：
    <root>/evaluation/tests/conftest.py  →  parents[0]=tests, [1]=evaluation, [2]=<root>
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 单测必须能离线跑：HF_HUB_OFFLINE 不影响纯逻辑，但 vadar_compat 的
# groundino 句柄构造会去看权重目录，提前指到项目内路径避免它去找用户缓存。
os.environ.setdefault("VADAR_GDINO_DIR", str(ROOT / ".cache" / "models" / "grounding-dino-tiny"))
