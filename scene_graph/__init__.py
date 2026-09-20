"""`scene_graph` —— 3D 场景图的中间表示层。

对外接口：
    schema.py     数据结构（本包的类型定义，Pydantic）
    relations.py  ★ 纯几何关系函数，零依赖、可脱离场景图单测
    builder.py    单遍构建：检测 → 分割 → 升维 → 关系（Phase 6）
    store.py      JSON 缓存读写 + 版本（Phase 6）

分工的铁律（§13.3(4)）：`relations.py` 只吃 `Node`、返回数学结论；
「校验 object_id 是否存在」「组装 evidence」「套上 ToolResult」全在 `tools/spatial.py`。
混成一层，关系函数就再也测不了了 —— 而「关系可单测」正是 Scene Graph 相对 VADAR 的
三条收益之一（§12.1）。
"""

__all__: list[str] = []
