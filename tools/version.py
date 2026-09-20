"""工具库版本号 —— 单一来源。

为什么要一个独立文件（§11 设计原则 5）：
    VADAR 的工具集是运行时从输入里随机抽样生成的（`--num-api-questions` 默认 10，
    `evaluate.py:42` 用 `random.sample` 抽题），所以每次运行的工具都不同、结果不可比。
    本项目的工具库固定 + 版本化：版本号进 `ToolMeta.version`，随 trace 与结果 JSON 落盘，
    于是「两次实验用的是不是同一套工具」这件事本身可被核对。

改工具行为时必须升版本，否则历史结果的可比性就断了。
"""

from __future__ import annotations

#: 工具库版本。改工具行为**或增减工具**时必须升版本，否则历史结果的可比性就断了。
#:
#: 1.0.0 → 1.1.0（2026-09-16）：新增 L5 场景级输出工具（`describe_scene` /
#:   `summarize_scene` / `diagnose_failure` / `counterfactual`，见 tools/scene_report.py）。
#:   为什么「只加不减」也要升：版本号存在的一半理由是回答「两次实验用的是不是同一套工具」，
#:   而工具集合变大意味着可选的程序空间变大 —— LLM 生成程序的行为**会**因此改变。
#:   已落盘的旧 scene.json 里 `build_meta.tools_version` 仍是 1.0.0，那是正确记录，不要回填。
#: 1.1.0 → 1.2.0（2026-09-18）：新增 L4 视觉语义工具 `get_attributes`（tools/attributes.py），
#:   它使自研臂的动作空间从 11 个变成 12 个。**这一条对可比性影响最大** ——
#:   1.1.0 之前跑出的自研臂结果不能再与之后的结果放进同一张表。
#: 1.2.0 → 1.3.0（2026-09-19）：工具数**一个没变**（仍 12 个），但模型可见面变了：
#:   ① 每个读类工具的 docstring **第一段**补上了 `res.value` 的形状 —— 而第一段正是
#:      渲染进提示词的那一段（`llm/schema.py::first_paragraph`），等于改了 prompt；
#:   ② `get_object` 的 value 多了 `bbox_2d`（图像像素系），二维区域类题目从此可答；
#:   ③ `query_relation` 的文档写明了 `tol` 的默认值（0.05 m 的判定死区）——
#:      「容差死区」从此是程序可控的参数，而不是一个看不见的隐含条件。
#:   为什么「只改文档」也要升版本：工具文档是**逐字节进 prompt 的**，模型照着它生成程序，
#:   所以它与工具集合一样属于实验条件。1.2.0 的结果与 1.3.0 的结果不能混进同一张表。
TOOLS_VERSION = "1.3.0"

#: 几何关系与位置算法的来源标记。写进 Edge.method 与 evidence["method"]。
#: 换了关系定义（例如从 bbox 版换成点云版）就换这个字符串 —— 重跑可 diff（§12.4）。
RELATIONS_METHOD = "geometry_v1"

#: 场景图 JSON 的存储格式版本，供 store.py 判断是否需要迁移。
SCENE_FORMAT_VERSION = "1.0"


def tool_version(name: str) -> str:
    """返回 `<name>@<TOOLS_VERSION>`，用于 ToolMeta 与实验臂指纹。"""
    return f"{name}@{TOOLS_VERSION}"


__all__ = ["TOOLS_VERSION", "RELATIONS_METHOD", "SCENE_FORMAT_VERSION", "tool_version"]
