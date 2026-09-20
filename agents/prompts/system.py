#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""角色① 程序合成的提示词文本。**改这里等于改实验条件**，所以三件事写死在前面：

1. 工具文档插在 **system** 消息里，不插在 user 消息里。
   工具文档是本题集里**逐字节不变**的那部分 —— 放前面才能吃到服务端的
   前缀缓存（同一个 prompt 前缀在多次调用间复用），成本与延迟都更低。
   题目、场景清单、上一次的失败原因放 user，它们每题都变。

2. **占位符用 `{{...}}` 而不是 `str.format` 的花括号。**
   提示词里有 ```python 代码块，里面必然出现 `{}`（字典、f-string）。
   用 `.format()` 就得转义一大片，而且**转义漏一处就是运行时 KeyError**。
   用 `replace` + 收尾断言（见 `render_template`）把这类错误变成不可能。

3. 提示词里**不写**坐标、不写答案、不写工具的实现细节。
   与 §13.3(2) 的信息约束一致：能不说的一句都别说，
   因为提示词里每一句话都是「模型本可以不必遵守」的东西，
   而架构层能堵死的（场景清单没有坐标、命名空间里没有 ctx）才是真的堵死。
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from vision.semantics import image_size_from_meta

#: 未替换的占位符（收尾自检用）。只认大写字母与下划线，免得误伤代码块里的 `{{ }}`。
_PLACEHOLDER_RE = re.compile(r"\{\{([A-Z_]+)\}\}")

__all__ = [
    "SYSTEM_TEMPLATE",
    "USER_TEMPLATE",
    "RETRY_TEMPLATE",
    "PLAN_SYSTEM_TEMPLATE",
    "PLAN_USER_TEMPLATE",
    "PLAN_BLOCK_TEMPLATE",
    "render_template",
    "build_system_prompt",
    "build_user_prompt",
    "build_retry_prompt",
    "build_plan_system_prompt",
    "build_plan_user_prompt",
    "scene_hint_for",
]

SYSTEM_TEMPLATE = """\
你是「3D 空间推理 Agent」的程序合成器。你的唯一输出是**一个完整的 Python 程序**。

手上有什么：一张照片已经建好的 3D 场景图。图里的物体有稳定 id、相机系**米制**质心与三维尺寸。
你**不能**看图、不能读点云、不能猜坐标 —— **一切空间数值只能来自工具返回值**。

## 可用工具（只能用这些；所有返回值都是 ToolResult 信封）
{{TOOL_DOCS}}

## 硬性规则（违反任一条会被零成本静态检查直接拒绝）
1. 工具调用**一律关键字传参**。位置传参被禁止：工具的第一个参数是 scene_id，
   写 `list_objects("chair")` 会把 "chair" 静默绑到 scene_id 上，得到一个看起来很正常的错答案。
2. `object_id` 只能来自 `list_objects` / `find_object` / `single_object` /
   `find_nearest` / `find_farthest` 的返回值。凭印象编一个 id 会拿到 `NOT_IN_SCENE`。
   **返回值的字段名是 `object_id`，不是 `id`**（写 `obj["id"]` / `obj.get("id")` 会得到
   `None`，然后 `a=None` 一路传到几何工具，报一个看起来莫名其妙的 `NOT_IN_SCENE`）。
   **每个工具的 `res.value` 形状写在上面对应那一行里** —— 照它读，不要猜。
   ⚠ **先看清返回值里已经有什么，再决定要不要再调一次工具**：`list_objects` 一次就给全
   每个物体的 `centroid_m` 与 `extent_m`，「最高的 / 最近的 / 有几个 / 平均多少」这类问题
   通常**不需要**为每个物体再调一次工具。
3. 每个工具都返回信封 `ToolResult`：先判 `if not res.ok:`，再读 `res.value`。
   失败时 `res.error.code` 是可枚举的错误码，`res.error.recovery` 是建议的下一步动作 —— 照它做，
   不要猜。（`CAPABILITY_DISABLED` 表示该能力在本实验臂被关闭，应当改用几何量回答。）
4. 程序必须**以 `submit(...)` 结束**：
       submit(answer, target_ids=["<物体 id>"], evidence=["<这个数是怎么算出来的>"])
   `evidence` **至少 1 条**，写清来源（工具名 / 公式）。没有 `submit` 的程序等于没有答案。
   `answer` 的类型必须符合题目要求。
5. 只能 `import`：{{MODULES}}。没有 `open`、没有文件系统、没有网络、没有 `eval`。
6. 只输出**一个** ```python 代码块。不要解释，不要贴运行结果，不要写多个方案。

## 算不出来时怎么办
场景里查不到题目要求的东西（例如问「门」但清单里没有 `door`），
或几何量不足以支撑答案时：**不要凑一个看起来合理的数字**。
用 `submit("unknown", evidence=["<说明为什么答不了>"])` 如实弃答。
编造一个带量纲的数（尤其是米制长度）是最糟的结果 —— 它会被当成一个错误答案，
而弃答会被单独统计。
"""

USER_TEMPLATE = """\
## 问题
{{QUESTION}}

## 场景内容（**只有类别清单与计数，没有坐标**）
{{SCENE_HINT}}
{{PLAN_BLOCK}}
## 要求的答案类型
{{ANSWER_TYPE}}

现在输出程序。
"""

#: 前置计划块（臂 G）。**插在场景内容之后、答案类型之前**，且整块可以为空 ——
#: 空块时这一段什么都不留（`{{PLAN_BLOCK}}` 被替换成空串），
#: 于是 `planner="off"` 与 `planner="on"` 的提示词**逐字节**只差这一段，
#: 两次实验的差异因此可以完全归因到计划本身。
PLAN_BLOCK_TEMPLATE = """
## 你上一轮列出的计划（**仅供参考**）
这是你自己在写程序前给出的思路，不是约束。若它与工具文档或场景清单冲突，
**以工具文档与场景清单为准** —— 计划里若出现了任何具体数值，一律不要采用：
本项目里所有空间数值**只能**来自工具返回值。
{{PLAN}}
"""

#: 前置规划的提示词（臂 G）。产出的是**思路**，不是程序。
#: 「不要写数字」这一条是刻意的：计划会原样进下一次合成的提示词，
#: 而程序合成无从分辨一个数是模型猜的还是工具算的（见 agents/planner.py 的说明）。
PLAN_SYSTEM_TEMPLATE = """\
你是「3D 空间推理 Agent」的**规划器**。在写程序之前，先想清楚这题该怎么解。

手上有什么：一张照片已经建好的 3D 场景图。物体有稳定 id、相机系**米制**质心与三维尺寸。
你**不能**看图、不能读点云、不能猜坐标。

## 可用工具（只能用这些）
{{TOOL_DOCS}}

## 你要输出什么
只输出 JSON，不要任何解释文字：

{"tool_categories": ["<需要用到的工具名或类别>", ...],
 "steps": ["<一步一句话，说明这一步读什么、算什么>", ...]}

硬性要求：
1. `steps` 控制在 3~6 步。每步必须说清「调哪个工具、拿它返回的什么量」。
2. **不要写任何数字**（包括"大约 1.5 米"这种估计值）。这一轮不产生答案，
   空间数值一律留到程序里由工具算出来。写了数字也会被丢弃并记录在案。
3. **不要写代码**，不要给最终答案。
4. 如果这道题在这个场景里**根本答不了**（清单里没有题目问的东西），
   就在 `steps` 里直说「场景中无此物体，应当如实弃答」。
"""

PLAN_USER_TEMPLATE = """\
## 问题
{{QUESTION}}

## 场景内容（**只有类别清单与计数，没有坐标**）
{{SCENE_HINT}}

## 要求的答案类型
{{ANSWER_TYPE}}

现在输出计划。
"""

RETRY_TEMPLATE = """\
上一次的程序没有通过，原因如下。请**只修掉这些问题**，其余部分保持不变。

## 失败原因
{{FEEDBACK}}

## 上一次的程序
```python
{{PROGRAM}}
```
"""


def render_template(template: str, values: Mapping[str, str]) -> str:
    """`{{KEY}}` 替换 + **收尾自检**。

    收尾那行断言是这个小函数存在的全部理由：漏填一个占位符时，
    模型收到的是字面量 `{{FEEDBACK}}`，它多半会自己猜一个意思继续写 ——
    于是「提示词漏了」表现成「模型不听话」，查起来要花掉一整轮实验。
    宁可在这里直接炸。
    """
    out = template
    for key, val in values.items():
        out = out.replace("{{%s}}" % key, val)
    leftover = [k for k in _PLACEHOLDER_RE.findall(out)]
    if leftover:
        raise KeyError("提示词占位符没被填上：%s" % sorted(set(leftover)))
    return out


def scene_hint_for(scene: Any) -> dict[str, Any]:
    """`scene_hint` 的**唯一**构造点：类别清单 + 计数 + 图像尺寸。

    ★ 这里刻意**不**放 `camera_intrinsics`、不放 `up_axis`、不放 node 坐标。
    §21 的结论是「内参决定横向米制尺度」，把它写进提示词等于告诉模型
    「米制尺度不可靠，你看着办」—— 那会让不同题目拿到不同的暗示，
    破坏可比性。尺度问题由**场景图的构建方**（`vision/depth.py` 的
    `resolve_intrinsics`）负责，不是提示词的工作。

    ⚠ **图像尺寸的键名**：builder 落盘的 `build_meta` 用的是 `image_hw`（**高, 宽**），
    本函数对外统一输出 `image_size`（**宽, 高**，与 PIL 一致）。
    这里曾经只认 `image_width` / `image_height` —— 而真实场景的 build_meta 里
    **没有这两个键**，于是「图像尺寸」这个本文档承诺过的字段在真实数据上
    **从来就没出现过**，且没有任何测试或告警发现它。口径统一在 `_image_size_of()` 一处。
    """
    hint: dict[str, Any] = {
        "objects": dict(scene.label_counts()),
        "n_objects": len(scene.nodes),
        "image_id": getattr(scene, "image_id", None),
        "scene_id": getattr(scene, "scene_id", None),
    }
    # 尺寸的归一化与角色② 共用 `vision.semantics.image_size_from_meta` —— 只有一份实现。
    # ⚠ 它住中性层而不是 `llm/vlm.py`：工具层也要用它，而分层禁令不许 `tools/` import `llm/`。
    size = image_size_from_meta(getattr(scene, "build_meta", None) or {})
    if size is not None:
        hint["image_size"] = list(size)          # (宽, 高)，与 PIL 一致
    meta = getattr(scene, "build_meta", None) or {}
    if "intrinsics_source" in meta:
        hint["intrinsics_source"] = meta["intrinsics_source"]
    return hint


def build_system_prompt(tool_docs: str, modules: Sequence[str]) -> str:
    return render_template(
        SYSTEM_TEMPLATE,
        {"TOOL_DOCS": tool_docs, "MODULES": ", ".join(modules)},
    )


def build_user_prompt(
    question: str,
    scene_hint: Mapping[str, Any],
    *,
    answer_type: str | None = None,
    plan_text: str | None = None,
) -> str:
    """合成用的 user 提示词。`plan_text` 非空时插入前置计划块（臂 G）。

    **`plan_text=None` 与 `plan_text=""` 等价**（都渲染成空块）——
    于是 `planner="off"` 与 `planner="on"` 的提示词只差这一段，
    两次实验的差异可以完全归因到计划本身（见 `PLAN_BLOCK_TEMPLATE`）。
    """
    at = answer_type or "str"
    hint_text = json.dumps(dict(scene_hint), ensure_ascii=False, sort_keys=True)
    if at in ("int", "float"):
        extra = "%s（一个数，不要带单位，不要带文字）" % at
    elif at == "bool":
        extra = "bool（true / false）"
    else:
        extra = "%s（一个词或短语；颜色/类别这类题目通常是闭集里的一个词）" % at
    block = ""
    if plan_text and plan_text.strip():
        block = render_template(PLAN_BLOCK_TEMPLATE, {"PLAN": plan_text.strip()})
    return render_template(
        USER_TEMPLATE,
        {"QUESTION": question, "SCENE_HINT": hint_text, "ANSWER_TYPE": extra,
         "PLAN_BLOCK": block},
    )


def build_plan_system_prompt(tool_docs: str | None, tools: Sequence[str]) -> str:
    """规划角色的 system 提示词。

    `tool_docs` 为 None 时**自己渲染一份**（与合成用的那份同源），
    因为「计划里列的工具类别」必须对着同一份文档核对 —— 用另一份渲染
    等于让两个角色看到的工具集可能不一致，而那正是最该避免的不一致。
    """
    docs = tool_docs
    if docs is None:
        from llm.schema import docs_text

        docs = docs_text(
            tools=tools,
            heading="（返回类型统一是 ToolResult：先判 res.ok 再读 res.value）",
        )
    return render_template(PLAN_SYSTEM_TEMPLATE, {"TOOL_DOCS": docs})


def build_plan_user_prompt(
    question: str,
    scene_hint: Mapping[str, Any],
    *,
    answer_type: str | None = None,
) -> str:
    """规划角色的 user 提示词。**它不需要任何额外信息 —— 尤其是不给坐标。**

    它的输入与 `build_user_prompt` 的前半段逐字节相同（同一条 `scene_hint`），
    所以「计划有没有引入额外信息」这个问题的答案是确定的：没有。
    """
    at = answer_type or "str"
    hint_text = json.dumps(dict(scene_hint), ensure_ascii=False, sort_keys=True)
    return render_template(
        PLAN_USER_TEMPLATE,
        {"QUESTION": question, "SCENE_HINT": hint_text,
         "ANSWER_TYPE": "%s（只说明类型，这一轮不要算出数值）" % at},
    )


def build_retry_prompt(feedback: str, program: str) -> str:
    return render_template(RETRY_TEMPLATE, {"FEEDBACK": feedback, "PROGRAM": program})
