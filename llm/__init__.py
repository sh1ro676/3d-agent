"""LLM 层 —— 三角色里「要与外界说话」的那两个角色的落地实现（§13.3）。

    角色① 程序合成  `agents/synthesizer.py`（驱动者）+ `llm/schema.py`（工具文档）+ 本包的客户端
    角色② 视觉语义  `llm/vlm.py`（**尚未实现**，见下）
    角色③ 几何层    不需要 LLM，也不许调用 LLM（§13.3(6) 禁令三）

为什么和 `agents/` 分成两个包
============================
`llm/` 只回答「怎么跟一个 OpenAI 兼容端点说话」——
它不知道工具库、不知道场景图、不知道 `submit` 契约。
`agents/` 才回答「一次问答由哪几步组成」。

这条边界的实际收益：**换后端不动循环，改循环不动后端**。
`tests/test_agent_layering.py` 用 AST 静态检查把这条边界钉住：

    · `agents/` **不许** import `evaluation/` —— evaluation 是**测量** agent 的器械，
      被测方依赖测量方会把方向反过来（那一层塌了，循环和它的实验一起崩）
    · `tools/` 与 `scene_graph/` **不许** import `llm` —— §13.3(6) 禁令三：几何层永不调用 LLM
    · `llm/adapter.py` 与 `llm/render.py` 不 import `tools` / `scene_graph` ——
      它们是纯「跟端点说话」的层
      （`llm/schema.py` 是**唯一例外**：它要把工具签名渲染成提示词，所以在函数内部延迟
       `import tools`；这也是它必须留在 `llm/` 的原因 —— 它的产出是提示词，不是推理）

`llm/vlm.py` 为什么还没写
========================
角色②（颜色/材质/状态）不在主路径上 —— §13.3(3) 明确写了「主路径（纯空间问答）不依赖本角色」。
所以它缺席时，循环仍然完整、可跑、可评测；颜色类题目会拿到
`ErrorCode.CAPABILITY_DISABLED`（这是设计好的消融开关，不是崩溃）。
把它排在下一批，是为了不让「循环还没跑通」和「VLM 还没接」两件事互相阻塞。
"""

from __future__ import annotations

__all__ = ["adapter", "render", "schema"]
