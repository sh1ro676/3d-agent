"""自研 3D Spatial Agent —— 循环层（Phase 7）。

    一次问答的组成（主路径 = program synthesis，**1 次 LLM 调用**）
    ---------------------------------------------------------------
    ① `synthesizer.synthesize()`   角色① 生成一个完整 Python 程序
    ② `synthesizer.static_check()` AST 四项静态检查（零成本、不执行）
    ③ `executor.execute_program()` 沙箱执行：工具命名空间 + `submit()` + 硬超时 + trace
    ④ `llm.render.render_answer()` 模板渲染（数值槽位由 `submit` 锁定）
                          ↓
                    `loop.AgentRun`

与早期基线实现沿用的四段流水线（签名 → 逐 API 实现 → 自测 → 拼程序 → 执行）相比，
这里把「签名」和「逐 API 实现」整个删掉了：工具库是**固定的、版本化的**，
不需要模型在运行时发明 API。省下的不只是 3 次 LLM 往返，
更重要的是**去掉了「运行时随机生成 API」这个不可复现的根源**（§11 设计原则 5）。

三条禁令（§13.3(6)）在代码里的落点
--------------------------------
    ① synthesize 不得直接读点云数组
       → `executor` 的命名空间里**没有 `ctx`**，程序拿不到 SceneGraph，也拿不到点云
    ② describe 不得输出坐标或空间判断
       → `llm/vlm.py` 的 `attrs` 是 `Literal` 白名单，空间参数在类型层面写不出来
    ③ 几何层不得发起模型调用
       → `tools/` 与 `scene_graph/` 里没有任何 `import llm`
       （`tests/test_agent_layering.py` 用 AST 直接守这条）

包里各文件的职责边界
------------------
    executor.py     执行一段已经写好的程序（**不认识 LLM**）
    synthesizer.py  把问题变成一段程序（**不认识沙箱**，只管生成与静态检查）
    loop.py         把上面两者串起来，负责重试、记账、状态归类
    memory.py       工作记忆：本轮已确认的 id 与失败原因（压缩后进下一次 prompt）
"""

from __future__ import annotations

__all__ = ["executor", "loop", "memory", "prompts", "synthesizer"]
