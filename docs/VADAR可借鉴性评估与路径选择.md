# VADAR 可借鉴性评估与路径选择

> 评估日期：2026-09-15
> 结论一句话：**没有无法克服的问题，但正确做法是「借 VADAR 的技术路线、重写实现」，而不是「在 VADAR 代码上打补丁」。**

---

## 0. 摘要

你提的定位是：VADAR 只作参考，目标是用国产模型做一个同类系统，遇到无法克服的问题就换路径。

核查后的答案是：

| 问题 | 答案 |
|---|---|
| VADAR 有值得借鉴的东西吗？ | **有，而且核心就一条，但它是整个项目最值钱的设计决策** |
| 有无法克服的问题吗？ | **没有。一条都没有。** |
| 需要换路径吗？ | 不需要换"技术路线"，但需要换"实现载体"——**不用 VADAR 的代码框架** |
| 我之前的判断有错吗？ | **有，两处说重了。第 3 节逐条更正** |
| 意外的收获 | **WSL2 不再是必需项。整条视觉栈可以在 Windows 原生跑** |

---

## 1. VADAR 的三层拆解

把一个开源项目"拆开来看"时，要区分三种东西：**思想**（能抄）、**代码**（能改）、**包袱**（必须扔）。混在一起谈就会得出"要么全用要么全弃"的错误结论。

### 1.1 第一层：真正值钱的那一条 —— 动作空间设计

**VADAR 的核心贡献不是 agent 架构，是它给 LLM 选的"动作空间"：让模型输出一段 Python 程序，而不是输出一串 JSON 工具调用。**

这个区别在**空间推理**任务上被放大成决定性差异。看下面这个典型问题：

> "哪个椅子离门最近？"

| | JSON tool calling | 程序合成（VADAR 的做法） |
|---|---|---|
| 需要几步 | 4 步以上，每步一次模型往返 | 1 次生成 |
| 能否表达 `argmin` | ❌ 表达不了。只能"逐个比较、记住最小的"，靠上下文维持状态 | ✅ `min(chairs, key=lambda c: dist(c, door_pos))` |
| 能否做算术 | ❌ 只能让模型自己算，容易错 | ✅ 交给 Python |
| 能否做过滤/聚合 | ❌ 极难 | ✅ `[c for c in chairs if c.x < door.x]` |
| 能否循环 | ❌ 不能 | ✅ `for obj in objs: ...` |
| 出错后 | 好定位（知道哪一步错） | 差一些（整段程序一起失败） |

**空间问题的本质是"组合 + 算术 + 聚合"**：找最近的、数一共有几个、判断在不在上面、算面积比。这些恰好是 JSON function calling 最不擅长的，也恰好是 Python 最擅长的。

所以：**这一条必须抄，而且要作为整个项目的技术底座。** 这是 VADAR 论文里最经得起时间考验的部分。

### 1.2 第二层：可借鉴但需重写

| 借鉴点 | VADAR 的做法 | 我们的做法 |
|---|---|---|
| **感知与推理分离** | 视觉模型只负责 `loc` / `depth` / 分割，空间关系全部由代码算 | 保留，并**强化**——把几何计算做成独立的一层 |
| **两段式 prompt 结构** | 先让模型提"签名"（函数名+docstring），再让它写程序 | 简化为**一段式**：给定固定工具文档，直接生成程序 |
| **工具文档即 prompt** | `MODULES_SIGNATURES` 字符串注入 prompt | 保留这个思路，但换成**我们自己固定、带版本号的工具库** |
| **评测协议** | Omni3D-Bench，按"程序能否执行 + 答案是否匹配"打分 | 沿用协议，扩展指标（见主方案 §13） |
| **`--oracle` 上限对照** | 给真值绕开视觉检测，单独测"推理能力" | 保留，这个消融设计很聪明 |

### 1.3 第三层：必须扔掉的三个包袱

#### 包袱 1：运行时动态生成 API ★最需要扔的

VADAR 的招牌卖点是"Dynamic API"——面对新问题，先让 LLM 发明一个新函数（SignatureAgent 提签名），再让 LLM 实现它（APIAgent 写代码 + 自测，**失败重试 5 次**），最后才写主程序。

**这个设计对 gpt-4o 是加分项，对 4B 模型是致命的：**

- 每一环都是一次零容错的生成（正则解析 `<signature>` / `<docstring>` / `<implementation>`，无兜底）
- APIAgent 那 5 次自测重试，本质是在**用算力掩盖模型能力不足**——4B 模型重试 5 次仍然写不对的概率很高
- 三环串联，成功率是**乘法**关系。假设每环 80%，0.8³ = 51%
- 生成出来的 API 不进版本控制、不可复现，**你的实验无法被自己重跑**

**替代方案：固定工具库。** 预先设计好 20–30 个空间/几何工具，写进 prompt，模型只需"选 + 组合"。这样：

- 成功率从"乘法"变成"加法"（只有一次生成）
- 工具是固定的 → **实验可复现**（这本就是你方案里的创新点 1）
- 模型只需理解接口，不需要发明接口——**难度降一个量级**

这不是"降级"，而是把 VADAR 里**为了论文卖点而加的知识负担**卸掉，让 4B 模型能专注在真正的空间推理上。

#### 包袱 2：Engine 执行层

`engine/engine.py` 用 `runpy` + `sys.settrace` + `signal.alarm(200)` 执行生成的代码。问题：

- `signal.SIGALRM` 是 Unix-only（这是"必须 WSL"的唯一来源）
- `sys.settrace` 造成全局性能污染，异常路径下钩子残留
- 用 `runpy` 在主进程里跑 LLM 生成的代码，**沙箱边界等于没有**

**替代方案：子进程 + 超时。** `multiprocessing.Process` + `join(timeout)` 或 `subprocess` + `timeout`，跨平台、真隔离、能拿到退出码。**顺带把"必须 WSL2"这个约束整个消掉。**

#### 包袱 3：钉死的依赖与 OpenAI

- `transformers==4.45.2`（2024-09）——这个版本里**没有** `Sam2Model`（2025-08 才进 transformers），也没有新一代 Qwen-VL
- `vqa()` 硬编码 `gpt-4o`，且用 `open(api_key_path)` 读 key（无 `base_url`）
- 失败分支是 `time.sleep(60)` + 无限递归重试 → **网络不通时静默卡死**

**替代方案：见第 5 节的新栈。**

---

## 2. 一个额外的发现：VADAR 主动丢掉了 3D 信息

这条独立于"借鉴还是抛弃"，是**你的项目创新点的地基**，已从源码逐行确认：

`unidepth/models/unidepthv2/unidepthv2.py` 的 `encode_decode` 里：

```python
pts_3d = outputs["rays"] * outputs["radius"]
outputs.update({"points": pts_3d, "depth": pts_3d[:, -1:]})
```

`infer()` 最终返回 **7 个键**：

| 键 | 含义 | VADAR 是否使用 |
|---|---|---|
| `depth` | `points[:, -1:]`，即 3D 点的 z 分量 | ✅ **只用了这个** |
| `points` | 相机坐标系下的完整 3D 点云 `(B,3,H,W)` | ❌ **丢弃** |
| `rays` | 每个像素的单位方向向量 | ❌ **丢弃** |
| `intrinsics` | 相机内参 | ❌ **丢弃** |
| `confidence` | 逐像素置信度 | ❌ 丢弃 |
| `radius` | 深度范数 | ❌ 丢弃 |
| `depth_features` | 深度特征 | ❌ 丢弃 |

而 VADAR 的 3D 尺寸估计用的是 `2D_pixel × depth`（连焦距都没除），是个**量纲都不对的近似**。

**结论：不增加任何模型、不增加任何显存，只要改成用 `points` 做几何计算，就能把"深度标量近似"升级为"真实三维几何"。** 这是本项目最低成本、最高说服力的创新点。

---

## 3. 我之前判断的更正（诚实记录）

调研中我下过几个过重的结论，逐条更正：

| # | 我说过的 | 实际情况 | 严重程度 |
|---|---|---|---|
| 1 | "UniDepth 依赖 `triton>=2.4.0`，triton 官方不支持 Windows → 这是比 SIGALRM 更硬的阻塞，必须 WSL2" | **过重。** xformers 在源码里是**软依赖**，失效时回落到 PyTorch 自带的 SDPA。推理路径完全不需要 xformers/triton（证据见下方代码块） | 🔴 结论性错误 |
| 2 | "GroundingDINO / SAM2 自带 CUDA 算子，`pip install -e .` 要现场编译，需要 MSVC + nvcc 版本严格匹配" | **过重。** 两条退路：(a) transformers 已有**原生纯 PyTorch 版**——`GroundingDinoForObjectDetection`（2024-04-11 起）、`Sam2Model`（2025-08-14 起），完全绕开编译；(b) 即便用官方 sam2 仓库，其 README 明写 `Failed to build the SAM 2 CUDA extension during installation, you can ignore it and still use SAM 2`——该扩展只用于掩码连通域后处理。 | 🟠 影响 WSL 决策 |
| 3 | "Windows 上 `signal.SIGALRM` 不存在 → 原生跑不通 VADAR" | **这条本身正确**（已实测 `hasattr(signal,'SIGALRM')=False`），但它只堵住**执行 VADAR Engine** 这条路。我们自己写执行器，该约束归零。 | 🟢 结论仍成立但适用范围小得多 |
| 4 | "`vqa()` 是 CLEVR/GQA 专属模块" | **已更正过一次**：它在 Omni3D 分支里同样存在，且被写进给模型看的 API 文档。**主模型必须多模态。** | 🟠 关键约束（保留） |

**净效果：WSL2 从"必须"降级为"可选"。** 三处硬依赖（SIGALRM / 编译链 / 脚本平台假设）里，前两处随"自建实现"消失，第三处是 bash 脚本，Git Bash 能跑。整条栈可以在 Windows 原生跑通。

#### 更正 1 的源码证据

`unidepth/models/backbones/metadinov2/attention.py`（UniDepth 的 DINOv2 backbone 注意力层）：

```python
try:
    from xformers.ops import fmha, memory_efficient_attention, unbind
    XFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("xFormers not available")
    XFORMERS_AVAILABLE = False

XFORMERS_AVAILABLE = XFORMERS_AVAILABLE and torch.cuda.is_available()


class MemEffAttention(Attention):
    def forward(self, x: torch.Tensor, attn_bias=None) -> torch.Tensor:
        # new pytorch have good attn efficient, no need for xformers
        if not XFORMERS_AVAILABLE or x.device.type == "cpu":
            assert attn_bias is None, "xFormers is required for nested tensors usage"
            return super().forward(x)          # ← 回落路径

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = unbind(qkv, 2)
        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        ...
```

而父类 `Attention.forward` 用的是 `F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])` —— PyTorch 2.x 自带的高效注意力（本机 `torch 2.6.0+cu124` 在 4060 上已确认该函数可用）。

其余文件的核查结果：

| 文件 | xformers / triton |
|---|---|
| `unidepth/models/encoder.py` | 无（只 import torch / torch.nn / 本地 backbones） |
| `unidepth/models/unidepthv2/unidepthv2.py` | 无 |
| `unidepth/models/backbones/convnext.py` | 无（纯 torch + timm） |
| `unidepth/models/backbones/dinov2.py` | 无（只从 `.metadinov2` 导入） |
| `pyproject.toml` | `build-backend = "setuptools.build_meta"`，**无 `ext_modules`** → 纯 Python 安装 |

**结论：`requirements.txt` 里的 `triton>=2.4.0` / `xformers>=0.0.26` 是过度声明。** 装的时候可以不管它们（或装上但不生效），推理照跑。

#### 更正 2 的官方表述

- transformers 的 `GroundingDinoForObjectDetection` 文档注明该模型 *"contributed to Hugging Face Transformers on 2024-04-11"*，官方用法是 `AutoProcessor` + `AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")`
- transformers 的 `Sam2Model` 文档注明 *"contributed to Hugging Face Transformers on 2025-08-14"*，官方用法 `Sam2Model.from_pretrained("facebook/sam2.1-hiera-large")`，并提供 `pipeline("mask-generation", ...)`
- 官方 sam2 仓库 README：*"If you see a message like `Failed to build the SAM 2 CUDA extension` during installation, you can ignore it and still use SAM 2"*

---

## 4. 阻塞项逐条判定

| 阻塞项 | 性质 | 判定 | 依据 |
|---|---|---|---|
| `signal.SIGALRM` 不存在 | 平台 API | ✅ **可绕** | 自建执行器用子进程超时；不改 VADAR 时才是问题 |
| GroundingDINO 编译 | 构建链 | ✅ **可绕** | transformers 原生纯 PyTorch 版 |
| SAM2 编译 | 构建链 | ✅ **可绕** | 扩展可选（官方明示）+ transformers 原生版 |
| `xformers` / `triton` | 依赖声明 | ✅ **不存在** | 源码 try/except + SDPA 回落 |
| OpenAI 不可达 | 网络 | ✅ **可换** | 8 个国产端点实测 401（服务器已应答） |
| **`vqa()` 需要 VLM 不是 LLM** | **能力约束** | ⚠️ **不能绕，只能满足** | 主模型必须多模态；Qwen3-VL / DeepSeek-V4-Flash-Vision 满足 |
| 4B 模型能否驱动程序合成 | **能力未知** | ⚠️ **未验证，是最大风险** | 见第 9 节，这个必须尽早测 |
| VADAR 三段式零容错正则 | 设计缺陷 | ✅ **已绕开** | 不用它的 agent 层 |
| Omni3D-Bench 许可 | 法务 | ⚠️ CC BY-NC | 课程作业可用，不可商用 |
| `predefined_modules.py:19` 硬编码包名 `VADAR` | 命名约束 | ✅ **归零** | 不用它的代码就不受约束 |

**唯一真正的未知是"4B 级模型能不能守住这个协议"，不是环境能不能搭。** 这是好消息——未知项从"一堆环境问题"收敛成一个"跑一次就出答案"的实验。

---

## 5. 推荐路径：借骨架、换实现

### 5.1 保留什么（来自 VADAR 的骨架）

```
用户问题
   ↓
LLM 生成一段 Python 程序          ← 抄 VADAR 的动作空间（第 1.1 节）
   ↓
程序调用【固定工具库】              ← 替换 VADAR 的运行时 API 生成
   ├── 视觉基元：detect / segment / metric_depth
   └── 几何工具：get_3d_pos / distance / angle / nearest / above / left_of
   ↓
子进程执行（带超时）                ← 替换 VADAR 的 runpy + settrace + SIGALRM
   ↓
「观测」回灌给 LLM（可选多轮）       ← VADAR 没有这一环，我们补上
   ↓
最终答案 + 3D Viewer 高亮
```

### 5.2 新架构（相对 VADAR 的三处实质升级）

1. **几何接地**：一切空间量来自 `UniDepth.points` 的真实 3D 坐标，不由 LLM/VLM 判断
2. **固定工具库**：可复现、可评测、可做工具选择准确率
3. **观测闭环**：VADAR 是"生成即终局"，我们允许"执行 → 看结果 → 再生成"，这是从"程序合成"升级成"Agent"的那一步

### 5.3 新版技术栈（全部可在 Windows 原生运行）

| 层 | 组件 | 来源 | Windows | 4060 8GB | 许可 |
|---|---|---|---|---|---|
| 检测 | `IDEA-Research/grounding-dino-tiny` 或 SwinT-OGC | **transformers 原生** | ✅ 纯 torch | ✅ | Apache-2.0 |
| 检测（备选） | Qwen3-VL 直接输出 bbox | Ollama / API | ✅ | ✅ | Apache-2.0 |
| 分割 | `facebook/sam2.1-hiera-base-plus` | **transformers 原生** | ✅ 纯 torch | ✅ 降分辨率 | Apache-2.0 |
| 3D | `lpiccinelli/unidepth-v2-vits14` | pip（**纯 Python 安装**） | ✅ SDPA 回落 | ✅ 最轻 | CC BY-NC |
| 几何 | NumPy / SciPy | pip | ✅ | CPU | BSD |
| LLM（本地） | `qwen3.5:4b`（3.4 GB） | Ollama | ✅ | ✅ | Apache-2.0 |
| LLM（本地大） | `qwen3.5:9b`（6.6 GB） | Ollama | ✅ | ⚠️ 紧张 | Apache-2.0 |
| LLM（云端） | `deepseek-v4-flash-vision-exp` | API | ✅ | — | 见官方 |
| 执行器 | `multiprocessing` + 超时 | 标准库 | ✅ | — | — |
| Viewer | Rerun（调试）→ Three.js（答辩） | pip / npm | ✅ | — | MIT |

已验证的环境事实（本机实测）：

- `base` 环境：**Python 3.13.9，torch 2.6.0+cu124，`cuda.is_available()=True`，4060 sm_89，`F.scaled_dot_product_attention` 可用**
- `mytorch` 环境：Python 3.9.25 —— **过旧，不可用**（UniDepth 要求 ≥3.10，SAM2 要求 ≥3.10）
- `transformers` 尚未安装

→ 需要新建一个 **Python 3.11** 的干净环境（3.11 是 transformers + timm + UniDepth 三方兼容最稳的点）。

---

## 6. 三条路径对比与建议

| | 路径 A：修补 VADAR | **路径 B：借骨架自建（推荐）** | 路径 C：端到端 VLM 问答 |
|---|---|---|---|
| 做法 | 在 WSL 里跑原版，打补丁换 LLM | 抄动作空间设计，自己写工具库/执行器/Agent | 直接问 Qwen3-VL "哪个椅子离门最近" |
| 保留 VADAR 代码 | 大部分 | **只保留设计思想** | 无 |
| Windows 原生 | ❌ 需 WSL | ✅ | ✅ |
| 可复现性 | 差（动态 API） | **好（固定工具库）** | 中 |
| 4B 模型可行性 | ❌ 要扛三段零容错生成 | ✅ 一次生成 + 固定接口 | ✅ 最简单 |
| 创新空间 | 小 | **大（3 个创新点都在这）** | 几乎没有 |
| 答辩说服力 | "我复现了" | **"我做了真几何 + 真 Agent"** | "我调了个 API" |
| 定量实验 | 只能做换模型对比 | **能做 A–F 消融** | 只能测准确率 |
| 结论 | ❌ 推荐度低 | ✅ **推荐** | ⚠️ 可作为对照臂（Baseline 0） |

**建议走路径 B，同时把路径 C 保留为一个消融臂。**

理由：路径 C（端到端 VLM 直接答）恰好是你要反驳的那个"LLM 看图猜答案"的反面教材。**把它做成 Baseline 0 跑一遍，量出它的错误率，然后证明"程序合成 + 真几何"好多少** —— 这对答辩极有说服力，而且成本几乎为零（同一个 Ollama 端点，换一个 prompt）。

---

## 7. 新增一个实验臂：动作空间对比

既然第 1.1 节论证了"程序合成优于 JSON tool calling"，这个论断**必须用实验证明，不能只是论证**。所以消融表里应该新增一臂：

| 臂 | 动作空间 | 说明 |
|---|---|---|
| A′ | JSON tool calling（Qwen3-VL 原生 function calling） | 对照组 |
| B′ | 程序合成（固定工具库） | 实验组 |

**测什么**：多步任务的端到端成功率、平均往返次数、聚合类问题（"有几个…"、"哪个最近"）的准确率。

**预期**：在多步/聚合类问题上程序合成明显占优；在单步简单问题上 tool calling 可能持平。**这种"有条件的优势"比"全面碾压"更像真结论，也更难被答辩老师挑毛病。**

这个实验是免费的——同一个模型、同一套工具、同一个数据集，只换动作空间。工作量小、结论硬。

---

## 8. 对原方案的影响

### 8.1 阶段路线调整

| Phase | 原计划 | 调整后 |
|---|---|---|
| 0 环境 | WSL2 + Ubuntu-22.04 + 编译链 | **改为：Windows 原生 Python 3.11 venv + transformers + UniDepth**。WSL2 降为可选（只在想跑原版 VADAR 做对照时才需要） |
| 1 跑通原版 | 在 WSL 里跑 VADAR | **改：不跑 VADAR 原版**。改为跑通"我们自己最小 pipeline"（一张图 → 检测 → 3D 点 → 一个问题 → 程序生成 → 执行 → 答案） |
| 2 吃透架构 | 逐文件精读 VADAR | **改：已完成**（本次调研已做完），产出的是设计决策而非代码理解 |
| 3 接开源 LLM | 打补丁换 Generator | **改：不需要补丁**，直接对新栈写 OpenAI 兼容客户端 |
| 5–7 工具库/SceneGraph/Agent | 不变 | 不变，但**基础从 VADAR 的 Engine 换成自建执行器** |
| 新增 | — | **动作空间对比实验（第 7 节）** |

### 8.2 需要保留 VADAR 源码的理由

只为一个用途：**读它的 prompt 和工具签名设计作为参考**（`prompts/modules.py` 的 `MODULES_SIGNATURES` 已经把"空间问题需要哪些基元"梳理过一遍，有参考价值）。

`vendor/VADAR/` 保留，但**不再是构建依赖**。

### 8.3 不再需要的东西

- ❌ WSL2 + Ubuntu（可选）
- ❌ `phase0/00_install_wsl.ps1`、`phase0/01_setup_ubuntu.sh`（降级为"可选：跑原版对照用"）
- ❌ `phase0/03_vadar_llm_bridge.py`（运行时适配器——既然不用 VADAR 代码，就没有要适配的绑定）
- ❌ `phase0/02_probe_llm_api.py` 里的"用 VADAR 原版正则测标签"部分（正则不再是我们的契约）——**但它的国产端点连通性表仍然有效**
- ✅ 保留：`phase0/04_install_ollama.ps1`、`phase0/probe3d.py`（改一下 import 路径即可）

---

## 9. 唯一剩下的真问题

环境和路径问题全部解决后，**项目成败收敛到一个未被验证的能力问题**：

> **4B 级模型（`qwen3.5:4b`，3.4 GB）能不能守住"给定固定工具文档 → 生成一段可执行、语法正确、调用合法工具、结果正确"这个协议？**

为什么这是真问题：

- 这是一次**零容错的生成**。虽然有观测闭环可以补救，但首轮成功率直接决定迭代成本
- VADAR 用的是 gpt-4o 才拿到 40.4 分，能力差距是数量级的
- 如果答案是"完全不行"，则顺序要反过来：**先做 QLoRA 微调，再接模型**（微调不是锦上添花，而是可行性前提）——这会改变 Phase 3 和 Phase 9 的先后

怎么测（零成本、零依赖、不需要 GPU、不需要 API key）：

用**我们自己的固定工具文档** + 20 道空间问题，让 `qwen3.5:4b` 生成程序，用 **AST 静态检查**（不用真执行）评估四个维度：

| 维度 | 检查方式 |
|---|---|
| 语法正确 | `ast.parse()` 是否通过 |
| 工具是否合法 | 所有被调用的函数名是否都在白名单内 |
| 参数是否合法 | 是否用了不存在的参数名（捕捉幻觉） |
| 是否产出 `final_result` | 是否有变量赋值给约定名 |

这四项**只需 AST 就能判**，不需要装 torch、不需要跑视觉模型、不需要联网。**一次批量跑 20 道题，半小时内就能得到"4B 能不能用"的定量答案。**

---

## 10. 下一步

1. **建新环境**：Python 3.11 venv（D 盘），装 `transformers` + `torch`（已有 2.6.0+cu124 可复用）+ `timm` + `einops` + `unidepth`
2. **验证视觉栈**：改好的 `probe3d.py` 跑一次，确认真实显存/延迟，并打印 `points` / `rays` / `intrinsics`
3. **测 4B 能力**（关键）：用固定工具文档 + AST 检查器，得到首轮成功率
4. **根据 3 的结果决定 Phase 顺序**：成功率可接受 → 正常推进；不可接受 → 先做数据集合成 + QLoRA

**这三步都不需要 API key，不需要 WSL，不需要租卡。**
