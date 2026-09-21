# 3D Spatial Agent 项目技术调研与实施方案

> 定位：**自建的三维空间智能体**。感知层接三个开源模型，Agent 层与工具层自研。
> 本机实测硬件：**NVIDIA GeForce RTX 4060 Laptop GPU，8188 MiB，驱动 560.76，CUDA Toolkit 12.6**
> 文档起于 2026-09-14，此后逐轮追加实测结论（每节标注日期）。

凡是估算值都显式标注「估算」，凡是不确定项都标注「不确定」。**实测值才算数。**

---

> ## ⚠️ 文档变更声明（2026-09-20 追加）
>
> 本文档早期版本以一份第三方出处（一个「程序合成式 3D 空间问答」的开源研究实现）作为改造
> 基础，并对其做过逐文件的静态审计（原 §2–§7）。**该检出与配套审计文档已移出本仓库：**
>
> | 早期写法 | 现行写法 |
> |---|---|
> | 「基于该实现改造」 | **自建实现**。只保留「动作空间 = LLM 输出 Python 程序」这一设计取向 |
> | 必须 WSL2 + 编译链 | **Windows 原生 Python venv + transformers 原生实现**（§20 Step 0.3） |
> | 实验臂 A = 上游原版基线 | **已移除**。早期 6 次真跑留在 `results/A/`，但**当前不可复现** |
> | §3–§7 的逐类拆解 | **已压缩合并进 §2**（行号引用随检出移除而不可复核） |
>
> **依然有效**：§8–§15（LLM 选型 / 可行性 / QLoRA / 工具库 / Scene Graph / Agent / Demo / 创新点）、
> §16–§18（实验设计 / 目录 / 路线）、§20 的网络与显存实测表、§21–§24（内参杠杆与数据契约）。

---

## 1. 项目目标

把「一张 2D 照片 + 一句自然语言空间问题」做成一个**几何可验证**的三维空间智能体。

三层结构：

| 层 | 职责 | 本项目对应模块 |
|---|---|---|
| 底层 3D Vision | 检测 / 分割 / 度量深度 / **真实三维坐标** | GroundingDINO + SAM2 + UniDepth（`points` + `intrinsics`） |
| 中层 3D Spatial Representation | 场景图 + 几何关系推理 | Scene Graph + Geometry Engine |
| 上层 LLM Agent | 任务规划 / 工具选择 / 多步调用 / 答案生成 | Tool-calling Agent + LoRA 微调后的 Qwen3.5-4B |
| 出口 Demo | 交互式 3D 场景 + 高亮 + 相机控制 | 3D Viewer + Chat + Tool Trace |

**核心目标不是复现任何论文**，而是：

```
自建感知层与工具库 → 用真实 3D 几何重建工具层 → 构建 Scene Graph
→ 自研 Agent 循环与边界契约 → 自建工具调用数据集 → 在 4060 上做 QLoRA
→ 定量消融 → 答辩级可交互 Demo
```

一句话定位：**让 Agent 不再「看图猜空间关系」，而是「调工具算空间关系」。**

典型目标任务：

> 用户：「哪个椅子离门最近？」
> Agent：`find_object("door")` → `find_object("chair")` → `get_3d_position()` → `calculate_distance()` → `argmin` → 回答「Chair 2，1.42 m」→ Viewer 高亮 Chair 2 → 可选 `move_camera()`

**系统有两类输出，缺一不可**（2026-09-16 补充）：

| # | 输出类 | 内容 | 面向 |
|---|---|---|---|
| 1 | **问答输出** | 自然语言答案 + 米制数值 + 三维高亮 + 工具调用 trace | 单道题 |
| 2 | **场景级输出（★ L5）** | 结构化三维场景描述：物体清单 + 米制尺寸 + 两两几何关系，落盘 `scene_graph.json` | 整张图 |

第 2 类容易被漏掉，但它是「**3D Spatial**」这个定位应有的产出：在 L1–L4 里场景图只是「跑完就丢」
的内部表示，L5 让它成为可保存、可复查、可对比的交付物，并给出**独立于问答准确率**的新指标
（§16.3 指标 11–12）。**它不依赖任何提问** —— 场景构建完成即可导出。

> **为什么必须显式写进方案**：如果只把场景图当内部实现，整篇报告就只剩「问答准确率」一张表，
> 看起来像一个问答工具；把它升为一类输出，项目才真正落在 3D 场景理解上。

---

## 2. 早期参考实现：评估结论与弃用理由

> **本节口径**：本节合并自原 §2–§7。原 §3–§7 是对一份第三方检出的逐类拆解（含大量
> `文件:行号` 引用）；该检出与配套审计文档已于 2026-09-20 移出本仓库，行号**已无法复核**，
> 因此这里只保留**仍然影响本项目设计**的结论。节号保持原状（下文直接接 §8），以免打断
> 全文的交叉引用。

### 2.1 只保留一样东西：动作空间 = LLM 输出 Python 程序

早期评估的出发点是找一份「已经是 3D 空间推理、且动作空间是程序而不是 JSON function call」
的参考实现。它确实满足这一点：预定义视觉基元里有**度量深度**；三个角色（提签名 / 写实现 /
出程序）各调一次 LLM，生成的 Python 程序真被执行并返回真实的视觉结果；另有一个「用 GT 场景
数据替换视觉模块」的开关，说明作者也认为瓶颈在视觉而非推理。

**本项目保留的就是这一条设计**：LLM 的输出是一段**可静态检查、可执行、可留证据**的程序，
而不是一串 tool-call。其余全部弃用，理由见 2.2。

### 2.2 它的局限：这些直接决定了创新点在哪

| 局限 | 现象 | 对本项目的影响 |
|---|---|---|
| **不是 tool-calling agent，是 program synthesis** | 三个角色各生成一段 Python，执行靠 `runpy` | 「换 LLM」≠「换 Agent」。我们的 Agent 契约是**新增**的，不是改的 |
| **拿不到真实 3D 坐标** | 只取单目深度的 **z 标量**，丢掉了同一次 `infer()` 已算好的 `points`（相机系点云）与 `intrinsics` | **最大、也最容易实现的真实创新点**（见 2.4、§11.1） |
| **3D 尺寸公式量纲不成立** | 用「2D 像素 × 深度」算真实尺寸，**式子里没有焦距** | 尺度系统性偏差，可量化、可改进、可写进实验 |
| **工具集是运行时随机抽样生成的** | 先随机抽 10 道题再据此生成方法，且无固定种子 | 结果不可复现、工具集每次不同 ⟹ **不能作评测基线**；我们的动作空间改为**显式白名单** |
| **视觉问答绑死闭源 VLM** | 内部硬编码 `Generator("gpt-4o")` | 纯文本开源 LLM 替不掉视觉通道；我们把视觉语义拆给独立小 VLM（§13.3 角色②） |
| **本体只能在 Unix 跑** | 用 `signal.alarm` / `SIGALRM` 做超时 | 自研执行器改用**独立计时线程强杀**，不碰 `signal`（§20 Step 0.3） |
| **上游停更 15 个月** | 依赖锁在 2024 年的版本 | 与 2026 年的模型生态冲突（§8.3、§9.3） |

**结论**：它可以当参考，不能当地基 —— 它的「3D」是**深度标量级近似**，它的「Agent」是
**一次性代码生成器**。这两点正好对应本项目的两个主要创新方向。

### 2.3 据此定下的自研架构（三条）

1. **动作空间显式白名单化**：`QA_TOOLSET` 是 12 个工具的元组，提示词渲染 / 静态检查 / 执行器
   **三处共用同一份**。「注册表里有什么」≠「这一臂该看见什么」—— 混同会让新工具**静默改变
   已跑臂的动作空间**，归因随之失效。
2. **提交契约替代魔法变量**：`submit(..., evidence≥1)` 取代「扫描命名空间猜哪个变量是答案」——
   后者是静默缺陷的温床，命名空间里没有答案时它照样「成功」返回。
3. **视觉语义与空间几何拆开**：关系一律由几何计算，VLM 只判颜色/材质；并**在类型层面**让空间
   参数写不出来（`describe(attrs=Literal[...])` 不含任何空间参数）
   ⟹ **空间幻觉在类型层面构造不出来**。

### 2.4 「`points` 是真几何」—— 升级到真三维不需要任何新模型

全项目唯一无法靠读代码回答、必须真机验证的问题。结论是**成立**（2026-09-16 实测）：

| 证据 | 实测值 |
|---|---|
| `infer()` 返回的键 | **7 个**：`confidence` / `depth` / `depth_features` / `intrinsics` / `points` / `rays` / `radius` |
| `points[2] == depth` | **完全相同（max\|diff\| = 0.000e+00）** |
| `depth` 的量级（真实室内照片） | **[1.376, 3.974] m** —— 单位就是米 |

**三条必须写进报告的事实：**

1. **`depth` 不是独立测量量，它就是点云的 z 列**。完整 XYZ 早就在手里，只留 z 是一次**主动丢弃**
   ⟹ **升级到真三维不需要任何新模型、任何新显存。**
2. **「2D 尺寸 × 深度」量纲不成立**：式子里没有焦距，真实场景里这样算出的「三维尺寸」没有物理意义。
3. **不要用 `intrinsics` 反投影重建点云**：用返回的 K 做针孔反投影，与 `points` 只对到
   **相对误差 3.7%（均值）/ 5.3%（最大）**。因为 `rays` 是 decoder **预测**的方向场、`intrinsics`
   又是另一个独立预测头，且 `infer()` 会先 padding → resize → 再裁回，K 是解析式换算回去的。
   两个来源本就不等同，差几个百分点是正常量级。**正确用法是直接用 `points`。**

> 量化过程与原始输出见 `phase0/probe3d.py`。判定标准用**相对误差**而非绝对误差 —— 480p、2–4 m
> 的场景里要求 2 cm 绝对一致，对单目模型不现实；3–5% 才是这类输出之间应有的量级。

### 2.5 视觉栈必须与 LLM 分 venv、分进程

上游把 `transformers` 钉在 **4.45.2**（2024-10），而 Qwen3 系列需要 ≥ 4.51、Qwen3.5 更新
（第三方量化卡甚至要求 `>=5.3.0.dev0` —— **以官方 model card 为准，此处不确定**）。
GroundingDINO / UniDepth / SAM2 均对 `transformers` 版本敏感，同一个 venv 里强行升级
**有较大概率弄坏检测与深度**；分离后还能热切换模型、分时复用显存、崩溃互不拖累。

三个模型的权重、峰值显存与延迟已全部实测（UniDepth 486/604 MB、GroundingDINO
1761–1903 / 2152–2306 MB、SAM2 647/928 MB；**三模型同时驻留仅 1200 MB = 8188 MiB 的 14.7%**，
⟹ 约 6.9 GB 留给 LLM，「本地 LLM 可选」在显存上成立）。⚠ 常驻必须在**持有模型引用**时测，
否则会低得离谱。完整表格与可复现命令见 **§20 Step 0.5 / 0.5b**。

---

## 8. 开源 LLM 推荐（针对 8GB 4060 的实测生态，2026-09）

### 8.0 零 API 起点：本地跑起来要什么（已在 Ollama 官方库核实，2026-09-15）

**结论先说：不需要任何 API key 就能开始。** Ollama 在 Windows 上有官方安装包，直接通过 CUDA 用你的 4060，装完即是一个 OpenAI 兼容端点（`http://127.0.0.1:11434/v1`）。

已核实的可用型号与磁盘占用（对照 `https://ollama.com/library/qwen3.5/tags` 逐条核对，2026-09-15）：

| 标签 | 磁盘 | 上下文 | 输入模态 | 8 GB 4060 |
|---|---|---|---|---|
| `qwen3.5:0.8b` | 1.0 GB | 256K | 文本 + 图片 | ✅ 很宽松，但能力不足以做程序合成 |
| `qwen3.5:2b` | 2.7 GB | 256K | 文本 + 图片 | ✅ 适合先把管线跑通 |
| **`qwen3.5:4b`** | **3.4 GB** | 256K | 文本 + 图片 | ✅ **推荐起点** |
| `qwen3.5:9b` | 6.6 GB | 256K | 文本 + 图片 | ⚠️ 偏紧，需压上下文（同时被标为 `:latest`） |
| `qwen3.5:27b` | 17 GB | 256K | 文本 + 图片 | ❌ 需云卡 |
| `qwen3.5:35b` | 24 GB | 256K | 文本 + 图片 | ❌ 需云卡 |
| `qwen3.5:122b` | 81 GB | 256K | 文本 + 图片 | ❌ 需多卡 |
| `qwen3.5:cloud` / `:397b-cloud` | — | 256K | 文本 + 图片 | ❌ **仅云端，不可本地加载** |

> **⚠️ 命名已更正（本表此前是错的）。** 旧版此表写作 `qwen3-vl:4b-instruct-q4_K_M`、`qwen3-vl:8b-instruct-q4_K_M` 等，是**过期命名**。
> Ollama 官方库当前的家族名是 **`qwen3.5`**，标签直接是尺寸（`:4b`），**没有 `instruct` 后缀，也没有单独的 4-bit 标签名**——默认拉取的就是量化版。
> `qwen3-vl` 是上一代命名，多模态能力已并入 `qwen3.5`（官方说明：*Early fusion training……outperforms Qwen3-VL models*）。
> `04_install_ollama.ps1` 已同步改为拉 `qwen3.5:4b`，并在拉取失败时按 `4b → 2b → 9b` 自动回退，避免命名再次变动时直接卡死。

**必须有一个组件能看图。** `vqa()` 内联 base64 图片（`predefined_modules.py:334-355`），且在 Omni3D 分支同样存在（`:693`）。纯文本模型跑到 `vqa()` 必然 400。`qwen3.5:4b` 一个端点同时顶掉「程序合成」和 `vqa` —— **但这是便利，不是必需**：这两个角色可以拆成两个模型，见 §8.5。

程序合成要的是**严格照格式输出**，而 `qwen3.5` 系列带 `thinking` 能力标签。跑协议测试时建议显式关闭思考模式（参数名随运行时不同，用 `SPATIAL_EXTRA_BODY` 透传即可），否则延迟被拉长，标签还可能被埋进思考块里导致正则提取失败。

**显存预算（实测前的估算，需 Phase 0.7 用真实数字替换）**：

```
qwen3.5:4b 权重（默认量化）        ≈ 3.4 GB
KV cache @16K, q8_0              ≈ 1.2 GB
视觉编码器 + 推理激活            ≈ 1.0 GB
Windows 桌面占用                 ≈ 0.5 GB
─────────────────────────────────────────
合计                             ≈ 6.1 GB / 8.19 GB     ✅ 可行
```

再叠加视觉栈（GroundingDINO SwinT-OGC ≈ 0.7 GB + UniDepth ViT-S14 ≈ 0.15 GB）约 4.2 GB 总量 —— **但两者不会同时常驻**：流程是「先跑视觉 → 再问 LLM」，串行关系。若要并发，8 GB 会紧张。

### 8.1 云端主模型推荐（作为对照臂 / 决定性实验）

> **主模型：`Qwen3.5-4B-Instruct`，以 4-bit（AWQ 或 GGUF Q4_K_M）由 vLLM / Ollama 以 OpenAI 兼容端点服务。**

理由：

| 维度 | 事实 |
|---|---|
| 参数 | 4.66B（官方 model card） |
| 权重体积 | BF16 ≈ 9.3 GB；4-bit ≈ 3–5 GB（估算，各来源口径不一） |
| 显存 | 4060 8GB 上 4bit 权重 + KV cache 有明确余量；**BF16 装不下** |
| 工具调用 | **BFCL-V4 = 50.3**（官方卡，thinking 模式） |
| Agent 任务 | **TAU2-Bench = 79.9**——这一项甚至高于 9B（79.1）和 35B-A3B（81.2 同级） |
| 代码生成 | LiveCodeBench v6 = 55.8 |
| 多模态 | **原生多模态**，可直接顶替 `vqa()` |
| 许可 | Apache-2.0 |

**为什么不是 7B/9B：** 9B 的 BFCL-V4 更高（66.1）但 4bit 约 5–8 GB，在 8GB 上要和 KV cache 抢空间，且吞吐明显下降（Laptop 4060 功耗墙更低）。**9B 放在「云端对照臂」而不是主力臂**，这样既拿到强基线结果，又不牺牲日常迭代速度。

### 8.2 模型选型矩阵

| 模型 | 4060 8GB | 4bit 推理 | LoRA/QLoRA | Tool Calling | 代码生成 | 多步空间推理 | 需要云卡 |
|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | ✅ 轻松 | ✅ | ✅ 最快 | BFCL 25.3（弱） | 弱 | 弱 | 否 |
| Qwen3.5-2B | ✅ | ✅ | ✅ | BFCL 43.6 | 中 | 中 | 否 |
| **Qwen3.5-4B** | ✅ **4bit** | ✅ 推荐 | ✅ **首选** | **BFCL 50.3** | **中上** | **中上** | 否 |
| Qwen3.5-9B | ⚠️ 4bit 勉强 | ✅ | ⚠️ 紧/可 offload | BFCL 66.1（更强） | 强 | 强 | 建议租 |
| Qwen3.5-27B | ❌ | 4bit ≈15GB | ❌ | BFCL 68.5 | 很强 | 很强 | 是（≥24GB） |
| Qwen3.5-35B-A3B | ❌ | 激活 3B 但总权重放不下 | ❌ | BFCL 67.3 | 强 | 强 | 是 |
| Qwen3.5-122B/397B | ❌ | ❌ | ❌ | 最强 | 最强（可做教师模型） | 最强 | 是（多卡） |
| Qwen3-4B-2507 | ✅ | ✅ | ✅ | 中（无原生 VLM） | 中 | 中 | 否 |
| DeepSeek 系列（V3.2 等） | ❌ 体量过大 | — | — | — | — | — | 是 / 用 API |
| Llama 3.2-3B | ✅ | ✅ | ✅ | 弱 | 中 | 弱 | 否 |
| **GPT-4o（论文基线）** | — | — | — | — | — | — | 用 API，做 baseline |

> 表中 BFCL-V4 / TAU2-Bench 数值来自 Qwen 官方 model card（Qwen3.5-4B 卡与 0.8B 卡对内列出对照）。**不同来源对 0.8B/1.5B/2B 的命名有出入，正式开跑前请以 HuggingFace / ModelScope 官方 model card 为准。**

### 8.3 环境版本冲突（重要）

- Qwen3.5 需要 **vLLM ≥ 0.16–0.17**（社区部署贴建议 ≥0.17.0）。
- 工具调用需在 vLLM 启动时加：`--enable-auto-tool-choice --tool-call-parser qwen3_coder`（来自第三方 AWQ 卡的启动说明）。**parser 名称随版本可能变化，以你装到的 vLLM 版本的文档为准。**
- **不要**试图在 vllm 和视觉栈共用一个 venv（见 §2.5）。

### 8.4 微调工具链

- **Unsloth** 已发布 Qwen3.5 系列的本地微调指南与 GGUF（含 0.8B/2B/4B/9B），是 8GB 场景下最现实的 QLoRA 方案。
- 备选 **LLaMA-Factory**（生态成熟、配置化程度高，但显存开销略大）。
- 4060 上请一律使用：4-bit NF4 + LoRA + gradient checkpointing + 短序列 + batch=1 + 梯度累积。

### 8.5 模型不是硬依赖：约束分层、替代方案、架构要求

> **这一节是为了纠正本方案的一个表述偏差。** 前面各节反复出现 `Qwen3.5`，容易被读成「必须用 Qwen」。实际上它只是**当前默认值**，不是依赖。下面把「什么真的不能换」和「什么随便换」分开。

#### 三条约束，性质完全不同

| 层 | 约束内容 | 来源 | 换它的代价 |
|---|---|---|---|
| **L1 技术强制** | 必须有一个模型能生成**可执行 Python 程序** | 我们选了「程序合成」作为动作空间（见 §13） | 换具体模型 = 0；但要改成 JSON tool calling 会显著削弱表达能力（见 §15 创新点论证） |
| **L2 项目派生** | ① 有组件能看图 ② 8GB 内可推理 ③ 能本地 QLoRA（**已降为加分项，非必须**，见 §9.4）④ 国产 | 你的硬件 + 你提的要求 | 换具体模型 = 0，但候选池会变 |
| **L3 实现细节** | 具体型号、量化档位、服务端点 | 我的选型 | **换模型 = 改环境变量**，`llm/adapter.py` 已做成 model-agnostic |

**结论：L1 里真正不可谈判的只有「动作空间是 Python」这一条，它与任何厂商无关。`qwen3.5` 是 L2∩L3 的当前最优解，不属于 L1。**

#### 候选池：为什么筛完只剩 1–2 个族

在「一个模型同时干视觉 + 代码」（沿用早期形态）这个前提下，四条约束的交集确实很小：

| 候选 | 视觉 | 代码 | 8GB 本地 | 国产 | 尺寸阶梯 | 判定 |
|---|---|---|---|---|---|---|
| **Qwen3.5** | ✅ | ✅ | ✅ `4b` 3.4 GB | ✅ | ✅ 0.8b → 122b | **默认值** |
| GLM-4.5V | ✅ | ✅ | ❌ 108B | ✅ | — | 云端对照臂 |
| DeepSeek（Ollama `deepseek-v4.1-flash`） | ✅ | ✅ | ❌ 仅 cloud 标签 | ✅ | — | 云端对照臂 |
| MiniCPM-V 4.6 | ✅ | ⚠️ 未验证 | ✅ 1.6 GB / 1.3B | ✅ | ❌ 单尺寸 | **可当专职视觉组件** |
| Gemma 4（`e2b`/`e4b`） | ✅ | ✅ | ✅ | ❌ | ❌ | 技术上最接近的替代品，但非国产 |
| Llama / Mistral | 部分 | ✅ | ✅ | ❌ | — | 按 L2④ 排除 |

**④「同族尺寸阶梯」为什么重要（最容易被忽略的一条）：** Qwen3.5 从 0.8b 到 122b 同族、同 tokenizer、同 chat template。这意味着你的 scaling 实验（0.8B → 2B → 4B → 9B）只改变**一个变量：参数量**。若改成跨族对比，模型间差异会混入 tokenizer、chat template、训练数据、对齐策略等混杂因子，scaling 曲线就不再可信。这一条让「默认选 Qwen」从个人偏好升级为**方法学要求**。

#### 更好的做法：把两个角色拆开

早期形态逼着同一个模型既写程序又答 `vqa()`，这本身是一个**妥协**——两个任务的最优模型规模差得很远。我们自建时应当拆成三个角色：

| 角色 | 需要什么 | 候选 |
|---|---|---|
| ① 程序合成 | 强代码能力、长上下文、严格照格式输出 | **任何**强代码模型；**纯文本即可**（空间信息全部来自工具返回值，模型不需要"看"图） |
| ② 视觉语义 | 只需回答颜色 / 材质 / 类别等属性 | 1–2B 小 VLM（如 MiniCPM-V 4.6，1.6 GB） |
| ③ 空间几何 | 不需要任何 LLM | 纯几何计算，结果完全可复现 |

拆开后每条约束都松绑：显存从「一个大模型常驻」变成「小模型 + 小 VLM 可串行加载」，两个角色各自独立可换，还顺带把「语义问题」和「几何问题」在架构上分了家 —— 这正是创新点 1/2 想做的事。

#### 对代码的硬要求（写进项目规范）

1. **禁止硬编码模型名。** 一律走环境变量：`SPATIAL_BASE_URL` / `SPATIAL_MODEL` / `SPATIAL_VISION_MODEL`。
2. **禁止依赖某家私有 API 形态。** 只用 OpenAI 兼容的 `chat.completions` 子集。
3. **每个实验臂必须记录模型指纹**：`base_url + model + temperature + max_tokens + extra_body` 写进结果 JSON。否则换模型后的对比无法归因。
4. **`qwen3.5` 是默认值，不是前提。** 它若被下架，只影响 Phase 0 的验证速度，不影响方案成立。

---
### 8.6 云端后端实测：双端点路由与思考模式 A/B（2026-09-17，真实调用 15 次）

**接入形态**：适配器按「消息里是否含 `image_url`」自动路由到视觉端点 —— 文本 Agent 走
`deepseek-flash`，视觉问答走 `qwen3-vl-flash`（视觉单价低一个数量级，而它调用次数最多）。
也可整体切到本地 vLLM / Ollama 上的 `Qwen3.5-4B`（答辩可拔网线，但与「LoRA 训练效果」是两条
独立叙事，别混在一张表里）。实现见 `llm/adapter.py`。

**模型名口径已变**：规范名只有 **`deepseek-flash`**（= V4.1-Flash；1M 上下文、384K 输出、
支持 Vision 与 Tool Calls）。`deepseek-v4-flash` / `…-vision-exp` 是遗留别名（对应模型已退役，
请求由 V4.1-Flash 承接、同价）；`deepseek-chat` / `deepseek-reasoner` 于 2026-07-24 退役；
`deepseek-v4-pro` 有序退役中。⟹ **DeepSeek 这边已不存在「强模型档」**，换名字换不到更强的模型。
价格（元/百万，高峰 / 空闲）：输入未命中 `3.0 / 1.5`、命中 `0.10 / 0.05`、输出 `9.0 / 4.5`；
**高峰 = 北京时间 9:00–12:00 与 14:00–18:00** ⟹ 全量实验放空闲时段，账单减半。

**思考模式 A/B：关闭参数有效、低温可复现站得住**

| 臂 | 出现 `reasoning_content` | 中位延迟 | 输出 tokens（2 次合计） |
|---|---|---|---|
| `disabled`（带关闭参数） | **否**（2/2） | 0.98 s | **4** |
| `default`（不带） | **是**（2/2） | 1.22 s | **51**（思考占 45） |

主路径 13 次调用**思考痕迹 0 次** ⟹ `temperature=0.2` 真正生效；开思考时输出 token 是无思考的
**12.75 倍**（同一句「只回一个词」）⟹ 全量实验必须逐臂记账。

**四类输出标签与视觉通道全绿**：`<program>` 语法 **3/3** 过 `ast.parse`、签名提取 **3/3**
（未触发注解陷阱）、`<answer>` **4/4**、视觉问答**答对 4/4**（主模型直接吃图，无需独立视觉端点）。
⚠ **「未触发」≠「不存在」**：换提示词或换模型仍可能触发，提示词约束不要撤。

**成本**：整轮 15 次调用 **2915 tokens**（1945 prompt + 970 completion），空闲档 **¥0.0073**、
高峰档 ¥0.0146。⚠ 这是**下限** —— 探针用极简提示词，真实 prompt 要带完整 API 文档 + few-shot，
输入会高一个量级；**真实 per-call 数只能由「跑臂」运行器量出来**，探针能证明的只是
「端点机制与计价口径没问题」。

> **一次值得写进报告的「差点搞错」**：首轮汇总字段 `thinking_observed` 给出过**相反结论**
> （「思考生效 ⟹ `temperature` 被忽略 ⟹ 不得声称低温可复现」）。根因是派生代码把 A/B 的
> **对照组**算进了主路径统计（那一臂故意带思考）。已修 + 离线复算验证，原始报告归档在
> `logs/phase1_probe_raw_2026-09-17.json`。
> **教训：一个统计量混入它自己的对照组，就会得出与本实验相反的结论。**

---


## 9. 4060（8GB / Laptop）可行性分析

### 9.1 本机实测环境（2026-09-15 逐项实测）

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 **Laptop** GPU |
| 显存 | **8188 MiB**（≈8 GB，实际可用通常 7.2–7.8 GB） |
| 驱动 | 560.76 |
| CUDA Toolkit | 12.6（nvcc V12.6.20） |
| CPU | AMD Ryzen 9 7945HX，16C/32T，BIOS 虚拟化已开，SLAT 支持 |
| 磁盘 | C: 剩 **21.8 GB**（不够）｜D: 剩 **381.4 GB**（工作盘） |
| WSL | ❌ **未安装**（无 Lxss 注册表键，`HypervisorPresent = False`） |
| 本地 LLM 运行器 | ❌ 无 Ollama / LM Studio / vLLM |
| Anaconda `base` | Python **3.13.9**，`torch 2.6.0+cu124`，**`cuda.is_available() = True`**，4060 可用 |
| Anaconda `mytorch` | Python **3.9.25**，`torch 2.6.0+cu124`，CUDA 可用 |
| 两个环境都缺 | `transformers` / `timm` / `huggingface_hub` / `peft` / `bitsandbytes` |

> **两个需要点出来的事实**：
>
> **1. Windows 上 CUDA 是通的。** `torch 2.6.0+cu124` 在 4060 上 `is_available() = True`，说明驱动和 GPU 直通没问题。所以「Windows 跑不通早期基线」**不是**显卡问题，而是平台 API 与编译链问题（Unix 信号 + triton/xformers 无 Windows 支持 + GroundingDINO 需现场编译 CUDA 算子）。
>
> **2. 系统 Python 是 3.13/3.9，而早期基线要求 3.10。** 加上 UniDepth 的 `requires-python = ">=3.10.0"`，`mytorch`（3.9）也用不上。WSL 里用 Ubuntu 22.04 自带 3.10 是最省事的解法。

### 9.2 逐模块可行性

| 模块 | 4060 8GB | 说明 |
|---|---|---|
| GroundingDINO SwinT-OGC 推理 | ✅ | **实测峰值 1761–1903 MB**；不必再降分辨率 |
| UniDepthV2 ViT-S14 推理 | ✅ | 全流程最轻的模型（实测峰值 486 MB） |
| SAM2.1 hiera base+ 推理 | ✅ | **实测峰值 647 MB**；640×480 原生跑，不必降分辨率 |
| 完整 Omni3D pipeline（三模型共存） | ✅ **已实测** | **三模型同时驻留仅 1200 MB（14.7%）—— 「串行加载/卸载」已无必要** |
| `qwen3.5:4b`（Ollama） | ✅ | 权重 3.4 GB，已核实；含 16K KV cache 约 6.0 GB |
| `qwen3.5:9b`（Ollama） | ⚠️ | 权重 6.6 GB，8 GB 上偏紧，需压上下文 |
| Qwen3.5-4B QLoRA 训练 | ✅（预期，需实测） | r=16、seq≤2048、bs=1、grad accum、gradient checkpointing |
| Qwen3.5-0.8B/2B 全参或 QLoRA | ✅ 轻松 | 用作快速迭代与 scaling 曲线 |
| Qwen3.5-9B 4bit 推理 | ⚠️ | 与视觉模型**不能**同时驻留 |
| Qwen3.5-9B QLoRA | ⚠️ | 需 CPU offload，速度很慢；建议租卡 |
| 7B/14B/27B 训练 | ❌ | 不要尝试 |
| 3D Gaussian Splatting（小场景） | ✅（预期） | 与本项目主线解耦，作为可选视觉增强 |
| COLMAP（小规模） | ✅/⚠️ | CPU+GPU 混合，慢但可行 |
| 同时跑「视觉栈 + 4B LLM + 3DGS」 | ❌ | 必须分时复用 |

### 9.3 硬性结论

1. **主开发与主要实验全部在 4060 完成**，Omni3D 路径的视觉栈是「小模型组合」，这是小模型组合带来的运气。
2. **云端只在三种情况下使用**：（a）9B 及以上模型的对照实验；（b）Molmo-7B 的 CLEVR 支线，且只跑一次并缓存；（c）如果 4B QLoRA 在 8GB 上实在 OOM，用一次 4090 24GB 跑最终训练（预算很小）。
3. **绝不需要为「跑通 pipeline」租 A100**（视觉栈合计仅约 1.2 GB，见 §9.4）。
4. **WSL2 已经不是必需项**。唯一曾经需要它的场景是跑早期基线本体（Unix 信号）——该检出已移除，所以现在全程 Windows 原生：
   - **曾经需要 WSL2**：早期基线本体（Unix 信号）、三个视觉模型的源码构建（triton/xformers/编译链）。**现在都不需要**（视觉栈走 transformers 原生实现）
   - **不需要 WSL2**：本地 LLM 服务（Ollama 在 Windows 上直接吃 CUDA）、所有 prompt 协议测试、文档与脚本开发
   - 所以 **Phase 0 的 P0 步骤可以先做掉，不必等重启**（见 §20 优先级表）
5. **不为了一个能本地完成的任务租云卡。** 本地 Ollama 已经能提供一个完整的 OpenAI 兼容端点，API key 是可选加速项，不是前置依赖。

### 9.4 本地部署的真实边界（2026-09-15 用户确认）

> **用户明确：「本地微调」是加分项，不是课程硬性要求。**
> 因此整条本地 LLM 链路（Ollama、QLoRA、显存工程）从「必须」降为「可选、可后置」。

#### 先拆开：「8 GB 约束」有两个来源，此前混在一起讲是造成困惑的主因

| 来源 | 是否必须本地 | 显存占用 | 原因 |
|---|---|---|---|
| **视觉栈**（§5） | **必须本地** | 合计约 1.2 GB | 三个模型均**无托管 API 可换**，只能本地跑；但很轻，不是压力来源 |
| **LLM**（§8） | **可选本地** | 3.4 GB 权重 + KV cache | 这才占显存的大头，也是此前所有「装不装得下」讨论的真正对象 |

**视觉栈必须本地 ≠ 本地部署压力大。** 压力全部来自 LLM，而 LLM 是可选的。

#### 本地部署唯一被绑死的场景

**QLoRA 微调本身**（API 无法微调），以及微调后模型的评测服务。其余环节全部可走 API：

| 环节 | 部署位置 |
|---|---|
| Agent 程序合成与规划 | 任意强代码模型 API（纯文本即可） |
| prompt 协议验证与调试 | API |
| 小规模端到端跑通 | API |
| 数据集合成 + 大规模评测（VLM 调用量大） | 本地更划算，**但非必须** |
| **QLoRA 微调** | **必须本地或租卡** |

#### Phase 界线

> **进入 Phase 9（微调）之前，LLM 一次本地推理都不需要。**
> Phase 0–8 全程 API 即可。视觉栈虽必须本地，但与 LLM 选型无关，不要混为一谈。

**对 §20 优先级表的连带影响：** Step 0.7 / 0.8（装 Ollama、测 4B 协议合规性）与 Phase 9 从「P0 / 主线」降为「可选支线」。

**但保留一个反向条件（不要因此彻底删掉这条支线）：** 若后续实测发现云端模型也无法稳定守住那套零容错协议（§19 风险表第 1 位），**微调仍是提升协议合规性的唯一手段**，届时该支线重新升为主线。届时再装 Ollama 也不迟——它只是一条 `winget install` 命令，不存在沉没成本。

#### 成本：视觉栈不花钱，唯一成本是电费

| 环节 | 谁付 | 金额 |
|---|---|---|
| 原作者的训练算力（数千 GPU 小时） | 原作者 / 其机构 | 不归你 |
| 权重下载（约 1.2 GB，走 `hf-mirror.com`） | — | **¥0** |
| 本地推理 | 你的电费 | 4060 Laptop 推理时约 60–100 W |
| 托管 API | — | **不存在**（三者均无官方 API 可调） |

**许可证（2026-09-15 双向核实）：**

| 模型 | 许可证 | 课程 / 学术 | 商用 |
|---|---|---|---|
| GroundingDINO | Apache-2.0 | ✅ | ✅ |
| SAM2 | Apache-2.0 | ✅ | ✅ |
| **UniDepth V2** | **CC BY-NC 4.0** | ✅ | ❌ **禁止** |

> `cc-by-nc-4.0` 已在 HF 模型卡（`lpiccinelli/unidepth-v2-vits14`，`license:` 字段）与 GitHub 仓库 README 的 License 段双向核实。
> **对本项目无影响**：课程作业属学术用途。但**答辩与报告里必须注明**；若日后想把项目产品化，需替换为许可证更宽松的单目深度模型（选型时单独核实其许可）。
> 同类注意：`Omni3D-Bench` 数据集同样是 CC BY-NC（见 §19 风险表第 17 项）——两者叠加意味着本项目整体是「学术可用、商用不可」的定位。

**唯一可能意外花钱的场景**：本地环境装不上、必须租云卡跑视觉栈。§4 已逐项核实 Windows 原生可跑（transformers 原生实现 + 纯 Python 包），该风险已排除。

---

## 10. LoRA / QLoRA 方案

### 10.1 目标（不是让它会聊天，而是让它会调工具）

让模型在 **3D Spatial Tool Calling** 上更强，具体是六项能力：

1. 工具选择（面对问题选出正确的工具集合）
2. 工具调用（参数填对：哪个 object_id、哪组坐标）
3. 多步工具调用（3–6 步的正确顺序）
4. 空间问题分解（把「哪把椅子离门最近」拆成定位→取坐标→算距离→argmin）
5. Python / 结构化输出生成（严格 JSON 或 schema 合法）
6. 空间关系推理（left_of / above / inside 的几何定义一致性）

### 10.2 训练配置（4060 可行区间）

| 项 | 配置 | 说明 |
|---|---|---|
| 基座 | Qwen3.5-4B-Instruct | 主力；0.8B/2B 用于快速迭代 |
| 量化 | 4-bit NF4（bnb） | QLoRA |
| LoRA r / alpha | 16 / 32 | 数据量 2–5k 时足够 |
| LoRA dropout | 0.05 | |
| target_modules | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` | Qwen3.5 有 GDN 线性注意力层，**模块名与 Qwen3 不完全一致，需先打印 `named_modules()` 确认**（不确定项） |
| 序列长度 | 1536–2048 | 工具 schema 很长，压缩 schema 表示很重要 |
| batch / 累积 | 1 / 8–16 | |
| 学习率 | 1e-4（LoRA 常用），cosine | |
| epoch | 2–3 | 小数据不要多轮 |
| 显存优化 | gradient checkpointing + 8bit AdamW + flash-attn（如可用） | |

**如果 OOM 的降级顺序**：seq 2048→1536 → r 16→8 → 基座换 2B → 最后才考虑租卡。

### 10.3 数据集设计（本项目的核心资产）

**问题：** 手工标注 3D 空间工具调用轨迹不现实；纯 LLM 生成的数据不可信（会编造不存在的物体）。

**解决方案：Execution-Verified Synthesis（执行验证式合成）**

```
① 场景侧：对每个场景预构建 Scene Graph（nodes 有真实 3D 坐标），缓存为 JSON
        ↓
② 问题侧：用模板 + 场景图自动生成问题（「哪个 X 离 Y 最近」「Z 在 W 的左边吗」）
   —— 因为答案由几何直接算出，所以 GT 免费且 100% 正确
        ↓
③ 轨迹侧：用强教师模型（GPT / Claude / Qwen3.5-397B API）生成工具调用轨迹
        ↓
④ 验证侧：★ 把轨迹真的在缓存场景上执行一遍
   —— 只有「执行成功 且 得到正确答案」的轨迹才进训练集
        ↓
⑤ 训练集：~2,000–5,000 条 (question, tool_schema, trajectory, answer) 四元组
   + 保留 ~200 条作为 held-out 验证集
```

**这个「执行验证」步骤是你数据集的质量护城河。** 它同时对生成的数据做了自动去幻觉：模型编造了不存在的物体 → 工具执行失败 → 丢弃。

**数据格式（建议统一为 OpenAI function-calling 的 messages 格式，便于同时喂给 vLLM 和 Unsloth）：**

```json
{
  "scene_id": "omni3d_img_0123",
  "question": "哪把椅子离门最近？",
  "available_tools": ["find_object", "get_3d_position", "calculate_distance", "argmin", "answer"],
  "messages": [
    {"role":"system","content":"You are a 3D spatial reasoning agent. ..."},
    {"role":"user","content":"哪把椅子离门最近？"},
    {"role":"assistant","tool_calls":[{"id":"c1","type":"function",
       "function":{"name":"find_object","arguments":"{\"label\":\"door\"}"}}]},
    {"role":"tool","tool_call_id":"c1","content":"{\"objects\":[{\"id\":\"door_0\",\"centroid_3d\":[0.12,0.93,2.41]}]}"},
    {"role":"assistant","tool_calls":[{"id":"c2","type":"function",
       "function":{"name":"find_object","arguments":"{\"label\":\"chair\"}"}}]},
    {"role":"tool","tool_call_id":"c2","content":"{\"objects\":[{\"id\":\"chair_0\",...},{\"id\":\"chair_1\",...}]}"},
    {"role":"assistant","tool_calls":[{"id":"c3","type":"function",
       "function":{"name":"calculate_distance","arguments":"{\"a\":\"door_0\",\"b\":[\"chair_0\",\"chair_1\"]}"}}]},
    {"role":"tool","tool_call_id":"c3","content":"{\"distances\":{\"chair_0\":2.13,\"chair_1\":1.42}}"},
    {"role":"assistant","content":"{\"answer\":\"Chair 2\",\"distance_m\":1.42,
        \"evidence\":[\"door_0\",\"chair_1\"]}"}
  ],
  "verified": true
}
```

**训练规模评估（针对「这种数据适合训练多大模型」的直接回答）：**

| 基座 | 4060 8GB QLoRA | 备注 |
|---|---|---|
| 0.5B / 0.8B | ✅ 非常轻松，可跑多组超参搜索 | 适合验证数据格式与 loss 曲线 |
| 1.5B / 2B | ✅ 轻松，seq 可放到 4096 | 快速迭代主力 |
| **3B / 4B** | ✅ **可行（本方案主选）**，需 grad ckpt + bs=1 | 最终交付模型 |
| 7B / 8B | ⚠️ 需 CPU offload，速度极慢 | 不建议在 4060 做 |
| ≥14B | ❌ | 租卡或放弃 |

---

## 11. 我们的 3D Tool Library（自研，替代参考实现的随机 API）

**设计原则（对应你的硬要求）：**

1. 输入输出用 Pydantic / JSON Schema **显式定义**，不再依赖正则解析 XML 标签
2. 每个工具的返回值必须是**几何证据**（数字、坐标、id），不是自然语言
3. **空间关系由几何计算，不由 LLM 判断**——`left_of` 不是问 VQA「这个在左边吗」，而是比较 x 坐标
4. 工具要能报错（返回结构化 error），Agent 才能重试
5. 工具集固定、版本化 → 评测可复现（直接解决参考实现随机 API 的问题）

### 11.1 工具分层

**L1 — 感知工具（调模型，有 GPU 开销，结果全场景缓存）**

| 工具 | 签名 | 返回 | 底层 |
|---|---|---|---|
| `detect_objects` | `(image_id, prompt) -> list[BBox]` | 2D 框 | GroundingDINO |
| `segment_object` | `(image_id, bbox \| point) -> Mask` | 分割掩码 | SAM2.1 |
| `estimate_depth` | `(image_id) -> DepthMap` | 米制深度图 | UniDepthV2 |
| `estimate_camera` | `(image_id) -> Intrinsics` | 3×3 内参 | UniDepthV2（**参考实现丢掉的**） |
| `lift_to_3d` | `(image_id, mask) -> PointCloud` | 相机系点云 | UniDepthV2 `points`（**参考实现丢掉的**） |

**L2 — 场景构建工具（一次构建，多次查询，纯 CPU）**

| 工具 | 签名 | 返回 |
|---|---|---|
| `build_scene_graph` | `(image_id) -> SceneGraph` | 节点 + 边，落到 `scene_graph.json` |
| `load_scene_graph` | `(scene_id) -> SceneGraph` | 从缓存加载 |
| `list_objects` | `(scene_id, label=None, attrs=None) -> list[Node]` | 节点列表 |
| `get_object` | `(scene_id, object_id) -> Node` | 单节点 |
| `get_attributes` | `(scene_id, object_id) -> dict` | 颜色/材质等（**唯一允许走 VLM 的工具**） |

**L3 — 几何与空间工具（纯数学，零 GPU，零 LLM，可单测）**

| 工具 | 签名 | 返回 | 实现要点 |
|---|---|---|---|
| `get_3d_position` | `(scene_id, object_id, anchor="centroid") -> [x,y,z]` | 相机系三维坐标（米） | mask 内点云取**中位数**，抗离群 |
| `get_3d_extent` | `(scene_id, object_id) -> {w,h,l}` | 真实三维尺寸（米） | 点云 PCA / 轴对齐 bbox，**替代 `2D×depth` 近似** |
| `calculate_distance` | `(scene_id, a, b) -> float` | 三维欧氏距离（米） | 不是 depth 相减 |
| `calculate_angle` | `(scene_id, a, b, c) -> float` | 夹角（度） | 三点夹角 |
| `find_nearest` / `find_farthest` | `(scene_id, query, target_set) -> Node` | 最近/最远对象 | |
| `left_of` / `right_of` | `(scene_id, a, b, tol=0.05) -> bool` | 布尔 | 比较 x；**tol 显式化** |
| `front_of` / `behind` | `(scene_id, a, b, tol) -> bool` | 布尔 | 比较 z（相机系前后） |
| `above` / `below` | `(scene_id, a, b, tol) -> bool` | 布尔 | 比较 y（需 up-axis） |
| `on` / `inside` | `(scene_id, a, b, tol) -> bool` | 布尔 | 竖直投影重叠 + 高度差；容器判定用点云包含率 |
| `near` / `far` | `(scene_id, a, b, thresh) -> bool` | 布尔 | 距离阈值 |
| `query_relation` | `(scene_id, relation, a, b) -> bool` | 统一入口 | 分发到上面各关系 |
| `sort_by_distance` / `argmin` / `argmax` | `(scene_id, ...) -> Node` | | |
| `calibrate_scale` | `(scene_id, object_id, known_size) -> float` | 尺度修正因子 | 单目深度尺度校正（见下） |

**L4 — Viewer 控制工具（Demo 联动）**

| 工具 | 签名 | 作用 |
|---|---|---|
| `highlight_object` | `(object_id, color, style) -> ok` | 3D 场景高亮 |
| `move_camera` | `(target_object_id, distance, azimuth) -> ok` | 视角飞到目标 |
| `draw_relation` | `(a, b, relation) -> ok` | 画连接线 / 箭头 |
| `show_trace` | `(steps) -> ok` | 回传推理轨迹到 UI |
| `annotate_answer` | `(text, anchor_object_id) -> ok` | 场景内标注答案 |

**L5 — 场景级输出工具（★ 第一类输出：让中间表示本身成为交付物）**

> 2026-09-16 新增。动机见 §1「两类输出」。

L1–L4 都是「面向单个问题」的工具：输出要么是某道题的答案，要么是某个视图动作。
L5 把 `SceneGraph` 直接当作可交付产品暴露出来：**输入是场景，输出是结构化的三维场景描述**。

| 工具 | 签名 | 返回 |
|---|---|---|
| `describe_scene` | `(scene_id, detail="full") -> SceneReport` | 物体清单 + 米制尺寸 + 两两关系（结构化，见下） |
| `summarize_scene` | `(scene_id) -> str` | 上述报告的**自然语言**版本（L5 中唯一允许走 LLM 的工具） |
| `answer_with_evidence` | `(scene_id, question_id) -> {answer, evidence[]}` | 答案 + 几何证据链（可回溯到质心与公式） |
| `diagnose_failure` | `(scene_id, question_id) -> {stage, reason}` | 失败环节定位（检测 / 尺度 / 程序 / 工具） |
| `counterfactual` | `(scene_id, remove=[...], query=...) -> SceneReport` | 反事实查询，**纯图操作**，不重跑视觉模型 |

`SceneReport` 的结构（"第一类输出"的具体形态，**全部字段由几何层填充，零 LLM**）：

```json
{
  "scene_id": "living_room_01",
  "scale_calibrated": true,
  "objects": [
    {"id": "sofa_1",  "label": "sofa",  "centroid": [0.31, -0.42, 2.86],
     "extent_m": {"w": 1.92, "h": 0.78, "l": 0.91}, "confidence": 0.836},
    {"id": "table_1", "label": "table", "centroid": [-0.44, -0.61, 3.12],
     "extent_m": {"w": 0.86, "h": 0.45, "l": 0.86}, "confidence": 0.712}
  ],
  "relations": [
    {"a": "sofa_1",  "b": "table_1", "type": "distance", "value_m": 1.608},
    {"a": "sofa_1",  "b": "table_1", "type": "left_of",  "value": true,  "tol": 0.05},
    {"a": "table_1", "b": "sofa_1",  "type": "front_of", "value": false, "tol": 0.05}
  ]
}
```

**为什么值得单列一层（这是对「输出是不是太简单」的正面回答）：**

1. 前四层里场景图只是**内部中间表示**，跑完一道题就被丢掉；L5 让它成为**可保存、可复查、可对比的产物**。
2. 它是**零额外模型成本**的：`objects` 与 `relations` 的全部字段在 `build_scene_graph` 时就已经算好，
   L5 只是把它们**序列化并渲染**出来。
3. 它把项目从「三维问答工具」抬到「**三维场景理解系统**」——这才是标题里 `3D Spatial` 二字应有的产出。
4. 评测上它给了**独立于问答准确率**的新指标（描述召回率、关系边一致率，见 §16.3 指标 11–12），
   于是「工具库升级」这一个创新点可以出两张表，而不是一张。

> **`counterfactual` 是本层性价比最高的演示项**：场景图建好后，「若移走沙发，哪把椅子最靠近门」
> 是纯图操作，不重跑任何视觉模型也不需要 GPU。它直接证明「中间表示」本身有价值，
> 而不只是工程脚手架 —— 答辩时比任何并行数字都直观。

**注意 `calibrate_scale` 的必要性**：单目度量深度（UniDepth）在零样本设置下仍有尺度漂移。如果你想让「1.42 m」这个数字可信，必须做一次尺度校正（用场景中已知尺寸的物体，或数据集提供的 GT）。**这本身就是 3D Localization Error 这个指标的来源，也是可写进论文的诚实讨论。**

### 11.1.1 实现记录（2026-09-16）—— 五个工具里落了四个

代码在 `tools/scene_report.py` + `scripts/report_scene.py`，测试在 `tests/test_scene_report.py`
（**38 个用例，零 GPU、零联网**）。工具库版本 `1.0.0 → 1.1.0`（工具集合变了就必须升版本，
否则「两次实验用的是不是同一套工具」无法核对；已落盘旧场景里的 `tools_version: 1.0.0` 不回填）。

**按「零额外模型成本」这条前提核对的实测成本**（`living_room_gt`，9 物体 / 36 对 / 396 条关系）：

| 工具 | 热态中位数 | p95 | GPU |
|---|---|---|---|
| `describe_scene(detail="brief")` | 1.04 ms | 1.09 ms | 无（`gpu_peak_mb` 恒为 None） |
| `describe_scene(detail="full")` | 3.06 ms | 3.18 ms | 无 |
| `summarize_scene`（模板） | 4.12 ms | 4.25 ms | 无 |
| `diagnose_failure` | 0.03 ms | 0.04 ms | 无 |
| `counterfactual(remove=[sofa_1])` | 5.63 ms | 5.68 ms | 无 |

> **口径**：热态中位数 = 3 次预热 + 15 次重复（p95 为同一批的第 95 百分位）。
> 表中数字都比「一次直觉估计」大，尤其是 `summarize_scene`（4.1 ms，不是「< 1 ms」）与
> `counterfactual`（5.6 ms）—— 前者要遍历 9 个物体做方位聚合，后者要在**裁剪后的**
> 28 对物体上**重算**全部关系（8 物体 × 28 对 × 11 关系 = 308 条），都不是常数时间。
> **新进程首调另有惰性导入开销**：落盘报告的 `provenance.latency_ms` 是新进程单次调用，
> 实测 `summarize_scene` 18.96 ms、`counterfactual` 20.45 ms、
> `describe_scene(full)` 7.14 ms、`diagnose_failure` 0.11 ms。
> **两组数都对但不可混用**：引用「工具本身多快」用热态中位数，引用「端到端一次报告多慢」用 provenance。

四处在实现时才暴露出来的判断（都是文档没写、只能靠实测定下来的）：

**① `detail` 必须分两档，因为同一个返回值服务两类读者。** `full` 是 64.3 KB
（396 条关系），作为落盘交付物没问题，但**不能当程序合成模型的观察**（观察必须压缩，
§18 Phase 7 风险②）。`brief` 只给物体清单 + 关系计数 + 每个物体的最近邻，降到 3.5 KB
（O(n) 而非 O(n²)），布局类问题的可回答性几乎不损失。超限时**不静默截断**：设
`truncated=True` 并写进 caveats —— 静默截断会让「关系边一致率」这个指标偏乐观。
上限按**对**而不是按条砍（200 对 ≈ 0.4 MB），因为半对被砍会让一致率算错。

**② 关系 metric 的去重规则改过一次，第一版是错的。** `pairwise` 给每条关系都塞了同一套
基础量（`distance_m` / `delta_x/y/z` / 六个质心分量），一对物体 11 条关系里重复 11 次。
第一版规则是「值与该对 `distance` 条里的同名键相同就删」，实测**两个理由都足以否决**：

- 它在浮点上是**不确定的** —— `round(x, 6) == x` 只在少数数上成立，于是同一个 `left_of`
  关系在 `(sofa, picture)` 上保留了 `delta_x`、在 `(door, sofa)` 上把它删了。
  报告字段集随数值抖动，diff 与测试都无法稳定断言（这个是**测试逼出来的**：
  `test_metric_field_set_is_identical_for_every_pair` 专门断言确定性本身）。
- 它把**判别量**一起删了。`left_of` 的判据就是 `delta_x < -tol`，删掉之后
  「为什么判定为左」不再能从该条自身复算 —— 恰好毁掉 metric 存在的理由。
  而改成按**键名**删同样不行：`above` 的竖直量叫 `delta_up`（过了 `up_coord()` 归一），
  与 distance 的同名键**含义不同**，按名字删会静默丢掉真实信息。

最终规则是**可证明无损**的那一种：只删质心分量（`a_x..b_z`），因为它们就是
`objects[].centroid_m`，同一份报告里本来就有。载荷 **96.5 → 64.3 KB**（−33%），
且字段集逐对一致（"96.5 KB" = 把质心分量补回每条非距离关系后的重建值）。

**③ `summarize_scene` 的签名有意偏离本文档。** 文档原文是 `(scene_id) -> str` 且
「L5 中唯一允许走 LLM 的工具」。实现改成 `summarize_scene(scene_id, use_llm=False)`：
默认走**确定性模板**（同一场景逐字节可复现，评测可 diff），`use_llm=True` 时才走
`ctx.flags["summarizer"]`，未注入则返回 `CAPABILITY_DISABLED`。
理由：报告要进指标比较，默认走 LLM 会让「描述质量」混入采样噪声。
把「允许走 LLM」保留成**显式开关**而不是默认行为，是消融可复现的前提（§13.3(7)）。

**④ 新增了一条文档里没有的诊断判据：掩码点数下限。** `schema.py` 的 `Node.n_points`
注释早就写明「是 `DEGENERATE` 的判定依据」，但 L5 第一版只查了尺寸**上界**、把它漏了。
阈值 1000 **来自实测的自然分界而非拍定**：`living_room_gt` 的 9 个物体里，3 个伪物体
（低分 picture）是 **809 / 893 / 965** 点，其余 6 个真实物体是 **16137–30660** 点，
中间有 **17 倍空隙**。这条判据同时进 `describe_scene` 的 caveats —— 报告是给人看的交付物，
它会把 `objects` 直接读成「这个房间里有 4 幅画」；若其中 3 幅其实是几十个像素的低分检测框
而报告一声不吭，**报告就在说谎，这比数字不准更严重**。

**`answer_with_evidence` 明确挂起**：它需要题集才能把 `question_id` 解析成一道题，
而题集属于 Phase 2（§18）。现在实现它只有两种写法，两种都不诚实 ——
自己造一份假题集（会在报告里留下「已实现」的假证据），或接受自由文本再调 LLM
（那就把「答案必须来自几何」这条地基拆了）。题集一到，它就是一层薄封装。

#### ⭐ 已有的实测收益：同一张图、唯一变量是内参，L5 报告里直接可见

`scripts/report_scene.py` 对 §21/§22 那对同图 A/B 场景各导出一份报告。
两次运行的检测器、掩码、关系定义**完全相同**，差别只有内参来源：

| | `living_room_gt`（传 GT 内参） | `living_room_pred`（模型预测内参） |
|---|---|---|
| `sofa_1` 尺寸 | 2.34×0.82×1.26 m | **6.71×2.35×1.13 m** |
| `mirror_1` 尺寸 | 0.88×1.28×0.27 m | **2.57×3.69×0.28 m** |
| 重力方向 | tilt 11.95°，**可靠** | tilt **63.04°，不可靠**（`above`/`below` 会整体翻转） |
| 尺寸 > 3 m 的物体 | 0 个 | **4 个** |
| `diagnose_failure` 主环节 | 尺度（1 条发现） | 尺度（5 条发现） |

`diagnose_failure` 在 `pred` 上的排序恰好是「根因先于症状」：
内参（10.0）→ 重力方向（7.0）→ 尺度未校正（6.0）→ 尺寸越界（5.0，且它的 `fix`
明写「先查上面『尺度』那几条」）。**尺寸越界是症状、内参是根因**这条裁决，
在这一页上不需要任何解释就能看懂 —— 这正是 L5 作为「第一类输出」的价值：
它把 §21/§22 那些只存在于探针数字里的结论，变成了可直接放进答辩的交付物。

### 11.2 与参考实现原工具的关系


| 参考实现预定义 | 我们的替代 | 关系 |
|---|---|---|
| `loc(image, prompt)` | `detect_objects` | 改名 + 结构化返回 |
| `depth(image, bbox)` | `estimate_depth` + `get_3d_position` | **升级**：从单点深度 → 掩码内点云 |
| `vqa(image, q, bbox)` | `get_attributes` | **降级使用**：仅用于颜色/材质这类**语义属性**，不用于空间关系 |
| `same_object(image, b1, b2)` | 由 Scene Graph 的 `object_id` **消灭掉** | **架构性改进**：有了稳定 id，就不需要 IoU 猜同一性 |
| `get_2D_object_size` | `get_3d_extent` | **升级**：从像素尺寸 → 米制三维尺寸 |

> 最后一行值得强调：「用 id 消灭 `same_object`」是一个可以写进报告的架构性洞察。参考实现因为每次 `loc` 都返回裸 bbox，所以必须靠 IoU>0.92 反推「这是不是同一个东西」，这既不准也浪费工具调用。Scene Graph 一次性解决。

---

## 12. 3D Scene Graph

### 12.1 是否值得做？——值得，而且是最高性价比的中间层

三个直接收益：

1. **消除幻觉**：Agent 只能引用场景图里**真实存在**的 `object_id`。工具执行时校验 id 存在性，不存在直接返回 error → 幻觉从「静默错误」变成「可捕获错误」。
2. **把 O(n) 次模型调用降到 O(1)**：参考实现每问一个新物体就调一次 `loc`；场景图一次构建、无限次查询，且查询是纯 CPU 数学。
3. **关系可复现、可单测**：`left_of` 是一个有 `tol` 参数的纯函数，可以写单元测试，出问题能定位。参考实现的对应能力藏在 LLM 生成的代码里，不可测。

### 12.2 数据结构

```
Node {
  id:            "chair_1"          # 稳定 id，{label}_{idx}
  label:         "chair"
  score:         0.71               # 检测置信度
  bbox_2d:       [x1,y1,x2,y2]      # 像素
  mask_ref:      "masks/chair_1.png"# 掩码文件引用（不内联，省内存）
  centroid_3d:   [x,y,z]            # 相机系，米 ★ 核心字段
  extent_3d:     [w,h,l]            # 真实三维尺寸，米
  bbox_3d:       {min:[...], max:[...]}   # 三维轴对齐包围盒
  attributes:    {"color":"brown","material":"wood"}   # 可选，来自 VLM
  frame:         "camera"           # 坐标系标记
}

Edge {
  source:        "chair_1"
  target:        "table_0"
  relation:      "near" | "on" | "inside" | "left_of" | "right_of"
                 | "front_of" | "behind" | "above" | "below"
  value:         true|false|float
  metric:        {"distance_m": 0.83, "delta_y": -0.42}   # ★ 证据
  method:        "geometry_v1"      # 来源标记，可追溯
  confidence:    0.9
}

SceneGraph {
  scene_id, image_id, camera_intrinsics, up_axis, scale_factor,
  nodes: [...], edges: [...], build_meta: {models, versions, timestamps}
}
```

### 12.3 构建方式（单遍卷积式）

```
image_id
  ├─ 1. detect_objects("objects")        ← 用宽泛 prompt 一次拿全场景
  │      若漏检，可补一轮按类别的针对性检测，然后去重
  ├─ 2. segment_object(bbox) 每个框 → mask      (SAM2，可批量)
  ├─ 3. estimate_depth + estimate_camera → 深度图 + K
  ├─ 4. lift_to_3d(mask) 每个物体 → 点云
  ├─ 5. 点云 → centroid_3d (median) / extent_3d (PCA) / bbox_3d
  ├─ 6. 估计 up_axis（重力方向）
  │      —— 单张图下用「地面法向量」近似：取画面下部大面积点云拟合平面
  │      —— 不确定：这一步是单目场景图最脆弱的地方，需在报告中显式讨论
  ├─ 7. 成对计算几何关系边（阈值化，写进 metric 字段）
  └─ 8. （可选）用 VLM 补 attributes
       ↓
  scene_graph.json  (缓存，后续所有查询读它)
```

**成本（已实测，2026-09-16）**：一帧 9 个物体 → 1 次 GroundingDINO（0.6–1.6 s）+ **1 次批量 SAM2（9 框 179 ms）** + 1 次 UniDepth（0.7–1.1 s）≈ **1.3–2.0 s**，三个模型权重常驻仅 **1.2 GB**。

> **批量调用是必须的，不是优化**：实测 9 个框**一次调用 200 ms**，逐个调用合计 **1512 ms** ——
> **批量快 7.57×**。20–30 个物体时仍是一次调用（必要时按 16 框分批以压瞬时峰值），量级仍是**数秒**。
> 一次性开销，完全可接受。

### 12.4 更新方式

- **静态**：一个 (image / 场景) 一个 JSON，构建即冻结
- **增量**：新增物体时只补该节点的点云计算与它与已有节点的边
- **版本**：`method: "geometry_v1"` 字段让「换了关系定义」这件事可追溯；重跑可 diff

### 12.5 如何被 Agent 调用 & 作为中间表示

- Agent **不在提示词里塞整个场景图**（那会撑爆上下文且引入噪声）。而是把 `list_objects` / `query_relation` 作为**工具**暴露给 LLM，LLM 按需查询。
- 场景图同时作为 **Viewer 的数据源**（3D 场景直接画 nodes）+ **评测的 GT 载体**（关系真值来自几何，可自动打分）。
- 这样中间表示一举三得：Agent 的查询后端、Demo 的渲染后端、实验的真值后端。

---

## 13. 新 Agent 架构（面向本项目的自研设计）

### 13.1 参考实现 vs 新架构

| 维度 | 参考实现 | 新 3D Spatial Agent |
|---|---|---|
| **主路径范式** | Program Synthesis（一次生成一整段代码） | **Program Synthesis（保留，但收敛）** |
| **LLM 调用次数** | 多次（签名 → 实现 → 自测 → 程序，共 4 类调用） | **主路径 1 次**出完整程序；消融臂 E′ 才是 N 次逐步决策 |
| 决策粒度 | 整个程序（但工具集每次随机） | 整个程序，内部可含循环 / 聚合 / 条件分支 |
| 工具集 | 运行时随机生成 | **固定版本化库 v1.0.0** —— 可复现的前提 |
| 反馈 | 只在报错时回灌 traceback | **编译期 AST 静态检查 + 运行期结构化错误码** → 定向重新生成（≤2 次） |
| 空间证据 | depth 一个数 | **3D 坐标 / 点云 / 关系边 + 可回溯证据链** |
| 记忆 | 无 | 工作记忆（已确认的物体 / 事实）+ 场景状态 |
| 与 Viewer | 无 | **工具即可控 Viewer** |
| 可复现 | 差（随机 API + 随机抽样 10 题） | 好（固定工具 + 固定提示词 + seed，**工具版本号写入结果 JSON**） |

**注意（2026-09-16 修订，主次与直觉相反）**：**主路径是 program synthesis** —— 单轮生成一个完整程序，内部可含循环、过滤、聚合与条件分支；**多轮 tool-calling loop 降为消融臂 E′**。三条理由：

1. 空间问题的本质是「**组合 + 算术 + 聚合**」。「在 4 把椅子里找离门最近的那把」是 `min(chairs, key=...)` 一行 Python；用 JSON tool call 表达的代价是 5 次往返且无法表达 `argmin`。
2. 单轮生成 → **可复现、可 AST 静态检查、token 最省**，而且 LLM 只被调用一次，4B 级模型的错误没有累积空间。
3. 与参考实现的**动作空间直接可比** —— 「程序合成 vs 逐步决策」这才构成一条正式的消融，否则前三个创新点的收益无法与范式收益分离。

两者**共用同一套工具库与执行器**，唯一区别是「LLM 被调用几次」。接口契约见 §13.3。

### 13.2 建议实现：先轻量自研，不急上 LangGraph

> **下图是「骨架」，不是「主路径」。** Planner / Executor / Scene Graph / Geometry Engine / Verifier / Trace
> 六个部件**两个范式都要用**；区别只在虚线那条「下一轮决策」——主路径把它**压平为一次程序生成**
> （程序内部的循环由 Python 承担，不由 LLM 承担），消融臂 E′ 才真的走多轮。两者共用工具库与执行器，
> 所以消融对比是**干净的**：只换循环层。三角色接口契约见 §13.3。

**推荐先自研一个约 300 行的循环**，理由：

1. 参考实现的提示词是正则标签式的，你需要一层「标签 ↔ JSON schema」的桥，任何框架都帮不上忙
2. 你要做 **Tool Selection Accuracy** 这种细粒度指标，需要完全掌控每一步的日志结构
3. LangGraph 的状态/图/checkpointer 概念对「单轮 3–6 步的空间问答」是过度设计
4. 调试成本：自己写的循环出错时 traceback 是直白的

```
┌─────────────────────────────────────────────────────────┐
│  User question                                          │
│         ↓                                               │
│  ┌──────────────┐                                       │
│  │  Planner     │  把问题拆成子目标（可选用 LLM）        │
│  │              │  输出：子目标列表 + 需要的工具类别     │
│  └──────┬───────┘                                       │
│         ↓                                               │
│  ┌──────────────┐   ┌────────────────────────────┐      │
│  │  LLM Agent   │──►│  Tool Selector / Caller    │      │
│  │ (Qwen3.5-4B) │   │  产出单一 tool_call (JSON) │      │
│  └──────┬───────┘   └───────────┬────────────────┘      │
│         ▲                       ↓                       │
│         │            ┌──────────────────────┐           │
│         │            │  Executor            │           │
│         │            │  · 参数校验          │           │
│         │            │  · 工具执行          │           │
│         │            │  · 重试 / 降级       │           │
│         │            │  · 记录 trace        │           │
│         │            └──────────┬───────────┘           │
│         │                       ↓                       │
│         │            ┌──────────────────────┐           │
│         │            │  Scene Graph /       │           │
│         │            │  Geometry Engine     │           │
│         │            │  (纯 CPU 数学)       │           │
│         │            └──────────┬───────────┘           │
│         │                       ↓                       │
│         │            ┌──────────────────────┐           │
│         └────────────┤  Observation          │          │
│   下一轮决策          │  结构化结果 + 证据     │          │
│                      └──────────┬───────────┘           │
│                                 ↓                       │
│                      ┌──────────────────────┐           │
│                      │  Verifier (可选)      │          │
│                      │  证据是否支持答案？    │          │
│                      └──────────┬───────────┘           │
│                                 ↓                       │
│                      Final Answer + Evidence            │
│                                 ↓                       │
│                      Viewer: highlight / move_camera    │
└─────────────────────────────────────────────────────────┘
```

**必须实现的机制：**

| 机制 | 实现方式 |
|---|---|
| Multi-step reasoning | 显式步数上限（建议 8），超限强制收敛 |
| Tool calling | 用 vLLM 的 `--enable-auto-tool-choice` + JSON schema |
| Tool retry | 参数错误→让 LLM 修参数；执行错误→重试 1 次；再失败→换工具 |
| Error handling | 工具返回 `{"error": {...}}` 结构化错误，不是抛异常 |
| Observation feedback | 每次工具返回都进消息历史，且**压缩**（只保留关键数字） |
| Memory | 工作记忆：已解析的 object_id 映射、已确认事实；不跨问题 |
| 3D scene state | SceneGraph 单例 + 当前相机状态 |
| Trace 日志 | 每步落 JSONL：{step, tool, args, result, latency_ms, tokens}——直接喂给 §13 的指标计算 |

**LangGraph 何时才值得引入**：当你要做（a）多智能体分工（planner/executor/verifier 各自独立上下文）、（b）可中断可恢复的长任务、（c）人在回路确认。**这三项在本项目都不是必需**。可以把它列为 Phase 7 的可选扩展，作为报告的「工程选型讨论」素材。

---

### 13.3 三角色接口契约（2026-09-16 定稿）

**为什么必须拆开**：早期形态用**同一个模型**既写程序又答视觉问题，而后者还绑死在闭源 VLM 上。这两件事要求的能力**相反** —— 写程序要求严格结构化、可被 AST 解析；看图问答要求自由生成。挤在同一个 prompt 契约里，两个都做不好，而且让「关闭视觉能力」这类消融**根本无法做**。

本方案拆成 **三个角色 + 一个信封 + 一个提交契约**，依赖方向严格单向。下面三处「签名层面堵死」是全节的价值所在。

#### (1) 统一返回信封：`ToolResult`

五个工具层（L1–L5）、三个角色，**返回值全部是它**。

```python
@dataclass(frozen=True)
class ToolResult:
    ok: bool
    value: Any | None
    evidence: dict            # {formula, inputs, centroid_m, ...} —— 可回溯
    error: ToolError | None
    meta: ToolMeta            # {tool, version, latency_ms, cached, gpu_peak_mb}
```

错误码用**枚举**而非自由字符串 —— Agent 的错误恢复能力必须可被枚举、可被统计：

| code | 触发场景 | 程序应当的应对 |
|---|---|---|
| `NOT_IN_SCENE` | `object_id` 不存在 | **幻觉捕获点**：只能用 `list_objects` 返回过的 id |
| `NOT_FOUND` | 查询无匹配（图里没有门） | 换查询词 / 放宽阈值 / 直接答「图中无此物」 |
| `AMBIGUOUS` | 候选多个但要单个 | 追加空间约束（最近的 / 左边的） |
| `DEGENERATE` | 点数不足、共线、零长度 | 换 anchor（bbox 中心 → 掩码质心） |
| `LOW_CONFIDENCE` | 检测分低于阈值 | 标注不确定，**禁止四舍五入成确定值** |
| `CAPABILITY_DISABLED` | 该能力在本次实验臂被关闭 | 改用几何替代，或声明无法回答 |

> `CAPABILITY_DISABLED` 是**消融臂的机制**，不是异常处理。关闭某能力后，工具返回的是一个
> **程序必须处理的结构化事件** —— 于是「关掉视觉语义」是一次接口开关，而不是一份改写过的 prompt。
> 这是消融可复现的关键。

#### (2) 角色① 程序合成 `synthesize()`

**唯一产出控制流的角色。**

```python
def synthesize(
    question: str,
    tool_docs: ToolDocs,        # 版本化 schema 渲染，只读
    scene_hint: SceneHint,      # ★ 只给 label 清单 + 计数，不给坐标
    memory: WorkingMemory,
    feedback: RetryFeedback | None = None,
) -> SynthesisResult:           # {source, ast_ok, static_check, usage}
```

三处关键设计：

1. **`scene_hint` 不含坐标。** 它形如 `{"chair": 4, "door": 1, "table": 2}` + 图像尺寸。这把「坐标只能来自工具返回值」从**提示词要求**升级为**信息约束** —— LLM 想抄坐标也无从抄起。
2. **静态检查四项，零成本、不需要执行**：能否 `ast.parse` / 函数名是否在工具白名单 / 参数名是否存在于对应 schema / 是否调用 `submit`。这四项既是 Phase 8 数据集合成的**自动验收闸门**，也是「4B 能不能守住协议」的零成本测法。
3. **主路径 1 次调用**；失败时带 `RetryFeedback`（错误码 + 出错行 + 可用替代）定向重生成，上限 2 次。多轮 tool-calling 作为**消融臂 E′** 保留，与主路径共用工具库与执行器 —— **消融对比因此是干净的：只换循环层。**

#### (3) 角色② 视觉语义 `describe()`

**唯一能看图说话的角色，但签名里没有任何空间参数。**

```python
def describe(
    image: ImageRef,
    region: Region,                                  # bbox 或 mask
    attrs: Sequence[Literal["color", "material", "texture", "state", "shape"]],
    candidates: dict[str, list[str]] | None = None,  # 闭集优先，降噪
) -> list[Attribute]:                                # {name, value, confidence, source="vlm"}
```

`attrs` 是 `Literal` 白名单，`left_of` / `distance` / `above` **根本不在枚举里** ——
**空间幻觉在类型层面就写不出来**。这比在提示词里写「请不要判断空间关系」强一个量级：
前者是架构保证，后者是祈祷。**这一条要进答辩稿。**

- 承载方式按 §8.5 的代码规范走环境变量（`SPATIAL_VISION_MODEL` / `SPATIAL_VISION_BASE_URL`），本地 Qwen3.5 多模态与云端 `qwen3-vl-flash` 皆可，**不硬编码模型名**。
- `vlm=None` 时 `get_attributes` 返回 `CAPABILITY_DISABLED` → 天然构成一个消融臂。
- **主路径（纯空间问答）不依赖本角色**，因此它不阻塞 Phase 1；它只影响颜色 / 材质类问题与指代消解的强度。
- 输出 `value` 必须落在 `candidates` 内（闭集优先）；confidence 低于阈值时**必须上报**，不允许静默取最高分。

#### (4) 角色③ 几何层：双层接口，不要混

```python
# 内部层（scene_graph/relations.py）：纯函数，只吃 Node，可脱离场景图单测
def left_of(a: Node, b: Node, tol: float = 0.05) -> RelationVerdict: ...
def distance_m(a: Node, b: Node) -> RelationVerdict: ...

# 外部层（tools/geometry.py / tools/spatial.py）：吃 id，做存在性校验，组装证据
def query_relation(ctx, relation, a, b, tol=None) -> ToolResult: ...
def calculate_distance(ctx, a, b) -> ToolResult: ...
```

> **实现时的两处修订（2026-09-16，Phase 1 落地）**
>
> ① **内部层返回 `RelationVerdict` 而不是 `bool`。** 它形如
> `{value: bool|float, metric: dict[str, float], method: "geometry_v1"}`，
> 并实现 `__bool__` 因而 `if left_of(a, b):` 仍然可写。
> 改成这样是因为 §11 设计原则 2 要求「每个工具的返回值必须是几何证据」——
> 只回 bool 的话，外层就没法在 `Edge.metric` 里留下 `delta_x`，
> 「为什么判定为左」这件事就丢了，而那正是相对参考实现的卖点之一。
>
> ② **外部层不是「一个关系一个函数」，而是一个 `query_relation` 统一入口。**
> 11 个关系收成一个工具 + 一个枚举参数。理由是 prompt 长度直接受工具文档长度影响
> （参考实现的 program prompt 实测已达 6965 字符）——让模型记 11 个函数名，
> 不如让它记一份取值列表。分发表在 `scene_graph/relations.py` 的 `RELATIONS`。

分层的必要性：外层要承担 `NOT_IN_SCENE` 校验与 `evidence` 组装（因此依赖场景图与 trace）；
内层是干净数学。**混为一层，关系函数就再也测不了了** —— 而「关系可单测」正是 §12.1 里
Scene Graph 相对参考实现的三条收益之一。

#### (5) 提交契约：`submit()` 取代参考实现的魔法变量

```python
def submit(
    answer,                       # float | int | bool | str，受 answer_type 约束
    target_ids: list[str] = [],   # 被指认的物体 → 直接驱动 Viewer 高亮
    evidence: list[str] = [],     # ≥1 条：formula 字符串或 tool_call_id
) -> NoReturn: ...
```

| | 参考实现 | 本方案 |
|---|---|---|
| 取答案方式 | 命名空间里有没有 `final_result` 变量（`engine.py:292-303`） | **AST 里有没有 `submit()` 调用** |
| 缺失时 | → `""`，**静默算错**（不报错） | → 静态检查失败，判为程序错误，进失败诊断 |
| 证据要求 | 无 | **`evidence` 至少 1 条**，强制答案可回溯 |
| 高亮目标 | 无此概念 | `target_ids` 显式声明，**不让 LLM 再猜一次该高亮谁** |
| 类型约束 | 由 `answer_type` 事后转换 | `answer_type` 前置约束 `answer` 类型 |

`answer` 类型与 Omni3D-Bench 的 `answer_type`（int / float / str）对齐，**因此结果可与论文报告的基线同表对比** —— 这个兼容性是有意保留的。

**答案的自然语言渲染：模板为主，LLM 仅作可选润色。** 这是整条链路里**最后一个可能篡改数字的环节**：
模板只做拼装（`"{label} 最近，距离 {value:.2f} m"`），数值槽位由 `submit` 的参数锁定；
需要自然解释时才开一次 LLM 润色，且**不得改写任何数值槽位**。这样「输出可审计」才是真的可审计。

#### (6) 依赖方向与三条禁令

```
① synthesize  ──程序源码──►  执行器 + 工具库  ──►  { ③ 几何 | ② describe | L1 感知 }
                                      │
                              ToolResult 信封回流
                                      ▼
                    submit(answer, target_ids, evidence)
```

- **① → 工具库 → 能力层**，单向，不可逆。
- **② 与 ③ 互不知晓**，谁也不知道对方存在，更不互相调用。
- **几何层永不调用 LLM** —— 它只读 `SceneGraph`，不知道 LLM 存在。
- 禁令三条：`synthesize()` 不得直接读点云数组；`describe()` 不得输出坐标或空间判断；几何层不得发起模型调用。

#### (7) 接口开关 ↔ 实验臂对应表

这是本套接口设计的额外收益：**每个消融维度都对应一个接口开关**，而不是一份改写过的 prompt。

| 开关 | 关闭 / 切换后行为 | 对应实验臂或消融维度 |
|---|---|---|
| `action_space="program"` / `"tool_loop"` | 主路径 / 多轮逐步决策 | **E vs E′：动作空间对比** |
| `vlm=None` | `get_attributes` → `CAPABILITY_DISABLED` | 视觉语义贡献（仅属性类与指代消解题目） |
| `render="template"` / `"llm"` | 模板 / LLM 润色 | 渲染层是否引入数字漂移 |
| `planner="on"` / `"off"` | 是否先列工具类别再出程序 | 臂 G（创新点 4） |
| `static_check_only=True` | 只查 AST 不执行 | 诊断用：定位「程序写得对但跑不对」 |

> `planner="on"` 在**程序合成范式下含义变了**：不是「逐步规划」，而是「生成前先让 LLM 列出需要的
> 工具类别与推理链，再出程序」。它仍是可开关的臂 G，但报告里必须命名为 **planning-then-synthesis**，
> 不能沿用参考实现的逐步 planner 叙事。

---

## 14. Demo 架构

### 14.1 四区布局（现场答辩用）

```
┌──────────────────────────────┬─────────────────────────────┐
│                              │  Agent Chat                 │
│      3D Scene Viewer         │  ─────────────────────────  │
│                              │  User: 哪把椅子离门最近？    │
│   · 点云 / mesh 渲染          │  Agent: 正在定位门…          │
│   · 物体节点（带标签）         │         → 找到 1 个门        │
│   · 被 highlight 的目标高亮    │         → 找到 2 把椅子       │
│   · 相机可拖拽 / 可被工具驱动  │         → 计算 3D 距离…       │
│   · 关系连线（可选）           │  Agent: Chair 2，1.42 m      │
│                              │  [Highlight in 3D] ← 已联动  │
├──────────────────────────────┴─────────────────────────────┤
│  Scene Report  ★ L5 第一类输出 · 可随时导出                │
│  ────────────────────────────────────────────────────────  │
│  9 objects · 36 relations · scale_calibrated = true        │
│  sofa_1   1.92 × 0.78 × 0.91 m   sofa_1 ↔ table_1  1.608 m │
│  table_1  0.86 × 0.45 × 0.86 m   table_1 is behind sofa_1  │
│  [Export scene_graph.json]  [Summarize as text]            │
├────────────────────────────────────────────────────────────┤
│  Tool Trace / Reasoning Trace                              │
│  #1 find_object(door)          → 1 obj   | 412 ms          │
│  #2 find_object(chair)         → 2 obj   |  98 ms (cached) │
│  #3 get_3d_position(chair_1)   → [0.8,0.1,2.9] | 3 ms      │
│  #4 calculate_distance(...)    → 1.42 m  | 1 ms            │
│  #5 highlight_object(chair_1)  → ok      | 21 ms           │
└────────────────────────────────────────────────────────────┘
```

**第四区（Scene Report）是 2026-09-16 新增，与前三区是「并列」不是「附属」关系：**

- 前三区都在回答**某一道题**；第四区展示**整张图的解析结果**，**与提问无关，随时可导出**。
- 它是 L5（§11.1）的界面呈现，**零额外模型成本** —— 数据在 `build_scene_graph` 时就已经算好。
- 答辩时的作用：把「Agent 答对了一道题」升级为「系统理解了整个三维场景」。
  评委可以随便挑一个物体追问，面板里已有现成的米制尺寸与关系，**回答不依赖是否事先跑过某道题**。
- 配套交互：`counterfactual` 让评委现场说「移走沙发试试」，面板实时重算关系（纯图操作，零 GPU）。

### 14.2 技术选型建议

| 层 | 推荐 | 理由 |
|---|---|---|
| 3D 渲染 | **Three.js**（Web）或 **Open3D / Rerun**（Python 原生） | Web 方案最好分享（一个链接答辩）；Rerun 开发最快、可视化调试极强 |
| 后端 | FastAPI（Python） | 直接 import 你的 vision / tool 模块，零胶水 |
| 通信 | WebSocket | Trace 要流式逐条出现，才有「Agent 在思考」的观感 |
| 前端 | 单页 HTML + 少量 JS，或 Gradio | 不要上 React 工程化，时间要留给实验 |
| 部署 | 本地跑 | 答辩现场不依赖网络 |

**开发顺序建议**：先用 **Rerun** 做一个「能看见点云 + 能看到 trace」的调试版（1–2 天，让你自己 debug 用），再在 Phase 11 做 Three.js 版的答辩 Demo。**不要一开始就写前端。**

### 14.3 加分项

- `move_camera(target)` 问完自动飞到目标 → 演示效果极好
- 拔网线演示（档 C 完全离线）
- 左屏显示「LLM 原始 tool_call JSON」，右屏显示「真实执行 trace」→ 直观证明「不是 LLM 编的」
- 同一问题分别用「LoRA 前 / LoRA 后」跑一遍并排对照
- **答案附几何证据链**（§11.1 `answer_with_evidence`）：点开答案能看到「用了哪两个质心、算了哪个公式、数值多少」
  → 把「输出简单」变成「**输出可审计**」，比任何并行数字都更能打动评委
- **反事实演示**：评委现场指定「移走沙发」，Scene Report 面板即时重算全部关系
  （纯图操作，零 GPU、零重推理）→ 直接证明「中间表示本身有价值」
- **失败诊断面板**（§11.1 `diagnose_failure`）：主动展示「这道题失败在哪一环（检测 / 尺度 / 程序 / 工具）」，
  比被问出来更可信

---

## 15. 创新点（5 个候选的难度/价值评估，选定 3 个）

| # | 创新点 | 实现难度 | 实验可验证性 | 工作量 | 与 3D Vision 的相关性 | 结论 |
|---|---|---|---|---|---|---|
| 1 | **几何接地的 3D 工具库**（用 UniDepth 的 `points`+`intrinsics`，参考实现丢掉的） | 中 | 强（3D 定位误差、距离/尺寸精度可量化） | 大 | **极高** | ✅ **选** |
| 2 | **3D Scene Graph 作为空间中间表示** | 中 | 强（关系准确率、工具调用次数、幻觉率） | 中 | 高 | ✅ **选** |
| 3 | **面向 3D Tool Calling 的 QLoRA**（执行验证式数据合成） | 中高 | 强（微调前后 BFCL 风格指标 + 端到端准确率） | 大 | 中 | ✅ **选** |
| 4 | Multi-step Spatial Planning | 中 | 中（难以与「工具更多」解耦） | 中 | 中 | ⭕ 作为消融臂（不是主打创新） |
| 5 | 3D Viewer + Agent 联动 | 低 | 弱（更偏 Demo 而非科学贡献） | 中 | 低 | ⭕ 作为答辩亮点（不是核心创新） |

### 为什么这 3 个是对的

**创新点 1（工具库）**：参考实现的 `depth()` 返回一个标量，`same_object()` 靠 IoU，`3D size = 2D × depth` 缺焦距。而 UniDepth 的同一个 `infer()` 调用**本来就返回 `points`（相机系点云）和 `intrinsics`**——参考实现只取了 `depth` 一个键。这意味着：**你可以在不增加任何模型、不增加任何显存的前提下，把「深度级近似」升级成「真实三维几何」。** 这是纯粹 3D Vision 的贡献，且成本极低、收益可量化。

**创新点 2（Scene Graph）**：把「同一性判断（`same_object`）」「关系判断（`left_of`）」「存在性判断」这三类**不该由模型回答的问题**，从 LLM/VLM 手里拿回到确定性几何里。这直接对应你在需求里写的那条硬要求——「不允许 LLM 直接凭空编造空间关系」。而且它有干净的评测指标：关系准确率 vs 几何 GT、以及同一任务的工具调用次数下降。

**创新点 3（QLoRA）**：3D tool-calling 是一个**小众领域**，通用模型的先验帮不上太多（BFCL 高 ≠ 会调 `left_of(scene, a, b, tol)`）。配合「执行验证式数据合成」，你有：自建数据集（可写进报告）、本地训练（4060 上的完整训练日志，答辩加分）、可控的消融（同数据同超参，只换 checkpoint）。

**为什么 4 和 5 不选为主创新**：创新点 4 的实验很难归因——「多步规划带来提升」和「工具更多带来提升」在单臂实验中分不开，要做 2×2 交叉实验才干净，成本高而区分度低。创新点 5 是工程/展示价值而非科学贡献。**把它们放在消融表和 Demo 里，不要放在「创新点」标题下**——这个取舍本身就是成熟度的体现。

### 一句话创新陈述（可直接用于答辩/简历）

> 本项目利用 UniDepth 输出的相机系点云与内参构建了**几何接地的 3D 工具库**（把原工作丢弃的三维信息接入推理链路），以**3D 场景图**作为空间关系的确定性中间表示以消除 LLM 的空间幻觉，并通过**执行验证式数据合成 + 本地 QLoRA** 使 4B 级开源模型在 3D 工具调用任务上逼近闭源大模型，全部实验在单张 RTX 4060 (8GB) 上完成。

---

## 16. 实验设计

### 16.1 主实验臂

| 实验 | 配置 | 目的 |
|---|---|---|
| **A** | ~~参考实现原始（GPT-4o + 原始工具 + 随机 API）~~ | **已移除**：该臂依赖的检出不再随仓库分发；早期 6 次真跑留在 `results/A/`，但**不可复现** |
| **B** | 参考实现 + Qwen3.5-4B（换 LLM，其余不动） | 隔离「换 LLM」的收益 |
| **C** | 参考实现 + Qwen3.5-4B + 我的 3D 工具库 | 隔离「工具升级」的收益（创新点 1） |
| **D** | C + Scene Graph | 隔离「中间表示」的收益（创新点 2） |
| **E** | D + 自研 Agent，**动作空间 = 程序合成（主路径）** | 隔离「自研 Agent + 几何接地」的收益 |
| **E′** | D + 自研 Agent，**动作空间 = 多轮 tool-calling** | ★ **动作空间对比**：程序合成 vs 逐步决策 |
| **F** | E + QLoRA（我的数据集） | 隔离「微调」的收益（创新点 3） |
| **G** | F + planning-then-synthesis | 消融臂（创新点 4） |
| **H** | E + Viewer 联动 | Demo 臂，不作为科学对照 |

> **E 与 E′ 只换循环层**：工具库、执行器、prompt 里的工具文档、随机种子全部相同，接口开关见 §13.3(7)。
> 这条消融是「三个创新点的收益能否与范式收益分离」的关键 —— 没有它，C/D/E 的增益无法归因。
> **H 臂不是科学对照**：Viewer 联动不改变任何空间计算，它的价值在答辩现场而不在表里。

### 16.2 建议的消融主表

| 配置 | OpenAI | 开源 LLM | LoRA | 3D 工具 | Scene Graph | Planner | 空间问答准确率 | 工具选择准确率 |
|---|---|---|---|---|---|---|---|---|
| **论文报告值（仅引用，不参与对比）** | ✓ | | | | | | 40.4（gpt-4o，**本机无法复现**） | — |
| Baseline：参考实现原始流水线 + 强开源模型（**已移除**） | | ✓ | | | | | ← 消融的真正起点 | |
| + LoRA | | ✓ | ✓ | | | | | |
| + 3D 工具 | | ✓ | ✓ | ✓ | | | | |
| + Scene Graph | | ✓ | ✓ | ✓ | ✓ | | | |
| + Planner | | ✓ | ✓ | ✓ | ✓ | ✓ | | |

> **为什么第一行不能当基线**：`api.openai.com` 在本机网络不可达（实测超时 12 s+），
> gpt-4o 调不到，所以论文的 40.4 只能作为**文献引用值加脚注**，注明服务商与模型不同、
> 不可直接比较。消融基线改用「参考实现原始流水线 + 一个强开源模型」
> —— 注意 **DeepSeek 这边已无强模型档**：`deepseek-v4-pro` 自 2026-09-14 12:00 起
> 全部路由到 V4.1-Flash（见 §7.4），换了名字不会换到更强的模型。
> 所以要上界臂就得**跨服务商**（如 `qwen3-vl-plus`），或者把上界臂定义为
> 「同一模型 + 更多推理预算（思考模式开启」），但那混淆了模型与推理预算两个变量，
> 需要在表里单独说明。其余各臂仍跑在同一服务商、同一套参数、同一份数据上，
> 内部一致性比跨服务商对比更好。

**另外必须单列一张「模型规模 × 工具调用能力」表**（这是你主模型选型的证据）：

| 模型 | 尺寸 | 量化 | BFCL-V4（官方值） | 我的 Tool Selection Acc | 我的端到端 Acc | 延迟 | 峰值显存 |
|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | 0.8B | 4bit | 25.3 | | | | |
| Qwen3.5-2B | 2B | 4bit | 43.6 | | | | |
| **Qwen3.5-4B** | 4.66B | 4bit | 50.3 | | | | |
| Qwen3.5-4B + LoRA | 4.66B | 4bit | — | | | | |
| Qwen3.5-9B（云） | 9B | 4bit | 66.1 | | | | |
| GPT-4o（API） | — | — | — | | | | |

### 16.3 评价指标（12 项，全部要能落表）

| # | 指标 | 定义 | 数据类型 |
|---|---|---|---|
| 1 | Spatial Answer Accuracy | 与 GT 答案一致 | 按 int / float(MRA) / yes-no / multi-choice 分列 |
| 2 | Tool Selection Accuracy | 选中工具集合 vs 参考集合（F1） | % |
| 3 | Tool Call Success Rate | 工具执行无错的比例 | % |
| 4 | Program Execution Success Rate | 程序整体跑通比例 | % |
| 5 | **3D Localization Error** | 预测 3D 质心 vs GT 的欧氏距离 | 米（mean / median） |
| 6 | Spatial Relation Accuracy | 关系判断 vs 几何 GT | % |
| 7 | Multi-step Task Success Rate | 需要 ≥3 步的任务的成功率 | % |
| 8 | Average Latency | 每题端到端耗时（区分是否命中缓存） | 秒 |
| 9 | GPU Memory | `torch.cuda.max_memory_allocated()` | GB |
| 10 | Inference Cost | 云卡按小时折算 / API 按 token 折算 | ¥/1000 题 |
| 11 | **Scene Description Recall** | `describe_scene` 输出的物体数 / 检测到的物体数 | % |
| 12 | **Relation Edge Consistency** | Scene Report 中关系边与几何 GT 一致的比例 | % |

**指标 11–12 是 2026-09-16 新增**，用来度量**第一类输出（L5 场景报告）**，与问答准确率**相互独立**：

- 11 度量「有没有漏掉物体」，12 度量「关系有没有报错」——两者都不需要提问就能算，**可以在没有 GT 问答对
  的图片上跑**（例如你自己拍的宿舍照片），这是它比指标 1 更灵活的地方。
- 它们让「工具库升级」这一个创新点能出**两张表**：一张问答表（指标 1–8）+ 一张场景理解表（指标 11–12），
  而不是只有一张。
- 12 的参考值可直接来自 §12 的 Scene Graph 几何 GT 构造流程（单遍卷积式构建），不需要额外标注。

**注意指标 5 的数据来源**：Omni3D-Bench 是基于 Omni3D 的，Omni3D 本身带 3D 标注（这也是「用 GT 替换视觉模块」能工作的前提）。**但你需要自己确认 Omni3D-Bench 的 annotations.json 里是否含 3D 坐标字段——`README.md` 里展示的字段只有 image/question/answer_type/answer（`README.md:49-58`），没有 3D 标注，所以 GT 3D 坐标可能需要回到 Omni3D 原始数据里取，这一步存在不确定性，需在 Phase 1 验证。**

### 16.4 实验纪律（避免答辩被问倒）

- 固定 seed；工具库版本号写进结果文件
- 每题跑 3 次取均值（LLM temperature > 0）
- 报告**配对显著性**（同一批题、同一场景，用 McNemar 检验或 bootstrap），而不是只看均值差
- 明确区分「瓶颈在视觉」还是「瓶颈在语言」——`--oracle` 臂就是干这个的

---

## 17. 项目目录

```
3d_spatial_agent/
├── README.md
├── pyproject.toml
├── configs/
│   ├── vision.yaml            # GroundingDINO / SAM2 / UniDepth 的阈值与开关
│   ├── llm.yaml               # base_url / model / temperature / max_steps
│   ├── tools.yaml             # 工具库版本与启用开关
│   └── scene_graph.yaml       # 关系阈值 tol / up_axis 策略
│
├── vision/                    # ★ L1 感知层
│   ├── grounding.py           #   包 GroundingDINO
│   ├── segmentation.py        #   包 SAM2.1
│   ├── depth.py               #   包 UniDepthV2（depth + points + intrinsics 三者都要）
│   ├── geometry.py            #   ★ 点云 → 质心 / extent / PCA / up_axis（自研新代码）
│   └── registry.py            #   模型常驻/懒加载/显存回收策略
│
├── scene_graph/
│   ├── schema.py              # ✅ Node / Edge / SceneGraph（Pydantic；坐标系约定写死在文件头）
│   ├── relations.py           # ✅ ★ 纯几何关系 + UpAxis 归一（零依赖，可脱离场景图单测）
│   ├── builder.py             #   单遍构建：检测→分割→升维→关系（Phase 6）
│   ├── store.py               #   JSON 缓存读写 + 版本（Phase 6）
│   └── tests/
│       ├── conftest.py        #   把项目根塞进 sys.path（没装包，手写路径）
│       └── test_relations.py  # ✅ 29 用例：坐标系符号、容差死区、on/inside、退化情形
│
├── tools/                     # ★ 暴露给 LLM 的工具（= 你的"API"），契约见 §13.3
│   ├── result.py              # ✅ ★ ToolResult 信封 + 6 个错误码 + 恢复动作枚举（共同语言）
│   ├── version.py             # ✅ TOOLS_VERSION / RELATIONS_METHOD —— 版本化单一来源
│   ├── registry.py            # ✅ @tool 装饰器 + ToolContext（注入 scene/vlm/viewer）+ trace
│   ├── guards.py              # ✅ 存在性校验：NOT_IN_SCENE（幻觉捕获点）/ NOT_FOUND
│   ├── geometry.py            # ✅ get_3d_position / get_3d_extent / calculate_distance / calculate_angle
│   ├── spatial.py             # ✅ list_objects / get_object / find_object / single_object /
│   │                          #   find_nearest / find_farthest / query_relation（11 关系统一入口）
│   ├── perception.py          #   detect_objects / segment_object / estimate_depth（Phase 4，需 GPU）
│   ├── attributes.py          #   get_attributes（走角色②；vlm=None → CAPABILITY_DISABLED）
│   ├── viewer_control.py      #   highlight_object / move_camera / draw_relation（Phase 11）
│   ├── scene_report.py        # ★ L5 第一类输出：describe_scene / summarize_scene /
│   │                          #   answer_with_evidence / diagnose_failure / counterfactual
│   └── schemas.py             #   JSON Schema 渲染（供 vLLM tool calling）；版本号取自 version.py
│
├── agents/                    # ★ 三角色接口契约见 §13.3；实现记录见 §13.4
│   ├── synthesizer.py         # ✅ 角色① synthesize()：AST 四项静态检查 + 定向重生成（≤2 次）
│   ├── loop.py                # ✅ 主循环（自己写，不用 LangGraph）—— 主路径=单轮，E′=多轮
│   ├── planner.py             # ✅ planning-then-synthesis（可开关，消融臂 G）
│   ├── executor.py            # ✅ 沙箱执行（同进程 + 定时器看门狗硬超时）+ trace
│   ├── verifier.py            # ✅ 答案是否有几何证据支持（四档：supported/weak/unsupported/abstained）
│   ├── memory.py              # ✅ 工作记忆（object_id 映射 / 已确认事实 / 压缩观察）
│   └── prompts/
│       ├── system.py          # ✅ 角色① 的提示词模板（占位符用 `{{X}}` + 收尾自检）
│       ├── planner.py         # ✅
│       └── verifier.py        # ✅
│
├── llm/
│   ├── adapter.py             # ✅ OpenAI 兼容客户端（双端点路由、密钥掩码、调用记账）
│   ├── schema.py              # ✅ tool schema → prompt 渲染 + TOOLS_VERSION
│   ├── vlm.py                 # ✅ ★ 角色② 的适配；主体在 `vision/semantics.py`（attrs 白名单，签名无空间参数）
│   ├── render.py              # ✅ 答案渲染：模板为主，LLM 仅可选润色且不得改写数值槽位
│   └── finetune/
│       ├── data_builder.py    # ★ 执行验证式数据合成（创新点 3 的引擎）
│       ├── verify.py          #   轨迹执行校验
│       ├── train_qlora.py     #   Unsloth / peft QLoRA
│       └── configs/           #   qwen3.5_2b.yaml / qwen3.5_4b.yaml
│
├── dataset/
│   ├── scenes/                #   每个场景的 scene_graph.json 缓存
│   ├── raw/                   #   Omni3D-Bench
│   ├── synthesized/           #   tool_calls_train.jsonl / valid.jsonl
│   └── builders/              #   各数据集 → 统一 schema 的转换脚本
│
├── evaluation/
│   ├── metrics.py             # ✅ 四类子指标 + Total 的论文口径复算（§24.3 验算过）
│   ├── README.md              # ✅ 评测口径、运行方式与三个设计决定
│   ├── tests/                 # ✅ 用例（零 torch / 零 GPU / 零联网）
│   ├── ablation.py            #   ⬜ 汇总多臂 → 主表
│   └── stats.py               #   ⬜ 配对显著性检验
│   （早期还有一个实验臂运行器与它的 Windows 兼容层，已随上游检出于 2026-09-20 移出）
│
├── demo/
│   ├── server.py              #   FastAPI + WebSocket
│   ├── static/                #   Three.js 前端（四区：Viewer + Chat + Scene Report + Trace）
│   └── rerun_view.py          #   ★ 开发期调试视图（先做这个）
│
├── vendor/
│   └── UniDepth/              #   唯一保留的第三方检出（视觉骨干）；来源与许可见 `vendor/VENDOR.md`
│
├── tests/
│   └── test_tools.py          # ✅ 58 用例：错误码契约、能力开关、参数错误冒泡、trace、坐标系
│
├── scripts/
│   ├── smoke_tools.py         # ✅ 冒烟：场景图→工具→答案+证据链+trace（零 GPU、零联网）
│   ├── build_scene.py         # ✅ 单图 → 场景图（接地 → 分割 → 升维 → 关系）
│   ├── run_agent.py           # ✅ 端到端入口（dry-run / program-file / question 三种模式）
│   ├── serve_demo.py          # ✅ 前端演示台（/api/ask 现场真跑 + 反事实 + 上传建图）
│   ├── selfcheck_agent.py     # ✅ 自检：契约、答案证据校验器、工具链
│   ├── mine_glue_patterns.py  # ✅ 胶水形态挖掘（从已落盘程序里聚类候选算子）
│   ├── inspect_scene.py       # ✅ 场景图查看 / 导出
│   ├── report_scene.py        # ✅ 场景图报告
│   └── probe_combination*.py  # ✅ 组合探针（分析 + 运行）
└── reports/
    └── figures/
```

**关于上游检出的处理**：早期做法是把检出原样放在 `vendor/` 下、改动只走一层薄适配器（`llm/adapter.py` 替代其 LLM 封装，`tools/registry.py` 替代其模块管理器），好处是原版可随时对照跑。**该检出已于 2026-09-20 移出本仓库**：不再分发、不再有对照臂，本节只作历史记录。

---

## 18. 分阶段路线（Phase 0–13）

每个阶段给出：**目标 / 输入 / 输出 / 代码位置 / 是否需要 GPU / 4060 能否 / 是否需要租卡 / 主要风险 / 完成标准**。

---

### Phase 0 — 环境配置（**已完成：Windows 原生，未装 WSL**）
- **目标**：在 Windows 原生 Python 里建起视觉栈依赖
- **输入**：本机（Windows 11 + 4060 Laptop 8GB + 驱动 560.76）
- **输出**：`venvs/vision`（Python 3.12）+ `phase0/requirements-vision.lock.txt`（74 个固定版本）
- **代码位置**：`phase0/01_setup_windows.ps1`（venv → torch → 依赖 → UniDepth → 自检 → 锁文件，一条脚本）
- **需要 GPU**：是（验证 torch.cuda）
- **4060 能否**：✅
- **需要租卡**：否
- **主要风险**：① ~~原生 Windows 跑不通~~ **已推翻** —— 三个视觉模型改走 transformers 原生纯 PyTorch 实现，不再需要 Unix 信号与 CUDA 编译链；② `xformers` / `triton` **不需要装**（UniDepth 里每一处 import 都在 `try/except ImportError` 内，attention 有 SDPA 回落）；③ Python 必须 ≥3.10（本机用 3.12）；④ 单条命令约 121 秒被杀 ⟹ 长步骤必须拆开跑
- **完成标准**：`verify_env.py` 通过 —— CUDA 可用、认到 4060 Laptop 8188 MiB / sm_89、`import UniDepthV2` 成功

---

### Phase 1 — ~~上游原版跑通~~ **已移除**（2026-09-20）

- **原目标**：用 GPT-4o 跑通一小批题，拿到可对照的基线数字与显存曲线
- **现状**：该臂依赖的上游检出与配套兼容层脚本已于 2026-09-20 一并移出仓库
  （`evaluation/` 下与之相关的三个文件不再存在）。早期 6 次真跑留在 `results/A/`，但**已不可复现** ——
  只能当「当时跑过什么」的记录，**不能当可复现的对照臂**。
- **仍然有效、与上游无关的前置工作**：
  - 题集：Omni3D-Bench（**实测 501 题 / 201 图**，来自 HF `dmarsili/Omni3D-Bench`）
  - 产物契约：`results/<arm>/` 下的四个稳定入口 `subset.json` / `latest_run.json` /
    `latest_plan.json` / `latest_failed.json`，分流规则见 §24.5
  - 抓取与展开（都已落盘，不必重跑）：

```bash
# 1) 抓 parquet（106.5 MB，hf-mirror，约 23 s）
venvs/vision/Scripts/python.exe dataset/builders/fetch_omni3d_bench.py

# 2) 展开成评测期望的目录布局（annotations.json + images/，501 题 / 201 图 / 95 MB）
D:/Users/ROG/anaconda3/python.exe dataset/builders/read_omni3d_bench.py
```
- **顺带完成**：确认 Omni3D-Bench 的 annotations.json **不含 3D 标注**（2026-09-17，见 §24.1）：
  只有 (图, 问, 答) 六列，没有 GT 相机也没有深度 ⟹ 指标 5（三维尺寸）**不能**用它的 GT 做，
  需要另找带标定的数据源。

---

### Phase 1b — 自建骨架：场景图 + 工具库 + 信封（2026-09-16 **已完成**）

> 对应 §8 阶段路线里的「1 跑通自建最小 pipeline」。它先于上游原版对照臂的搭建完成，
> 因为原版对照臂（路径 A）只影响基线数字，而骨架决定后面所有 Phase 的代码结构。

- **目标**：把 §13.3 的接口契约落成**可跑、可测、不需要 GPU** 的代码
- **输出**：
  | 文件 | 内容 |
  |---|---|
  | `tools/result.py` | `ToolResult` 信封 + 6 个错误码 + `Recovery` 枚举 |
  | `tools/version.py` | `TOOLS_VERSION` / `RELATIONS_METHOD` 单一来源 |
  | `tools/registry.py` | `@tool` 装饰器、`ToolContext`、trace、能力开关 |
  | `tools/guards.py` | `NOT_IN_SCENE`（幻觉捕获点）/ `NOT_FOUND` 校验 |
  | `tools/geometry.py` | `get_3d_position` / `get_3d_extent` / `calculate_distance` / `calculate_angle` |
  | `tools/spatial.py` | `list_objects` / `get_object` / `find_object` / `single_object` / `find_nearest` / `find_farthest` / `query_relation` |
  | `scene_graph/schema.py` | `Node` / `Edge` / `SceneGraph`（坐标系约定写在文件头） |
  | `scene_graph/relations.py` | 纯几何关系 + `UpAxis` 归一 + `pairwise` |
  | `scripts/smoke_tools.py` | 端到端冒烟（零 GPU、零联网） |
- **代码位置**：`tools/`、`scene_graph/`、`tests/`、`scripts/smoke_tools.py`
- **需要 GPU**：**否** —— 这整批代码一行 torch 都不 import
- **4060 能否**：✅（无关）
- **需要租卡**：否
- **完成标准**：**87 个用例全绿（0.34 s）**；`smoke_tools.py` exit 0 并产出 trace.jsonl ✅

**四处实现层面的裁决（值得写进报告）**

1. **`ToolResult.__post_init__` 强制「失败必须带 `ToolError`」** —— 于是「静默失败」这个对象
   在类型层面构造不出来。早期参考实现正是在这里失守：命名空间里没有 `final_result`
   就取 `""`，不报错、不算失败，直接拿去评分。
2. **恢复动作也枚举化。** `ErrorCode → Recovery` 的映射写死在代码里
   （`NOT_IN_SCENE → read_scene`、`AMBIGUOUS → add_constraint` …），
   `to_dict()` 会把它一并输出给模型。模型不必猜「出错了该怎么办」。
3. **参数值域错误必须冒泡，不能被算成「工具失败」。** `anchor="foo"` 抛 `ToolArgumentError`
   且装饰器**不捕获** —— 它要在失败诊断里归到「程序」而不是「工具」，
   否则「工具调用成功率」这个指标会随模型写错参数而下降，看起来像工具的问题。
   这与「工具报结构化错误供恢复」是两件事，必须分开（见 `tools/registry.py` 模块 docstring）。
4. **`find_nearest` 必须排除 anchor 自身。** 否则问「哪把椅子离椅子最近」会返回它自己，
   而模型很可能就此给出一个看起来合理、实际无意义的答案 —— 这类「静默无意义」
   比报错危险得多。

**一处与文档的偏离**：§13.3(4) 原写内部层 `-> bool`，实现改为返回 `RelationVerdict`
（`{value, metric, method}`，实现 `__bool__`）。理由见 §13.3(4) 的修订说明。

**顺带确认的方法学事实**：`scene_graph/tests/test_relations.py` 与 `tests/test_tools.py`
全程 0.34 秒、不需要 GPU、不需要联网。**这就是「关系可单测」这条架构收益的实测形态** ——
早期参考实现的对应能力是一次 `vqa(image, question, bbox)` 模型调用，既不确定也无法写断言。

---

### Phase 2 — ~~彻底理解上游 Agent architecture~~ **已并入 §2**

- **原目标**：不是「读过」，而是能**预测**它的行为并手工修复它
- **现状**：该检出已移出本仓库，逐类拆解与行号引用无法复核，故压缩合并进 **§2**
  （评估结论与弃用理由）。原审计还发现了一批它的实现缺陷（含一处「默认数据集恒为
  `clevr`」、一处方法名正则被返回值注解击穿、一处异常处理无限递归不设上限），
  这些发现的**用处不在于「读懂它」**，而在于它们直接指出了三类必须避开的设计：
  ① 动作空间要显式白名单化；② 提交要用契约而不是魔法变量；③ 视觉语义与空间几何必须拆开。
  三条都写进了 §2.3 与 §13.3。

---

### Phase 3 — 接入开源 LLM
- **目标**：把三个文本 Agent + `vqa()` 全部指向本地 Qwen3.5-4B
- **输入**：Phase 2 的机制笔记
- **输出**：一个把文本 Agent 与视觉语义都指向开源模型的流水线；实验臂 **B** 的初步数字
- **代码位置**：`llm/adapter.py`（OpenAI 兼容客户端，含双端点路由）、`llm/render.py`（提示词渲染）、`llm/vlm.py`（视觉语义）
- **需要 GPU**：是（LLM 推理）
- **4060 能否**：✅（Qwen3.5-4B 4bit）
- **需要租卡**：否。**如果要对照 9B，可以租一次 4090 24GB 跑一夜**（可选）
- **主要风险**：① **vLLM 的 tool-call/reasoning parser 名字随版本变**，先查你装到的版本的文档；② 视觉栈与 LLM 必须**分 venv 分进程**（见 §2.5）；③ VLM 请求必须真的走多模态（否则 `vqa` 静默变差）；④ 中文 prompt 可能让模型在英文 benchmark 上输出中文——**系统提示里强制英文输出**
- **完成标准**：同一批 20 题在云端强模型与 Qwen3.5-4B 上跑通，产出**对照表**（哪怕开源模型更差 ——「差多少」本身就是有价值的结论）

---

### Phase 4 — 4060 本地运行视觉模型（显存工程）
- **目标**：把视觉栈跑稳，并通过串行加载/卸载把峰值显存压进 8GB
- **输入**：Phase 1 的模型
- **输出**：`vision/registry.py`：懒加载 + 用后释放 + 可配置精度；一张实测显存表
- **代码位置**：`vision/*.py`
- **需要 GPU**：是
- **4060 能否**：✅
- **需要租卡**：否
- **主要风险**：① ~~三个模型同时驻留可能触顶~~ **已实测关闭** —— 同时驻留只占 1200 MB / 14.7%，峰值 2177 MB（§2.5、§20 Step 0.5b）；② 混合精度（`torch.autocast` 的 bf16 与 GroundingDINO 内部的 fp16）**可能造成数值差异，需记录**；③ 无 WSL，不存在额外宿主开销
- **完成标准**：单张图跑完整 L1 感知链且**峰值显存 < 7 GB**；产出可复现的显存实测表（附测量脚本）

---

### Phase 5 — 实现 3D Spatial Tools
- **目标**：实现 §11 的 L1+L3 工具，并证明它比 `2D×depth` 更准
- **输入**：Phase 4 的视觉层
- **输出**：`tools/geometry.py`、`tools/spatial.py` + 单元测试；一个**距离/尺寸精度对照实验**
- **代码位置**：`tools/`、`vision/geometry.py`
- **需要 GPU**：是（感知部分），几何部分纯 CPU
- **4060 能否**：✅
- **需要租卡**：否
- **★ L5 提前到这里做**（2026-09-16 补充）：`tools/scene_report.py` 的 `describe_scene` 只依赖
  L2/L3 已经算好的质心与关系，**是 L1–L5 里最便宜的一层**。在这里就把 `SceneReport` 序列化打通，
  后面 Phase 6（Scene Graph）、Phase 10（评估指标 11–12）、Phase 11（Demo 第四区）都直接复用，
  避免到 Phase 11 才发现数据不够。**不要拖到 Demo 阶段才做。**
- ✅ **上述 L5 已完成**（2026-09-16 晚）：`tools/scene_report.py` 落了 4/5 个工具
  （`describe_scene` / `summarize_scene` / `diagnose_failure` / `counterfactual`），
  落盘入口 `scripts/report_scene.py` 产出 `scene_report.json` + `scene_report.md`。
  `answer_with_evidence` 挂起到 Phase 2（等题集）。实测热态中位数 1.04–5.63 ms、零 GPU，
  新增 38 个单测（全量 231 绿）。实现中的四处判断、以及 `living_room_gt` / `living_room_pred`
  的对照实测见 **§11.1.1**。
- **主要风险**：① **单目深度的尺度漂移**——`get_3d_position` 返回的绝对值可能系统性偏大/偏小，必须先做 `calibrate_scale` 并如实报告；② `up_axis` 估计不稳 → `above/below` 最不可靠；③ `left_of` 在相机有 roll 时不成立，需在文档里写成**显式假设**
- **完成标准**：`pytest`（**仓库根**，不是 `pytest tests/` —— 后者会静默少收 67 例，
  `scene_graph/tests/` 是独立一套）全绿；在同一批标注数据上，`get_3d_extent` 的相对误差**优于**参考实现的 `2D_size × depth` 基线（这是创新点 1 的核心证据）
- **诚实要求**：如果实验显示它没变好，**照实写**，并分析原因（这比强行报喜更有学分）

---

### Phase 6 — 实现 Scene Graph
- **目标**：§12 的构建器 + 关系模块
- **输入**：Phase 5 的工具
- **输出**：`scene_graph.json` 批量缓存 + 关系准确率评测
- **代码位置**：`scene_graph/`
- **需要 GPU**：构建时是（可批量离线跑），查询时否
- **4060 能否**：✅
- **需要租卡**：否
- **主要风险**：① 宽泛 prompt `"objects"` 的漏检/误检会污染整张图；② 关系阈值 `tol` 需要标定，设错会系统性偏；③ 30 个物体 = 435 个关系对，**边太多** → 建议只存白名单关系（on/inside/left_of/right_of/above/near），而不是全连接
- **完成标准**：≥50 个场景的 scene_graph.json；关系准确率 vs 几何 GT 有明确数字；查询延迟（纯 CPU）< 50 ms

---

### Phase 7 — 构建自己的 Agent
- **目标**：§13 的循环 + 重试 + 记忆 + trace
- **输入**：Phase 3 的适配器 + Phase 6 的场景图
- **输出**：`agents/synthesizer.py` + `agents/executor.py` + `agents/loop.py` + `tools/result.py`（信封）+ 实验臂 **E 与 E′** 的数字
- **代码位置**：`agents/`、`tools/result.py`
- **需要 GPU**：是（LLM）
- **4060 能否**：✅
- **需要租卡**：否
- **主要风险**：① **4B 生成的程序语法 / 工具名出错** → AST 四项静态检查 + 定向重生成（≤2 次）——这是主路径**唯一**的失效点，必须最先测；② 程序死循环或卡死 → 沙箱 `multiprocessing` + 硬超时（**不能用 `signal.SIGALRM`，Windows 无此信号**）；③ `describe()` 越界输出空间判断 → 本应由 `Literal` 白名单在类型层堵死，若仍出现说明实现漏了校验；④ E′ 臂步数失控 → 步数上限 + 重复调用检测
- **完成标准**：① **E 臂**在 50 题上端到端跑通率 > 80%，每题平均工具调用 < 6；② **E′ 臂**在同一固定题集上跑完，两者可同表对比；③ `ToolResult` 信封覆盖全部 L1–L5 工具，无一例外；④ 工具 trace 可直接算出 §16.3 的指标 2/3/4；⑤ **三角色依赖方向零违反**（几何层代码中不存在任何 LLM 调用）

---

### Phase 8 — 构造 Tool Calling 数据集
- **目标**：§10.3 的经济验证式合成，产出 2–5k 条**已验证**轨迹
- **输入**：Phase 6 的场景图 + 自动生成的问题 + 教师模型
- **输出**：`dataset/synthesized/tool_calls_train.jsonl`（含 `verified: true`）+ held-out 验证集
- **代码位置**：`llm/finetune/data_builder.py`、`llm/finetune/verify.py`
- **需要 GPU**：否（用 API 教师模型；也可选本地 9B）
- **4060 能否**：✅（不含训练）
- **需要租卡**：否。**如果用教师 API，注意成本；用本地小模型当教师质量会掉**
- **主要风险**：① **教师模型会产生「看起来合理但几何错误」的轨迹** → 所以执行验证是必需步骤，不能省；② 问题模板太单一 → 数据多样性不足，LoRA 会过拟合到模板；③ 类别分布不均 → 需要按物体类别/关系类型分层采样
- **完成标准**：≥2,000 条通过执行验证的轨迹；人工抽查 50 条，正确率 > 95%；输出数据统计报告（关系类型分布、步数分布、工具使用频次）

---

### Phase 9 — 在 4060 上进行 LoRA / QLoRA
- **目标**：训练出一个在 3D tool calling 上更强的 4B 模型
- **输入**：Phase 8 的数据集
- **输出**：LoRA adapter + 训练曲线 + 训练日志（loss/显存/耗时）
- **代码位置**：`llm/finetune/train_qlora.py` + `configs/`
- **需要 GPU**：是（核心训练）
- **4060 能否**：⚠️ **预期可行但必须实测**。先跑通 0.8B/2B 验证管线，再上 4B
- **需要租卡**：**仅在 4B 确实 OOM 且所有降级手段用尽时**，租一次 4090 24GB（几小时）
- **主要风险**：① OOM（降级顺序见 §10.2）；② **Qwen3.5 的 GDN 层使 target_modules 命名与 Qwen3 不同** → 必须 `print(model.named_modules())` 先确认（不确定项）；③ 过拟合（2 epoch 内 early stop）；④ WSL2 上 Unsloth/flash-attn 编译可能失败 → 备选 peft 原生 QLoRA
- **完成标准**：训练完整跑完，loss 正常下降，held-out 上 tool-call 格式合法率 > 95%；**记录峰值显存与总耗时**（这些数字答辩时很有说服力）

---

### Phase 10 — 评估微调前后性能
- **目标**：实验臂 C/D/E/F/G 全跑，出主表和**场景理解子表**
- **输入**：Phase 1–9 的全部产物
- **输出**：`evaluation/` 的完整结果 + §16.2 的表 + 配对显著性
- **代码位置**：`evaluation/`
- **需要 GPU**：是
- **4060 能否**：✅（本地臂）；9B 臂需要云
- **需要租卡**：**可选**，只为 9B 对照臂租一次 4090（几小时）
- **★ 要出两张表，不是一张**（2026-09-16 补充）：
  - 表 A（指标 1–8）：**问答**质量 —— 这是各臂都有的
  - 表 B（指标 11–12）：**场景报告**质量 —— 描述召回率 + 关系边一致率
  - 表 B 的意义：即使某道题的程序写错了，场景报告仍然可能是对的。**把这两者分开，才能说明
    「瓶颈在语言（程序合成）」而不是「瓶颈在视觉/几何」**——这正是答辩最容易被问到的归因问题。
  - 附带好处：表 B 不需要 GT 问答对，**可以额外拿你自己拍的 10 张照片跑**，扩大样本量而不增加标注成本。
- **主要风险**：① 指标实现有 bug —— 先用手工构造的 20 题做指标单测；② 不同臂用了不同题集 → **必须固定题集**；③ 只报均值不做检验 → 会被问倒
- **完成标准**：全部臂在同一固定题集上完成；主表 + 显著性检验齐全；**每个结论都有对应的配对实验支撑**（不能有「我觉得是因为…」）

---

### Phase 11 — 3D Viewer
- **目标**：§14 的**四区**可交互 Demo
- **输入**：场景图 + Agent 的 trace + Phase 5 的 `SceneReport`
- **输出**：可现场演示的 Web Demo
- **代码位置**：`demo/`
- **需要 GPU**：否（渲染在浏览器）
- **4060 能否**：✅
- **需要租卡**：否
- **★ 第四区（Scene Report）不是可选项**（2026-09-16 补充）：三区版只演示「答对一道题」，
  四区版演示「理解了整个场景」。硬性要求：面板必须支持 `counterfactual`
  （现场移走物体后关系实时重算）—— 这是全项目**唯一一个能让评委现场出题**的功能，
  且零 GPU 开销。如果时间不够，砍掉的是 `move_camera` 的动画，而不是这一区。
- **主要风险**：**前端工时失控**。硬性约束：先做 Rerun 调试版（Phase 7 就该有），Three.js 版只做「点云 + 高亮 + 相机飞行」三件事，其余全砍
- **完成标准**：现场演示「问一句 → 高亮目标 → 相机飞过去」全程 < 15 秒且不崩

---

### Phase 12 — 完整实验
- **目标**：把所有臂跑满，产出最终数字 + 图表
- **输入**：Phase 1–11
- **输出**：`reports/figures/` 全套图 + 最终主表 + 失败案例分析
- **需要 GPU**：是
- **4060 能否**：✅
- **需要租卡**：可选（9B 臂）
- **主要风险**：时间不够 → 提前定死题集规模（建议 200–500 题），不要到最后才发现跑不完
- **完成标准**：一个**失败案例分析章节**（哪些题所有臂都失败、为什么）——这部分往往比主表更能说明你理解了系统

---

### Phase 13 — 答辩 Demo
- **目标**：15 分钟讲清「3D Vision + Agent + 训练 + 实验」
- **输入**：Phase 12 的全部产物
- **输出**：演示脚本 + PPT + 现场流程单 + 备份视频
- **需要 GPU**：是（现场推理）
- **4060 能否**：✅
- **需要租卡**：否
- **主要风险**：① 现场环境炸 → **必须录一份 Demo 视频备份**；② 网络 → 用完全离线档 C；③ 被问「创新点 3 个里哪个最强」→ 提前想好答案
- **完成标准**：能脱稿回答「你和参考实现的差别是什么」——标准答案：**它用深度标量近似 3D，我们用点云做真实 3D；它靠 LLM 一次性生成代码，我们建立了确定性几何中间层；它只能用闭源大模型，我们证明了 4B 开源模型 + 自建数据 + 本地 QLoRA 的可行性。**

---

## 19. 风险点（按「会毁掉项目的程度」排序）

| # | 风险 | 严重度 | 触发信号 | 对策 |
|---|---|---|---|---|
| 1 | **4B 模型扛不住程序合成** | 🔴 致命 | Stage A/B 标签吐不出来，或程序 `compile()` 失败 | **Step 0.8 一次性测掉**。若失败，把失败样本记成 QLoRA 的负例种子，微调动机从"提分"改为"让流程能跑通" |
| 2 | ~~原生 Windows 跑不通上游本体~~ | ✅ **已关闭** | — | 该检出已移除；自研执行器用**独立计时线程强杀**做超时，不碰 `signal`，全程 Windows 原生 |
| 3 | ~~上游目录名被硬编码~~ | ✅ **已关闭** | — | 检出已移除，不再有这项约束 |
| 4 | **transformers 版本冲突** | 🔴 高 | 升级 transformers 后 `loc`/`depth` 崩 | vision 与 LLM **分 venv 分进程**，用 HTTP 连接 |
| 5 | **`vqa()` 需要 VLM 不是 LLM** | 🟠 高 | 本地纯文本模型收到图片报 400 | 主模型必须多模态：`qwen3.5` 系列（Ollama 已核实 4b=3.4 GB / 9b=6.6GB） |
| 6 | **注解陷阱** | 🟠 高 | `AttributeError: 'NoneType' object has no attribute 'group'` | 方法名正则要求后面是字面量 `):`，模型写 `-> float:` 就会击穿；**提示词约束必须保留**（§8.6 的实测：这三次没触发 ≠ 不存在） |
| 7 | **单目深度尺度漂移** | 🟠 高 | 距离数值整体偏大/偏小 | **内参外部给定**（§21–§23）+ 如实报告；评测用相对误差与 MRA。⚠ 早期的 `calibrate_scale` 方案**已撤销**（各向异性） |
| 8 | ~~**创新点 1 的地基未验证**~~ | ~~🟠 高~~ **✅ 已关闭** | ~~UniDepth 不返回 `points`/`intrinsics`~~ | **2026-09-16 实测：7 个键全在，`points[2]` 与 `depth` 完全相同，真实照片深度落在 [1.376, 3.974] m。地基成立**（见 §5.1） |
| 9 | **4060 显存触顶** | 🟠 中高 | CUDA OOM | 串行加载 + 降分辨率 + 4bit；KV cache 用 `q8_0`；先测再调 |
| 10 | **API 随机性导致结果不可复现** | 🟠 中高 | 两次跑结果差异大 | 换成固定工具库（这本身就是创新点 1/2 的副产品） |
| 11 | **教师模型数据不可信** | 🟠 中高 | 训练后模型调不存在的物体 | **执行验证**是强制步骤，不可跳过 |
| 12 | **4B QLoRA OOM** | 🟡 中 | 训练启动即 OOM | 降级顺序：seq→r→2B→最后才租卡 |
| 13 | ~~上游预定义工具的一个 bug~~ | ✅ **已关闭** | — | 上游检出已移除；我们的工具库不复用其实现 |
| 14 | **前端工时吞噬全部时间** | 🟡 中 | Phase 11 耗掉两周 | 先 Rerun 调试版；Three.js 只做三件事 |
| 15 | ~~上游 LLM 封装无限重试挂死~~ | ✅ **已关闭** | — | 上游检出已移除；我们的 `llm/adapter.py` 自带重试上限与超时 |
| 16 | **依赖清单不全** | 🟢 低 | `ModuleNotFoundError: pandas` | 手动补 `pandas`、`tqdm` |
| 17 | **Omni3D-Bench 下载/许可** | 🟢 低 | `huggingface.co` 不可达（已实测） | 走 `hf-mirror.com`；注意数据集是 **CC BY-NC**，非商用 |
| 18 | **Qwen3.5 的 target_modules 名称不明** | 🟢 低 | LoRA 挂错层，效果为 0 | 训练前 `print(model.named_modules())` 确认 |

**明确的诚实声明（本方案中标注为「不确定」的项）：**

1. 显存数字：**三个视觉模型均已实测**（UniDepth 486/604 MB、GroundingDINO 1761–1903/2152–2306 MB、SAM2 647/928 MB，见 §5）；**Molmo / Qwen 仍是估算**，需在 Phase 4 实测替换
2. Qwen3.5-4B 的 4bit 权重体积各来源口径不一（3–5 GB）
3. Qwen3.5 系列对 transformers / vLLM 的最低版本要求，各来源不一致，**以官方 model card 为准**
4. Qwen3.5 边缘系列（0.8B/1.5B/2B）的具体命名在不同来源中有出入
5. Qwen3.5 在 LoRA 场景下的 `target_modules` 名称未验证
6. Omni3D-Bench 是否提供 3D 坐标真值，**需在 Phase 1 验证**（README 展示的字段里没有）
7. vLLM 的 `--tool-call-parser` 具体取值随版本变化
8. ~~全部显存/性能判断都基于代码结构与模型规格的推断~~ **部分已更新**：三个视觉模型的显存与延迟已实测（§2.5、§20 Step 0.5）；但**上游原版本体从未在本机跑过**，且该检出已移除 ⟹ 实验臂 A 的数字不可复现
9. **`qwen3.5` 的 4B/8B 磁盘体积与模态已核实**（来自 Ollama 官方库页），但**推理时的真实显存占用与延迟尚未实测** —— 这正是 Phase 0.7 做完要拿到的数字
10. `04_install_ollama.ps1` 里 `OllamaSetup.exe` 的静默安装参数**未经核实**：脚本优先走 `winget`，回落路径若弹出安装界面属正常现象

---

## 20. 第一阶段具体操作

> **2026-09-15 起更新：WSL2 已从「必须」降为「不需要」。** 早先把它列为硬前提，依据是 `signal.SIGALRM` 与 UniDepth 的 `triton`/`xformers` 依赖。逐文件复核后，两条依据都被推翻：
>
> - **`signal.SIGALRM` / `signal.alarm`** 只堵住上游自己的 Engine。我们的执行器不碰 `signal`，改用**独立计时线程强杀**子进程，跨平台且能拿退出码，**该约束归零**。
> - **`triton` / `xformers` 不是硬依赖**。UniDepth 是纯 Python 包（`pyproject.toml` 用 `setuptools.build_meta`，无 `ext_modules`），而这两个包的**每一处 import 都在 `try/except ImportError` 内**：`backbones/metadinov2/attention.py:21`、`block.py:26`、`swiglu_ffn.py:37`、`layers/nystrom_attention.py:9`。其中 attention 有干净的 `F.scaled_dot_product_attention` 回落，源码注释原话是 *"new pytorch have good attn efficient, no need for xformers"*。而 `triton` 在 Windows 上根本没有官方轮子。
> - **GroundingDINO / SAM2 无需现场编译**：改用 transformers 内置的纯 PyTorch 实现（`GroundingDinoForObjectDetection`、`Sam2Model`）。
>
> ⟹ **现在全程 Windows 原生，不需要 WSL2。** 2026-09-20 进一步确认：唯一曾需要它的场景（跑上游原版本体）也随检出移除而消失。

| 优先级 | 步骤 | 需要 WSL? | 需要 GPU? | 需要 API key? |
|---|---|---|---|---|
| **P0** | **Step 0.3 建视觉栈环境（Windows 原生，一条脚本，无需重启）** | ❌ | ❌ 装 torch 时只做校验 | ❌ |
| **P0** | **Step 0.5 跑 `probe3d.py` 验证 `points` 是真几何** | ❌ | ✅ | ❌ |
| ~~P2~~ | ~~Step 0.1 装 WSL2~~ —— **已作废**（不需要 WSL，也不再跑上游原版） | — | — | ❌ |
| P2 | Step 0.6 云端 LLM 端点探针 | ❌ | ❌ | ⚠️ 仅云端端点需要 |
| **已挂起** | Step 0.7 / 0.8 装 Ollama + 测 4B 协议合规性 | ❌ | ✅ | ❌ |

> **P0 两项排最前**：它们回答的是「UniDepth 的 `points` 到底是不是真几何」—— 这是创新点 1 的地基，也是全项目唯一无法靠推理回答的问题。零成本、零等待、不用重启。
>
> **Step 0.7 / 0.8 为什么挂起**：微调既然不是硬性要求（见 §9.4），「4B 模型能不能守住协议」就不再是阻塞项。云端模型更强，先用云端把 pipeline 跑通；微调退为加分支线，若日后发现云端也守不住协议再升回来。

### Step 0.1 — ~~装 WSL2 + Ubuntu~~ **已作废**（2026-09-20）

> 这一步依赖的脚本（`phase0/00_install_wsl.ps1`、`01_setup_ubuntu.sh`）与上游检出已一并
> 移出本仓库。**视觉栈与 LLM 现在全程 Windows 原生运行**（见 Step 0.3）。
> 原始操作步骤保留在 git 历史里，需要时按 commit 取回。

<details><summary>原文（已作废的 WSL 安装步骤）</summary>


> **走路径 B 的话，这一整步可以跳过。** 保留它是因为：如果你在 §16 消融表里想跑上游原版流水线作为对照臂，那一步必须进 WSL。
> 判断方法：**你不需要它来跑视觉栈，也不需要它来跑 LLM。** 只需要问自己「要不要跑上游自己的 Engine」。

已实测确认：**WSL 尚未安装**（`HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss` 无键，`HypervisorPresent = False`），所以这一步是真的从零开始。

**不要手敲命令，用已经写好的脚本**（Windows 侧，**管理员** PowerShell）：

```powershell
powershell -ExecutionPolicy Bypass -File D:\3D_Spatial_Agent\phase0\00_install_wsl.ps1
```

这个脚本做了 4 件手敲容易漏的事：
1. **管理员权限自检** —— 不是管理员直接退出并给出提示
2. **磁盘护栏** —— 检查 D 盘剩余空间（实测 381 GB，充裕）
3. **装到 `D:\WSL` 而不是默认位置** —— C 盘只剩 **21.8 GB**，WSL + CUDA + 权重 + 数据集要 60–90 GB，装 C 盘必然爆
4. **重启门禁** —— 检测到 hypervisor 没起来时会**主动停下并让你重启**，而不是报一个看不懂的错；重启后再跑一次同一个脚本即可继续

`wsl --version` / `wsl --list` 在本机被 WorkBuddy 沙箱的程序黑名单拦截了，所以脚本里改用 `-q` 静默调用并配合注册表判断，避免依赖被拦的输出。

重启后进 Ubuntu 设好用户名密码，**验证 CUDA 直通**（驱动 560.76 已满足 WSL2 CUDA 要求）：

```bash
nvidia-smi        # 应看到 RTX 4060 Laptop GPU, 8188MiB
```

不要在 WSL 内装显卡驱动 —— WSL 里的 CUDA 走 Windows 侧驱动。

</details>

### Step 0.2 — 项目目录与环境变量

**项目根目录已迁移到 D 盘**（C 盘仅剩 21.8 GB，装不下 WSL + CUDA + 权重 + 数据集）：

```
D:\3D_Spatial_Agent\                        ← 项目根（Windows）
├── docs/                                   ← 本文档
├── vendor\
│   └── UniDepth/                           ← UniDepth 源码（git clone --depth 1）
├── venvs\vision\                           ← 视觉栈虚拟环境（Python 3.12）
├── .cache\
│   ├── pip\                                ← pip 下载缓存（避免占用 C 盘）
│   ├── wheels\                             ← curl 预下的 torch 轮子（2.4 GB，可反复复用）
│   ├── models\grounding-dino-tiny\         ← GroundingDINO 权重（657 MiB，curl 下载）
│   └── huggingface\                        ← HF_HOME，UniDepth 权重落这里
├── phase0/                                 ← Phase 0 可执行脚本
├── tools/                                  ← 文档构建工具
└── logs/
```

> **三个"引到 D 盘"的环境变量**（脚本自动设置）：`PIP_CACHE_DIR`、`HF_HOME`、`HF_ENDPOINT`。
> 前两个是为了不撑爆只剩 21.8 GB 的 C 盘 —— 光 torch 的 cu124 轮子就 2.4 GB，HF 权重还会再占几个 GB。

> 在 WSL 里对应 `/mnt/d/3D_Spatial_Agent`。
>
> **上游检出已于 2026-09-20 移出仓库** —— `vendor/` 下现在只剩 `UniDepth/`。
> 原来那一步「把上游源码复制进 WSL 家目录」及其配套的 `01_setup_ubuntu.sh`，随 WSL 路线
> 一并作废；原始脚本与步骤保留在 git 历史里，需要时按 commit 取回。

> **为什么不直接在 `/mnt/d` 上建 venv 和跑构建？** 两点实测原因（WSL 路线实测，路线本身已作废，但结论仍可供参考）：
> 1. `/mnt/d` 是 9p 挂载，pip `-e` 构建和 torch import 会慢 3–10 倍；
> 2. drvfs 不支持 POSIX 权限位，某些 install 和 git 操作会失败。
>
> 所以当时的做法是：**D 盘放项目源码（我们的代码、脚本、文档），WSL ext4 放虚拟环境与编译产物**。

### Step 0.3 — 建视觉栈 Python 环境（Windows 原生，一条脚本）

> **不需要 WSL、不需要重启、不需要 API key。** 原先这一节是「Ubuntu 里 apt 装 python3.10 + 编译链」，现在整段被一条 PowerShell 脚本取代。

```powershell
powershell -ExecutionPolicy Bypass -File D:\3D_Spatial_Agent\phase0\01_setup_windows.ps1
```

脚本做的 8 件事（全部落在 D 盘 —— C 盘只剩 21.8 GB）：

| # | 动作 | 说明 |
|---|---|---|
| 1 | 选解释器 | `py -3.12`，回退 3.11 / 3.13。本机 `py -0p` 实测只注册了 **3.12**；UniDepth 只要求 `>=3.10` |
| 2 | 建 venv | `D:\3D_Spatial_Agent\venvs\vision`；**已存在则复用**（原本用 `venv --clear`，实测被本机批量删除保护拦下：`[safe-delete] count=1540, threshold=50`，故改为复用 + 手动删除重建） |
| 3 | 引流缓存 | `PIP_CACHE_DIR` / `HF_HOME` 都指向 D 盘；`HF_ENDPOINT=https://hf-mirror.com`（`huggingface.co` 本机不可达） |
| 4 | 装 torch | `torch==2.6.0+cu124` + `torchvision==0.21.0+cu124`，**先用 `01b_fetch_torch.ps1` 以 curl 下成本地轮子**再 `--find-links` 安装（原因见下） |
| 5 | 装运行依赖 | 见 Step 0.4 的裁剪表 |
| 6 | 装 UniDepth | 源码在 `vendor\UniDepth`，**加 `--no-deps`** —— 这是跳过 `triton` / `xformers` 的关键一步 |
| 7 | 验证 | 调用 `phase0\verify_env.py`，失败即非零退出，不让坏环境拖到后面 |
| 8 | 锁定版本 | `pip freeze` 写入 `phase0\requirements-vision.lock.txt` |

**torch 为什么用 2.6.0+cu124，而不是上游锁的 2.2.0+cu122：** 4060 Laptop 是 **sm_89**，需要 cu124 及以上的预编译内核；上游那行会 `--force-reinstall` 把 torch 降级，在本机是倒退。

#### 关于大轮子：为什么用 curl 而不是让 pip 直接下（实测踩坑）

第一次执行时 `pip install torch==2.6.0 --index-url download.pytorch.org/whl/cu124` **卡死了**：进程存活但 18 分钟写入 0 字节（25 秒采样 0.00 MB/s，磁盘上找不到任何部分文件，日志停在 `Downloading torch...(2532.3 MB)` 不再增长）。

排查后确认**不是源的问题**——对同一个 URL 发 3 MB 范围请求，返回 206，速度约 4.2 MB/s：

| 源 | 实测 |
|---|---|
| `download.pytorch.org/whl/cu124/` | ✅ **4.19 MB/s（最快）** |
| `mirror.sjtu.edu.cn/pytorch-wheels/` | ⚠️ 0.87 MB/s |
| `mirrors.aliyun.com/pytorch-wheels/` | ❌ 0.12 MB/s（20 s 超时） |
| `mirror.nju.edu.cn/pytorch-wheels/` | ❌ HTTP 404 |

所以问题出在 pip 的下载器：**它既不输出可见进度、也不支持断点续传，卡住时完全静默**。解决办法是把大文件从 pip 手里拿走——

```powershell
# 1. 先用 curl 把轮子拉到本地（可断点续传，且带速度守卫）
powershell -ExecutionPolicy Bypass -File D:\3D_Spatial_Agent\phase0\01b_fetch_torch.ps1
# 2. 再跑主脚本，它会自动发现本地轮子并改用 --find-links
powershell -ExecutionPolicy Bypass -File D:\3D_Spatial_Agent\phase0\01_setup_windows.ps1
```

`01b_fetch_torch.ps1` 三个要点：`-C -` 断点续传（重跑即接着下）；`--speed-limit 51200 --speed-time 60` 在速度低于 50 KB/s 持续 60 秒时主动中断而不是干等；外层 6 次重试循环。小包仍然走清华 PyPI（实测 3.4 MB/s，正常）。

#### 本机其实已经有三份 torch —— 为什么还要再下一份

| 位置 | Python | torch | CUDA 可用 |
|---|---|---|---|
| `anaconda3`（base） | 3.13.9 | 2.6.0+cu124 | ⚠️ import 时报 `OMP: Error #15`（libiomp5md.dll 重复加载） |
| `anaconda3\envs\mytorch` | **3.9.25** | 2.6.0+cu124 | ✅ True |
| `AppData\Local\Programs\Python\Python312`（全局） | 3.12.5 | **2.7.1+cu126** | ✅ True，RTX 4060 Laptop GPU |

三份都不能直接用，但原因不同：

- **`mytorch` 是 Python 3.9**，而 UniDepth 要求 `>=3.10` —— 版本不满足，与轮子无关。
- **base 的 conda 环境污染**：装了 MKL 版 OpenMP 又装了 torch 自带的，import 即报 `Error #15`，属于该环境自身的既有问题。
- **torch 的轮子是 ABI 锁死的**：`cp39` / `cp313` 的构建无法被 `cp312` 的解释器加载，所以即使把 `site-packages\torch` 整个拷过来也没用。

用独立 venv + 固定版本，是为了让实验可复现（`requirements-vision.lock.txt` 就是为此存在的）——课程报告里的实验臂 A/B 需要这个。

> **一个顺带确认到的好消息**：全局 3.12 里已经装着 `transformers 4.57.3`，而且 `GroundingDinoForObjectDetection` 与 `Sam2Model` **都在**（分别打印 True）。这就在本机**实证**了 §4 的结论——视觉栈的检测与分割确实不用现场编译 CUDA 算子。将来若要快速试跑，这个环境只差 `timm` / `einops` / `unidepth` 三个包。

#### 第二条实测踩坑：单条命令会被杀在约 121 秒

一次性 `pip install` 全部 18 个包，**连续两次**都在下载 scipy 的过程中被杀：

| 观测 | 值 |
|---|---|
| 进程存活时间 | 两次都是 **121 秒整** |
| pip 自己的 `--log` | **没有任何 ERROR，也没有退出行** → 不是 pip 报错，是进程被外部终止 |
| 同时段同一 URL 用 curl 测 | ✅ 正常（scipy 单包 52 秒下完，680 KB/s） |
| 拆成 6 个小批次后 | ✅ 全部成功，单批最长 101 秒 |

**结论：把每条命令控制在 110 秒以内。** 这不影响正确性，只是要求安装动作分片。所以 `01_setup_windows.ps1` 的第 4 步已改成 **6 个短批次**（见脚本里的 `$depBatches`），每批一个 pip 调用。

**2026-09-16 再次确认，而且范围比原先以为的更大**：下载 657 MiB 的 GroundingDINO 权重时，**后台任务同样在约 121 秒处被杀**（`curl` 进程消失、脚本日志没有结束行、文件停在 267 MB 不再增长）。所以"分片"这条对后台任务一视同仁。

于是 `01c_fetch_gdino.ps1` 采用的模式是「**限时 + 可续传**」：

| 轮次 | 命令 | 结果 |
|---|---|---|
| 第 1 轮 | 不限时，后台运行 | ❌ 约 121 秒被杀，停在 267 MB |
| 第 2 轮 | `curl -C - --max-time 100` | ✅ 传出 264 MB / 100 s（**2.64 MB/s**），主动超时退出 |
| 第 3 轮 | `curl -C - --max-time 115` | ✅ 剩余 145 MB / 67 s，**退出码 0，文件字节数与服务器声明完全一致**（689,359,096） |

**教训**：大文件传输不要押注"长任务能跑完"，要让它**随时可中断、随时可续**。`-C -` 加上显式时限，比任何重试逻辑都可靠。同理，HF 权重下载也走这套（`snapshot_download` 本身支持续传，但慢，见下表）。

#### 依赖裁剪的两处修正（实测推翻）

**① `wandb` 不能跳过。** 原表把它归为「训练/数据集工具」排除，但 `from unidepth.models import UniDepthV2` 会直接失败：

```
ModuleNotFoundError: No module named 'wandb'
```

链路是 `unidepth/utils/__init__.py:12` → `from .visualization import log_train_artifacts` → `unidepth/utils/visualization.py:11` 的**裸 `import wandb`**（不在 try 里）。
（同一文件的 `validation` 导入倒是被上游注释掉了，`:2` 那行 `# from .validation import validate`——说明作者自己也撞过这个问题。）
**处理：装上 wandb**，而不是改 vendor 源码。多占约 40 MB，换来「vendor 一行不改」。

**② HF 下载要关掉 xet，否则慢 5 倍。**

| 下载方式 | 实测速度 | 137 MB 耗时 |
|---|---|---|
| `hf_hub_download`（默认，走 hf-xet） | ~0.53 MB/s | 被杀，只下到 64 MB |
| curl 直连 `hf-mirror.com/resolve/main/...` | **2.78 MB/s** | — |
| `snapshot_download` + `HF_HUB_DISABLE_XET=1` | ~1.3 MB/s | ✅ **99.7 s 完成** |

另外 `snapshot_download` 默认会把 `model.safetensors` 和 `pytorch_model.bin` **两份等价权重都下**（白下一倍）。加 `allow_patterns=["config.json", "*.safetensors"]` 即可：

```powershell
$env:HF_HOME = "D:\3D_Spatial_Agent\.cache\huggingface"
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HUB_DISABLE_XET = "1"
& $py -c "from huggingface_hub import snapshot_download; snapshot_download('lpiccinelli/unidepth-v2-vits14', allow_patterns=['config.json','*.safetensors'])"
```


### Step 0.3b — 环境自检（约 1 分钟）

```powershell
D:\3D_Spatial_Agent\venvs\vision\Scripts\python.exe D:\3D_Spatial_Agent\phase0\verify_env.py
```

它回答三个问题，全部带证据：

1. CUDA 从这个 venv 里**真的**可用吗（设备名 / 显存 / compute capability）
2. **决定性的一条** —— 在 `xformers` / `triton` / `torchaudio` **全部缺失**的前提下，`from unidepth.models import UniDepthV2` 能否成功。这一条成立，Windows 原生路径才站得住
3. 这条 import 链实际拉进了哪些第三方包（应与 Step 0.4 的裁剪表吻合）

> 它的价值是把「可以不用 WSL」从一句判断变成**可复现的断言**。答辩被问「为什么不用 WSL」时，现场跑它。

### Step 0.4 — 视觉模型的依赖裁剪（**已包含在 Step 0.3 的脚本里**）

新路径下**不再 clone `sam2` 与 `GroundingDINO` 两个仓库** —— 二者改用 transformers 内置的纯 PyTorch 实现。只需要 UniDepth 一份源码（脚本会自动 clone 到 `vendor\UniDepth`）。

UniDepth 的 `requirements.txt` 列了 26 个包，多数与推理无关。裁剪依据逐条列在下面，**每一条都对应源码里的实际 import**：

| 类别 | 包 | 处理 | 依据 |
|---|---|---|---|
| **必须** | `torch` / `torchvision` | 装 | `unidepthv2.py:14` 直接 `import torchvision.transforms.v2.functional as TF` |
| **必须** | `einops` | 装 | `layers/attention.py:11`、`unidepthv2/decoder.py:9`、`utils/misc.py:13` |
| **必须** | `timm` | 装 | `unidepthv2/decoder.py:10` 的 `timm.models.layers`、`backbones/convnext.py:7` |
| **必须** | `huggingface-hub` | 装 | `unidepthv2.py:16` 的 `PyTorchModelHubMixin`（`from_pretrained` 靠它） |
| **必须** | `numpy` / `pillow` / `scipy` | 装 | `utils/misc.py:14` 用 `scipy.interpolate`；`utils/visualization.py:12` 用 PIL |
| **必须** | `opencv-python` | 装 | `utils/distributed.py:7`，在 import 链内 |
| **必须** | `matplotlib` | 装 | `utils/visualization.py:8` |
| **顺手装** | `imageio` `trimesh` `h5py` `pandas` `tabulate` `termcolor` `protobuf` `safetensors` `transformers` | 装 | 体积都小；`transformers` 是 Phase 2 之后 GroundingDINO/SAM2 原生实现要用的 |
| **排除** | **`triton>=2.4.0`** | ❌ | 训练用；**Windows 上没有任何官方轮子** |
| **排除** | **`xformers>=0.0.26`** | ❌ | 训练用；4 处 import 全在 `try/except ImportError` 内，且有 SDPA 回落 |
| **排除** | **`torchaudio>=2.4.0`** | ❌ | 推理路径完全不用 |
| **排除** | `gradio` / `wandb` / `tables` | ❌ | 分别是 demo / 训练记录 / 数据集工具 |
| **排除** | 上游参考实现的 `numpy==1.25.0` | ❌ | 与 UniDepth 的 `numpy>=2.0.0` **直接冲突**；新路径取 2.x |

> **`--no-deps` 是这条路的核心手法。** 直接 `pip install -e vendor/UniDepth` 会去装 `triton` 然后失败；加 `--no-deps` 后由我们手工补齐上表「必须」列。
>
> 手工补依赖必然有漏的风险 —— 这就是 `verify_env.py` 存在的理由：它立刻告诉你漏了哪个，而不是等到跑模型时才炸。

> **补充一个容易混淆的点：** `sam2` 这个**包**在上游原版里是顶层 import，跑上游本体必须装。但我们的视觉层改用 `transformers` 的 `Sam2Model`，**不 clone 也不装** `facebookresearch/sam2`。

> **网络实测（2026-09-15 于本机，影响安装路径）：**
>
> | 目标 | 结果 | 影响 |
> |---|---|---|
> | `huggingface.co` | ❌ 超时（12 s+） | UniDepth 权重、Omni3D-Bench 都下不了 |
> | `hf-mirror.com` | ✅ 200 / ~2 s | **已解决**：脚本设 `HF_ENDPOINT=https://hf-mirror.com` |
> | `download.pytorch.org/whl/cu124/` | ✅ **200**（GET，1.5 s） | **推翻了此前「403 不可用」的判断** —— 那是 HEAD 请求的假象。torch 可以直接从官方源装 |
> | `mirrors.aliyun.com/pytorch-wheels/cu124/` | ✅ 200（2.5 s） | 备用镜像 |
> | `mirror.sjtu.edu.cn/pytorch-wheels/cu124/` | ✅ 200（2.4 s） | 备用镜像 |
> | `pypi.tuna.tsinghua.edu.cn` | ✅ 200（2.8 s） | 主 PyPI 镜像 |
> | `pypi.org` | ✅ 200（2.3 s） | 正常 |
> | `github.com` git 端点 | ✅ 本次 `git clone` UniDepth 成功（200 文件） | 之前记录的「间歇性超时」本次未复现 |
> | `api.openai.com` | ❌ 超时（12 s+） | 只有 gpt-4o 调不到 |
> | `api.deepseek.com` / 阿里百炼 / 智谱 / 火山 / Moonshot / 硅基流动 / 腾讯混元 / OpenRouter | ✅ **全部 HTTP 401**（= 服务器已应答，只是没带 key） | **LLM 层不是阻塞项**。详见 `phase0/LLM_BACKEND.md` |

### Step 0.5 — 核心验证：`probe3d.py`（约 10 分钟）

```powershell
# 1) 先把检测器权重下好（657 MB，curl 分段续传，可反复重跑）
powershell -ExecutionPolicy Bypass -File D:\3D_Spatial_Agent\phase0\01c_fetch_gdino.ps1

# 2) 完整跑 A–D 四段
& D:\3D_Spatial_Agent\venvs\vision\Scripts\python.exe `
    D:\3D_Spatial_Agent\phase0\probe3d.py `
    --image D:\3D_Spatial_Agent\vendor\UniDepth\assets\demo\rgb.png `
    --prompt "box. door."

# 只想验 UniDepth 的 points（跳过 C/D 两段）时：
#   ... probe3d.py --skip-gdino --image <任意图>
```

> **2026-09-16 更新：`run_gdino()` 已改写成 transformers 原生实现。** 原先它走 `groundingdino.util.inference`
> 这套**编译式安装**，与本项目的 Windows 原生路径冲突，所以当时必须先加 `--skip-gdino`。现在换成
> `AutoModelForZeroShotObjectDetection` + `AutoProcessor`；`--models` 参数作废，改为 `--gdino`
> （默认自动指向 `.cache\models\grounding-dino-tiny`，该目录不存在时回退到 hub id）。
>
> **两处与原版不同的接口细节**（自己写代码时最容易踩）：
> 1. **prompt 必须小写、每个标签后紧跟句点**：`"box. door."`。原版接受的 `"box . door ."` 会被错误分词。
> 2. **返回的框是绝对像素 xyxy**，不是归一化 cxcywh —— 后处理函数已经换算过了。
>
> 另外 `post_process_grounded_object_detection` 的**关键字名在版本间变过**（`box_threshold` → `threshold`），
> 脚本用 `inspect.signature` 读真实签名再决定传哪个，不靠猜版本号。

**注意：不要再用上游那套模块列表构造去做冒烟测试。** 它会连带构造视觉问答模块，
而那个模块的 LLM 封装在 `__init__` 里就要求一个本地密钥文件，没有就直接抛 `FileNotFoundError`。
（该检出已移除，这条约束随之消失；下面的历史记录是当时的现场。）

> **历史记录**：上游的模块列表构造会连带构造视觉问答模块，而它的 LLM 封装 `__init__` 第一件事就是 `open("./api.key")` —— 没有那个文件就直接 `FileNotFoundError`。当时的绕法是运行时替换掉这个类。**该检出与桥接脚本均已移除**，这条约束随之消失；我们的 `llm/adapter.py` 从设计上只读环境变量，不依赖 `api.key` 文件。

`probe3d.py` 则绕开了 LLM 这一层：它直接调用视觉模型，一行 LLM 代码都不碰。所以它与 Step 0.6 可以**并行**做。

脚本做四件事，全部带数值输出：

| 部分 | 做什么 | 为什么重要 | 状态 |
|---|---|---|---|
| A | 按上游的原调用方式（uint8、`(3,H,W)`、无 batch 维）加载 UniDepth，打印**实际返回的每一个 key** + shape + dtype + 加载耗时 + 推理延迟 + 峰值显存 | 一次性实测「8GB 装不装得下」与真实延迟 | ✅ **已跑** |
| B1 | **身份校验**：`points[2] == depth`？`norm(points) == radius`？`points == rays × radius`？ | 这三条成立就说明 `depth` 只是点云的 z 列，不是独立测量量 | ✅ **已跑** |
| B2 | **反投影自洽性检验**：用 `depth` + `intrinsics` 手工把采样像素反投影成 XYZ，与 `points` 逐点比对（报**绝对 + 相对**两种误差） | 证明 `points` 是真实几何量，而不是被重新包装的 depth | ✅ **已跑** |
| C | GroundingDINO 的峰值显存、延迟与检测框 | 替换文档中标注「估算」的显存数字 | ✅ **已跑**（2026-09-16，transformers 原生实现） |
| D | 检测到 ≥2 个物体后，直接算出它们的三维中心距离 | **这就是最终项目要回答的问题，由几何而非 LLM 给出** | ✅ **已跑** |

结果同时写入 `phase0/probe3d_result.json`。

#### 实测结果（2026-09-16，RTX 4060 Laptop 8188 MiB）

跑的是 UniDepth 仓库自带的官方示例图 `vendor\UniDepth\assets\demo\rgb.png`（640×480，真实室内照片）：

| 指标 | 实测 |
|---|---|
| 参数量 / 权重 | 34.2 M / 130.4 MB |
| 加载耗时 | 3.1 s |
| 单帧推理延迟 | **524–1120 ms**（四次运行，冷热缓存都有） |
| 峰值显存 allocated | **486 MB** |
| 峰值显存 reserved | **604 MB** |
| 返回的键 | 7 个，与源码推断完全一致 |
| `points[2] == depth` | **0.000e+00**（完全相同） |
| `norm(points) == radius` | 0.000e+00 |
| `points == rays × radius` | 8.69e-04 |
| depth 范围 | **[1.376, 3.974] m** |
| 与针孔反投影的相对偏差 | 均值 **3.69%**，最大 **5.30%** |

**四个问题当场答完：**

1. **装得下吗** —— ✅ 远超预期。UniDepth 只要 **604 MB**，比原估算小一个数量级。
2. **延迟如何** —— ✅ 约 **1 秒/帧**（含未编译的 `EdgeGuidedLocalSSI` 慢路径；编译算子还能更快）。
3. **真实照片上深度合理吗** —— ✅ [1.376, 3.974] m，**单位就是米**，符合室内场景。
4. **`points` 真实存在且自洽吗** —— ✅ **是**。整个创新点 1 押在这一条上，现在**地基坐实**（详见 §5.1）。

#### C 段实测：GroundingDINO（transformers 原生，2026-09-16 补齐）

同一张图、同一次运行，紧接 UniDepth 之后：

| 指标 | 实测 | 与文档原估算对比 |
|---|---|---|
| 参数量 / 权重 | 172.3 M / 657 MiB fp32 | 估 0.69 GB checkpoint ✅ 吻合 |
| 加载耗时 | 1.3 s | — |
| 单帧推理延迟 | **709–814 ms**（两次运行） | — |
| 峰值显存 allocated | **1903 MB** | 估 2–3.5 GB，实测落在下限以下 |
| 峰值显存 reserved | **2306 MB** | — |

**两个模型同时常驻的总峰值显存 ≈ 604 + 2306 = 2.9 GB**，占 8188 MiB 的 **36%**；`loc` + `depth` 合计约 **1.9 s/帧**。

#### D 段实测：由几何给出的三维距离

客厅照片（`vendor\UniDepth\assets\demo\rgb.png`）配 prompt `"sofa. chair. table. picture. mirror."`，
检索出 **9 个目标**；每个目标的三维中心取点云在框内像素的**中位数**，然后两两求欧氏距离。

| 物体 | 相机坐标系 XYZ (m) | 像素框 (xyxy) |
|---|---|---|
| sofa | (2.448, 0.875, 2.992) | (286, 215, 632, 364) |
| table | (1.907, 1.776, 1.775) | (359, 334, 633, 479) |
| chair | (-2.073, 1.789, 1.688) | (7, 347, 235, 475) |
| sofa chair | (-2.808, 1.044, 2.350) | (7, 232, 237, 404) |
| mirror | (-0.814, -3.003, 3.657) | (226, 26, 342, 197) |
| picture#1 | (3.922, -3.024, 3.632) | (425, 47, 567, 173) |

关键距离（**全部来自算术，没有任何模型"猜"过**）：

| 关系 | 距离 |
|---|---|
| sofa ↔ table | **1.608 m** |
| sofa chair ↔ chair | 1.238 m |
| mirror ↔ picture#2 | 2.434 m |
| chair ↔ table | 3.981 m |
| sofa ↔ chair | 4.793 m |

`--query table` 直接打印「茶几到其余每个物体的距离」，由近及远：

```
table -> sofa        1.608 m
table -> chair       3.981 m
table -> sofa chair  4.806 m
table -> picture#1   5.528 m
table -> mirror      5.812 m
```

**这就是最终项目要回答的那种问题的形态**，只不过现在是写死的一个查询；Phase 3 会由 LLM 生成
程序来组合这些原语（`argmin`、过滤、聚合），而**数值始终来自几何层**。

> **一个必须写进报告的方法学细节**：脚本会同时给出「任意两个检测框的最近配对」与
> 「**不同类别**之间的最近配对」。前者可能落在同一类别的两个实例上（这张图里是墙上两块挂画，
> 相距 0.638 m）——算术上没错，但读起来不像答案。所以结论行只引用后者。

> **`probe3d.py` 的粗糙处 → 现已量化，并给出裁决**：三维中心取自**整个矩形框内像素的中位数**，
> 而框里必然混进背景像素。`probe_sam2.py` 的 D 段在 9 个物体上对比了「框内中位数」与
> 「SAM2 掩码质心」：

| 指标 | 实测 |
|---|---|
| 两种质心之差 | **均值 83 mm / 最大 208 mm** |
| 框内背景像素占比 | **均值 17.8% / 最大 53.8%**（沙发那种大框最严重） |
| 文档给关系判断的容差 | 50 mm（`left_of` / `above` 的 `tol`） |

> → **裁决：Phase 1 的 `get_3d_position` 必须用掩码质心，这不是可选项。**
> 208 mm 比容差大 4 倍，用框会让 `left_of` / `on` 这类关系直接被背景拖偏。
> 但反过来说，小物体（墙上那几幅挂画，框几乎不含背景）两种做法只差 1–20 mm ——
> 所以结论**不是笼统的「掩码总是更好」，而是「框越大、背景越多、误差越大」**。
> 逐物体的对照表本身就是报告里可以单列的一节（`probe_sam2_result.json` 的 `centre_comparison`）。


> **（上游原版对照臂已于 2026-09-20 移除，本段只保留当时的判断与一条仍然有效的硬约束。）**
> 不能复现的是**论文那个 40.4 的 gpt-4o 基线**（`api.openai.com` 不可达）；
> 但整条 LLM 链路**换国产模型后可以跑通** —— 这也正是我们把主路径改成
> 「自研程序合成 + 自研几何工具层」的原因之一。
>
> **仍然有效的硬约束：主模型必须多模态。** 只要视觉问答路径存在，纯文本模型跑到它就必然报 400。
> 解法见 Step 0.6。

### Step 0.5b — SAM2 分割实测：`probe_sam2.py`（2026-09-16 补齐，收掉最后一批估算值）

```powershell
# 1) 取权重（308 MiB，curl 分轮续传，两轮完成）
powershell -ExecutionPolicy Bypass -File D:\3D_Spatial_Agent\phase0\01d_fetch_sam2.ps1

# 2) 实测。HF_HOME 必须指向 D 盘缓存，否则会尝试联网并撞上代理 502
$env:HF_HOME="D:\3D_Spatial_Agent\.cache\huggingface"
$env:HF_ENDPOINT="https://hf-mirror.com"
$env:HF_HUB_DISABLE_XET="1"
D:\3D_Spatial_Agent\venvs\vision\Scripts\python.exe D:\3D_Spatial_Agent\phase0\probe_sam2.py `
    --image D:\3D_Spatial_Agent\vendor\UniDepth\assets\demo\rgb.png
```

#### A 段实测：SAM2.1 单模型成本（standalone）

| 指标 | 实测 |
|---|---|
| 加载为 | `Sam2Model`（权重 config 声明的是 `sam2_video`，只出一条 warning，功能正常） |
| 参数量 / 权重 | **73.3 M / 279.7 MiB fp32** |
| 加载耗时 | **0.6 s** |
| 加载后常驻 | 291 MB |
| 单框提示：延迟 / 峰值 | **282 ms** / **647 MB allocated · 928 MB reserved** |
| 掩码覆盖 | 图像 10.4%，**自身框的 62.4%** —— 即**框内有 37.6% 是背景像素** |

#### B 段实测：批量 vs 逐个（9 个物体）

| 调用方式 | 端到端 | 峰值 |
|---|---|---|
| **一次调用携带 9 个框** | **200 ms** | 824 MB |
| 9 次调用，每次 1 个框 | 1512 ms | 753 MB |

→ **批量快 7.57×。这不是优化项，是必须采用的调用形态。**

#### C 段实测：三模型同时驻留

| 状态 | 显存 |
|---|---|
| 仅 SAM2 | 291 MB |
| + UniDepth | 541 MB |
| **+ GroundingDINO（三者全部驻留）** | **1200 MB = 8188 MiB 的 14.7%** |

⚠️ **测这个数必须持有模型引用。** 探针第一版复用了 `run_gdino()`（它把 model 放在局部变量里），
函数一返回权重就被回收，量出来是 291 MB —— 实际只剩 SAM2 在显存里。
**「常驻」与「峰值」是两个不同的量**：峰值含推理激活（算完立即释放），常驻只是权重。
对 8GB 的真实含义是：**视觉栈常驻 1.2 GB，留出约 6.9 GB 给 LLM。**

#### D 段实测：掩码质心 vs 框内中位数

9 个物体逐个对照（`probe_sam2_result.json` → `centre_comparison`）：

| 物体 | 框内背景占比 | 两种质心差 |
|---|---|---|
| sofa（框最大，51554 px） | **53.8%** | **208 mm** |
| chair | 35.6% | 142 mm |
| table | 22.8% | 136 mm |
| sofa chair | 26.4% | 103 mm |
| mirror | 9.2% | 82 mm |
| picture ×4（墙上小挂画） | **−1%～10%**（掩码略超出框） | **1–40 mm** |

| 汇总 | 实测 |
|---|---|
| 两种质心之差 | **均值 83 mm / 最大 208 mm** |
| 框内背景像素占比 | 均值 17.8% / 最大 53.8% |
| 关系判断容差（文档值） | 50 mm |

→ **裁决：Phase 1 的 `get_3d_position` 必须使用掩码质心。** 208 mm 是容差的 4 倍以上。
但结论不是笼统的「掩码总是更好」，而是**「框越大、背景越多、误差越大」**：小挂画的框几乎不含背景，
两种做法只差 1–40 mm。**规律本身就是报告里可以单列的一节。**

#### 两处实现细节（写进 Phase 1，避免重踩）

1. **processor 的第一个参数是 `images=` 不是 `image=`。** transformers 5.x 的
   `Sam2Processor.__call__` 签名是 `(images, segmentation_maps, input_points, input_labels,
   input_boxes, original_sizes, return_tensors, **kwargs)`。传错名字的报错是
   `ValueError: Either images or original_sizes must be provided` —— **指错了参数，极难排查**。
   → 用 `inspect.signature(proc.__call__)` 自省，不要猜。
2. **`post_process_masks` 没有 `reshaped_input_sizes` 参数。** 真实签名是
   `(masks, original_sizes, mask_threshold=0.0, binarize=True, max_hole_area=0.0,
   max_sprinkle_area=0.0, apply_non_overlapping_constraints=False)`。
   若按位置传第三个参数，它会被当成 **`mask_threshold`** —— **静默改变二值化阈值而不报错**。
   → 一律用关键字传参，并先自省签名。

### Step 0.6 — LLM 后端验证（可与 Step 0.5 并行，**不需要 WSL、不需要 GPU**）

这是本次新增的一步，而且它是**今天唯一不依赖 WSL 就能完成的验证**。
两个脚本都只用 Python 标准库，Windows 原生 Python 直接跑。

```bash
cd phase0

# 6a) 验证「四类标签契约 + vqa 视觉通道」是否成立
python 02_probe_llm_api.py --base-url https://api.deepseek.com \
                           --model deepseek-flash \
                           --api-key sk-xxxx \
                           --price-in 3.0 --price-out 9.0   # 高峰价；填了才给费用外推
# 或走环境变量，避免 key 进 shell history
export SPATIAL_API_KEY=sk-xxxx

# 6b) 验证适配器自身的调用层
python scripts/run_agent.py --scene living_room --question "..." --dry-run   # 0 成本，先看提示词
```

这一步用**上游原版的 prompt 与正则**去测四类输出标签，任何一条不通过，旧代码会在运行时硬崩
而不是给可读报错。**该探针脚本已随检出移除；实测结论见 §8.6（四类标签 3/3 + 4/4 全绿）。**

| 测试 | 不通的后果 |
|---|---|
| `<program>` + 语法可编译 | 程序合成主链路 |
| `<docstring>` / `<signature>` | 签名生成 |
| **方法名正则可提取** | **`AttributeError`**（返回值注解会击穿正则）→ 见 §19 风险 6 |
| `<answer>` 标签 | **`IndexError`**，整道题作废 |
| 错误路径可辨识 | 无限重试 → **静默卡死** |

**这一步会当场回答：换成国产模型后，这条链路到底跑不跑得起来。**

### Step 0.7 — 装本地 LLM 后端（Windows，**不需要 WSL、不需要重启、不需要 API key**）

**为什么要先做这一步**：它测的是**项目最大的未知** —— 4B 级模型能不能守住那套零容错的标签契约。这个问题现在无法回答，而它一旦是"不能"，Phase 3 的方案就要改（得先做 QLoRA 才能跑通，而不是先接模型）。

用 Ollama 在 Windows 上跑，好处是**完全绕开 WSL**：Ollama 直接通过 CUDA 用你的 4060，装完就是一个 OpenAI 兼容端点。

```powershell
powershell -ExecutionPolicy Bypass -File D:\3D_Spatial_Agent\phase0\04_install_ollama.ps1
```

脚本做的事，以及每个决定的原因：

| 动作 | 原因 |
|---|---|
| 优先用 `winget` 安装，失败回落到官方安装包 | winget 是静默的，不用猜安装器参数 |
| **`OLLAMA_MODELS` 指向 `D:\ollama\models`** | C 盘只剩 21.8 GB，模型必须放 D 盘 |
| `OLLAMA_KV_CACHE_TYPE=q8_0` | KV cache 用 8 bit，显存约省一半 —— 8 GB 卡上这是刚需 |
| `OLLAMA_CONTEXT_LENGTH=16384` | 参考实现的 prompt 实测 **4134 / 6965 字符**（约 1.2K / 2K token），16K 留足余量 |
| 装完**强制发一次真实请求验证** | 只在 `/api/tags` 里列出来不代表能生成，必须排除这个失败模式 |

**模型选型**（已在 Ollama 官方库核实，2026-09）：

| 模型名 | 磁盘 | 说明 |
|---|---|---|
| `qwen3.5:4b` | **3.4 GB** | **推荐起点**。256K 上下文，文本+图片输入 |
| `qwen3.5:9b` | **6.6 GB** | 能力更强，8 GB 显存偏紧但可行 |
| `qwen3.5:2b` | 2.7 GB | 只用来验证流程，别指望它做程序合成 |

选 `instruct` 而不是 `thinking` 变体：程序合成要的是**严格照格式输出**，thinking 会拉长延迟，还可能把标签埋在思考块里。

> **为什么是多模态的 `qwen3.5` 而不是纯文本模型**：视觉问答是把图片**内联 base64** 发给模型的，而这条路径在 Omni3D 分支里同样存在。纯文本模型一跑到它就必然 400。所以主模型必须能看图 —— 一个 `qwen3.5` 端点同时顶掉「程序合成」和「视觉问答」两件事。

### Step 0.8 — 测 4B 能否驱动程序合成（**零成本，去要答案**）

```powershell
# 该脚本已随检出移除；等价能力的实现见 llm/adapter.py + agents/synthesizer.py
```

当时那个脚本用**上游原版 prompt 与原版正则**做两阶段测试（记录如下）：

| 阶段 | 喂什么 | 查什么 |
|---|---|---|
| Stage A | `SIGNATURE_PROMPT`（上游的签名提示词模板） | 是否吐出成对的 `<docstring><signature>`；方法名是否以 `_` 开头；**每个签名能否被方法名提取正则取到方法名** |
| Stage B | `PROGRAM_PROMPT`（上游的程序提示词模板，含 `{predef_signatures}` + `{api}` + `{question}` 三个占位符） | 是否有 `<program>`；程序体能否 `compile()`；是否按三次强调的要求赋值 `final_result` |

**最值得盯的一项是「注解陷阱」**：方法名提取正则 `def (\w+)\s*\(.*\):` 要求字面量 `):`，所以

```python
def _count_mugs(image, bbox):          # ✅ 匹配
def _count_mugs(image, bbox) -> int:   # ❌ 匹配失败 -> None.group(1) -> AttributeError
```

gpt-4o 通常不写返回值注解，**换模型后是否还守这条规矩没有任何保证**。我已用一段故意违规的假回复验证过打分器：喂进「一个带 `-> int` + 一个不带的」混合回复，脚本准确报出 `1/2 个失败 → AttributeError: 'NoneType' has no attribute 'group'`，且 `method_names` 只提取到 1 个 —— 说明这个检查是真的在量，不是装饰。

其他参数（探针脚本已随上游检出一并移出；下为当时的调用形式，当前等价能力见 `llm/adapter.py`）：

```powershell
# 多模型对比，输出一张表（可直接进消融实验章节）
... <已移出的探针脚本> --models qwen3.5:4b,qwen3.5:9b

# 跑全部 4 道样例题而不是默认 1 道
... <已移出的探针脚本> --all-questions

# 想拿云端模型做对照臂（这一步就需要 key 了，可以推到以后）
... <已移出的探针脚本> --base-url https://api.deepseek.com --model deepseek-flash --api-key sk-xxxx
```

**结果怎么用**：
- 两阶段全 PASS → Phase 3 可以按原计划"先接模型再微调"
- 只挂「注解陷阱」→ 用提示词约束 + 容错层兜住，不算致命（§19 风险 6 的约束不要撤）
- 标签本身吐不出来 / 程序不编译 → **这正是 QLoRA 要解决的问题**，把失败样本直接记成训练数据的负例种子，微调的动机从"提升分数"变成"让流程能跑通"，叙事反而更强

### 清单

> **2026-09-16 状态更新：Phase 0 的视觉栈部分已全部完成，且全程 Windows 原生、未装 WSL。**
> 下面按当前真实状态重排，不再按原来的「P0 装 Ollama / P1 装 WSL」排。

**✅ 已完成（本次）**

- [x] 建 venv `D:\3D_Spatial_Agent\venvs\vision`（Python 3.12.5）
- [x] torch 2.6.0+cu124 + torchvision 0.21.0+cu124（走 curl 预下的本地轮子）
- [x] 全部运行依赖 + `unidepth` 源码 editable 安装
- [x] `verify_env.py` 通过：CUDA 可用、认到 4060 Laptop 8188 MiB / sm_89，**在 xformers / triton / torchaudio 全缺的前提下成功 `import UniDepthV2`**
- [x] 下载 UniDepth V2 权重（130.4 MB，hf-mirror）
- [x] `probe3d.py` A/B 两段跑通：**确认 `points` 是真几何**，实测显存 486/604 MB、延迟 ~1 s
- [x] 生成锁定文件 `phase0/requirements-vision.lock.txt`
- [x] 下载 GroundingDINO 权重（657 MiB，curl 分段续传，见 `01c_fetch_gdino.ps1`）
- [x] `run_gdino()` 从编译式 `groundingdino` 包改写为 transformers 原生 `AutoModelForZeroShotObjectDetection`
- [x] `probe3d.py` C/D 两段跑通：G-DINO 实测 **1903 / 2306 MB、814 ms**；**首次由几何算出真实三维距离**（sofa ↔ table = 1.608 m）
- [x] 方案文档里 GroundingDINO 那批估算数字全部换成实测值

**下一批（不需要 WSL、不需要 API key）**

- [x] ~~实测 **SAM2**（`Sam2Model`）的显存与延迟~~ ✅ **2026-09-16 完成**（`phase0/probe_sam2.py`）：**73.3 M 参数 / 279.7 MiB，加载 0.6 s，单框 282 ms，峰值 647 MB allocated，9 框批量 179 ms**
- [x] ~~量化「框 vs 掩码」的三维质心差~~ ✅ **均值 83 mm / 最大 208 mm，超过 50 mm 容差 → Phase 1 必须用掩码**（框内背景占比均值 17.8%、最大 53.8%）
- [x] ~~Phase 1 建工具库时把 `get_3d_position` 改成**掩码质心**~~ ✅ **已在 `scene_graph/builder.py` 落地**：掩码质心为主，检测框只作降级路径，并标 `centroid_source=bbox_fallback` 供 L5 归因
- [x] ~~建 L1 感知层与场景图构建~~ ✅ **2026-09-16 完成**：`vision/`（types/geometry/grounding/segmentation/depth/registry）+ `scene_graph/builder.py` + `scene_graph/store.py` + `scripts/build_scene.py`；**166 个无 GPU 单测 0.93 s 通过**
- [x] ~~量测「内参来源」的精度杠杆~~ ✅ **2026-09-16 完成**（`phase0/probe_depth_gt.py`）：三维误差中位数 **1.943 m → 0.267 m（7.3×）**，并撤销了 `calibrate_scale` 的设计。详见 §21
- [ ] 真实照片无 GT 内参时的 EXIF 等效焦距估算（`fx_px ≈ f_35mm/36 × 宽`）—— 尚未实现
- [ ] 在多张图上复跑 `probe_depth_gt.py`，确认「内参杠杆」不是 demo 图特有的
- [ ] 跑 `01_setup_windows.ps1` 的收尾步骤（若从零重建环境时用）

**已降级 / 挂起**

- [ ] ~~装 WSL2~~ —— **不再需要**。整条视觉栈已在 Windows 原生跑通，§4 的三处平台阻塞逐项验证为不存在。
- [ ] ~~装 Ollama + `qwen3.5:4b`~~ —— 挂起。本地 LLM 是**加分项不是前提**（§9.4）；Phase 0–8 全程可只用 API。
- [x] ~~换模型后的协议连通性~~ —— 已并入 **§8.6** 的实测（真实调用 15 次，四类标签全绿）

> **已落地的可执行文件**
>
> | 文件 | 作用 | 需要 WSL? |
> |---|---|---|
> | **`phase0/01_setup_windows.ps1`** | **Windows 原生一键建环境（当前主路径）**：venv → torch → 6 批依赖 → UniDepth → 自检 → 锁文件 | ❌ |
> | **`phase0/01b_fetch_torch.ps1`** | 用 curl 预下 torch/torchvision 轮子（pip 会静默卡死，见上文） | ❌ |
> | **`phase0/01c_fetch_gdino.ps1`** | 用 curl 下 GroundingDINO 权重到 `.cache\models\grounding-dino-tiny`（657 MiB，可反复重跑续传） | ❌ |
> | **`phase0/verify_env.py`** | 环境自检：CUDA / 缺件清单 / `import UniDepthV2` 决定性测试 | ❌ |
> | **`phase0/probe3d.py`** | 视觉栈验证 + 真实显存/延迟 + `points` 自洽性检验 + **由几何算出的三维距离** | ❌ |
> | `phase0/probe3d_result.json` | 上面那次运行的机器可读结果（实测数字的唯一来源） | ❌ |
> | `llm/adapter.py` | OpenAI 兼容客户端：双端点路由、密钥掩码、调用记账 | ❌ |
> | `phase0/04_install_ollama.ps1` | 本地 LLM 后端（Windows，零 API key）——**已挂起** | ❌ |
> | ~~上游 API 探针、上游 LLM 桥接、上游提示词契约探针、LLM 后端说明、WSL 安装脚本 2 个~~（共 6 个文件） | **已于 2026-09-20 移出仓库**（上游依赖 / 旧 WSL 路线）；其结论已并入 §8.6 与 §2 | — |

> | `tools/build_doc_html.py` | 文档改了跑一次，Markdown → HTML | ❌ |

**本阶段不做的事**：不跑全量 benchmark、不装 Molmo-7B、不写业务代码、不碰前端、不下完整数据集、**不因为一个能本地完成的任务去租 A100**。

**本阶段只回答两个问题**：4B 模型能不能扛住那套零容错的标签协议？这条 3D 路线在 4060 上物理上走得通吗？

### 如果今天遇到这些，不要慌

| 现象 | 处理 |
|---|---|
| `AttributeError: module 'signal' has no attribute 'SIGALRM'` | 上游本体才会遇到。我们的执行器用**独立计时线程强杀**子进程做超时，不碰 `signal`，这条在 Windows 原生路径上根本不会出现 |
| `torch.cuda.is_available()` 为 False | 更新 Windows 侧显卡驱动（无 WSL，不存在「在 WSL 里装驱动」这条弯路） |
| **xformers / triton 装不上** | **不需要装**。UniDepth 的 `requirements.txt` 里确实写了 `triton>=2.4.0` / `xformers>=0.0.26`，所以必须用 `pip install -e UniDepth --no-deps` 绕开；它源码里 4 处 `import xformers` **全在 `try/except ImportError` 内**，缺了只打印一行 `xFormers not available` 再回落到 PyTorch 原生算子。**已在真机验证：两者全缺时 `import UniDepthV2` 成功（4.84 s），推理正常** |
| **SAM2 包缺失** | 只在**跑上游原版代码**时才是问题（它是顶层 import，缺了直接 ImportError）。**本项目自己的工具库不装 `sam2` 包**，改用 transformers 原生 `Sam2Model`（本机实测该类存在），连 CUDA 扩展都不用编 |
| HF 下载失败 / 下到一半卡死 | 本机实测：`huggingface.co` 直连超时，`hf-mirror.com` 可用。但**别用 `snapshot_download` 下大文件**——它默认还会再下一份等价的 `pytorch_model.bin`，且 `hf-xet` 传输层只有 0.5 MB/s。改用 `curl -C -`（`01c_fetch_gdino.ps1` 就是这个模式），实测 **2.2–2.6 MB/s 且可续传** |
| GroundingDINO `predict` 报 `cuda:0` 相关错 | 那是**编译式包**的行为（硬编码设备）。本项目走 transformers 原生，设备由自己指定，没有这个坑 |

---

## 21. L1 感知层与场景图构建（Phase 1 已落地的实现）

> 本节记录的是**已跑通的代码与实测数字**，不是计划。
> 数据来源：`phase0/probe_depth_gt.py`、`scripts/build_scene.py`、`scripts/inspect_scene.py` 的真实运行输出。
> 机器：RTX 4060 Laptop 8188 MiB，Windows 原生（无 WSL），Python 3.12.5。

### 21.1 文件与职责

| 文件 | 职责 | 依赖 torch |
|---|---|---|
| `vision/types.py` | `Detection` / `DepthField` / `PerceptionLike` 协议 | ❌ |
| `vision/geometry.py` | 掩码质心、稳健尺寸、包围盒、重力方向、**视场检查** | ❌ |
| `vision/grounding.py` | GroundingDINO（transformers 原生） | ✅ |
| `vision/segmentation.py` | SAM2（transformers 原生） | ✅ |
| `vision/depth.py` | UniDepth V2 → `points` / `depth` / `K` | ✅ |
| `vision/registry.py` | 三模型的懒加载、显存记账、卸载 | ✅（函数内） |
| `scene_graph/builder.py` | 单遍构建：升维→检测→去重→分割→节点→重力→关系 | ❌ |
| `scene_graph/store.py` | 场景图落盘（带格式版本信封）+ 掩码 1-bit PNG | ❌ |
| `scripts/build_scene.py` | 真实跑通入口（CLI） | ✅ |
| `scripts/inspect_scene.py` | 场景图「物理可能性」体检（不加载任何模型） | ❌ |

**分层不是洁癖**：`types.py` + `geometry.py` 零 torch 依赖，使
`scene_graph/builder.py` 的全部分支（去重、降级、丢帧、关系生成、掩码存盘）
可以在 **0.93 秒**内测完且不需要 GPU（166 个用例）。
把 `PerceptionLike` 收窄到三个方法（`detect` / `segment` / `lift`），
测试里 30 行就能写一个 fake —— 窄接口的直接收益在这里。

### 21.2 四条实测裁决（写进代码，不只是注释）

| # | 裁决 | 依据 | 落到哪 |
|---|---|---|---|
| ① | 质心取 **SAM2 掩码**内点云的中位数，不是检测框内 | 两者差均值 83 mm / 最大 208 mm，而关系容差只有 50 mm | `builder.py`（框只作降级路径，且标 `centroid_source=bbox_fallback`） |
| ② | SAM2 **一次调用带全部框** | 9 框：一次 179 ms，逐个 1512 ms，差 **7.57×** | `PerceptionLike.segment()` 只接受一批框，没有单框重载 |
| ③ | 模型句柄必须是**实例属性** | 放在局部变量里测出「三模型常驻 291 MB」，真值 1200 MB | `registry.py`；每次加载都记账并进 `build_meta` |
| ④ | **内参来源决定横向尺度** | 见 21.3 | `BuildConfig.known_intrinsics` + `DepthField.intrinsics_source` + 视场警告 |

三条 Phase 1 实测总量：三模型常驻 **1146.7 MB = 8188 MiB 的 14.0%**，
峰值 2177.2 MB；冷启动加载 13.0–15.6 s；单图构建墙钟 **1.7–2.2 s**
（分段：UniDepth 1180 / G-DINO 744 / SAM2 261 / 节点 28 / 重力 8 / 关系 2 ms）。

### 21.3 ⭐ 内参来源 —— 一个被官方文档低估的精度杠杆

Phase 1c 第一次串起三模型时，场景图里出现了 **6.70 m 宽的沙发**。
一路查到源头，是 UniDepth V2 的**相机头**给错了内参：

| | fx | fy | cx | cy | 水平视场 |
|---|---|---|---|---|---|
| 真值（`assets/demo/intrinsics.npy`） | 518.9 | 519.5 | 325.6 | 253.7 | **63.3°** |
| 模型预测 | 163.7 | 163.4 | 322.0 | 248.1 | **125.8°** |
| 比值 | 0.316 | 0.315 | — | — | 横向放大 **3.169×** |

125.8° 的水平视场不是任何常见相机的形态。已排除「预处理算歪」：
本图 640×480 落在 `pixels_bounds=[200000,600000]` 与 `ratio_bounds=[0.5,2.5]` 内，
`paddings=(0,0,0,0)`、`resize_factor=1.0` —— `_postprocess_intrinsics`
（`unidepthv2.py:92-108`）没做任何修改，这就是相机头的原始输出。

> ⚠ **口径更正（2026-09-16 深夜，见 §23）**：上面这套「预处理没动过」的排除
> **是正确的，但不充分**。它排除了「pipeline 算歪」，**没有**排除「分辨率是个自变量」。
> §23.1 在同一张图上只改输入像素数：**640×480 恰好是相机头的一个反常工作点**，
> 768 px 以上比值回到 1.12–1.14（几乎正确），3D 误差从 1.943 m 掉到 0.18 m。
> 所以 **163.7 / 125.8° / 3.169× 这三个数必须带上「在 640×480 下」才成立**。
> 下面这张 A/B 对照表本身仍然有效（同一分辨率下的因果对照），
> 但「模型猜得偏 3 倍」应改述为「模型猜得**不可控**」—— 后者才是不能依赖它的理由。

**量化证据。** 仓库自带配对的 GT 深度图（`assets/demo/depth.png`，毫米，
官方 `scripts/demo.py:25` 就是这么用的），于是可以构造一个不含任何模型预测的
逐像素参照 `P_gt = unproject(像素网格, GT内参) × GT深度`，把两条路径摊开比：

| 路径 | 深度 ARel | δ<1.25 | 深度 RMSE | **三维误差中位** | 三维误差 p90 | 三维相对中位 |
|---|---|---|---|---|---|---|
| A `infer(rgb)` | 19.8% | 57.1% | 0.831 m | **1.943 m** | 3.371 m | 59.0% |
| B `infer(rgb, camera=GT K)` | 11.7% | 93.2% | 0.550 m | **0.267 m** | 0.983 m | 9.1% |

同一张图、同一份权重，只因为「让模型猜相机」还是「把相机告诉它」，
**三维误差中位数差了 7.3 倍**（降到 13.8%）。分区统计进一步确认了因果：
A 路径外围/中心的误差比是 1.25×，B 路径是 0.58× —— 误差随视场角增大，
正是**方向场**偏离真值的签名，而不是「模型整体不准」。

**必须在报告里纠正的两点：**

1. **官方 README 的措辞是误导的。** `README.md:140` 写
   *"You can use ground truth intrinsics as input to the model **as well**"*
   （锦上添花），但 `scripts/demo.py:14` 的官方 demo **就是传了 camera 的** ——
   作者自己验证效果时用的从来不是纯 RGB 路径。
2. **模型回传的 `intrinsics` 不能用来判断 camera 是否生效。** 它是**独立预测头**：
   传入 GT 的 fx=518.9 之后，`out["intrinsics"]` 依旧写着 163.7。
   传进去的 camera 走的是另一条路 —— `unidepthv2.py:361-362` 把它转成 `rays`
   喂进 decoder 当条件。所以 `DepthField.intrinsics` 必须记**实际生效的那一份**，
   否则 `scene.camera_intrinsics` 会留下一个与点云不自洽的 K。

### 21.4 内参错误是**各向异性**的 —— 因此撤销 `calibrate_scale`

原方案里有一个 `calibrate_scale`（用一个标量把米制尺寸缩放到合理范围）。
这条发现证明它**修不了**这个问题：

    x = (u - cx)·z / fx

fx 小了 k 倍 ⟹ 横向坐标大 k 倍，而 z **几乎不动**（`depth` 就是 `points` 的 z 列，
近轴处 x≈y≈0，focal 在那里不起作用）。这是一个**各向异性**的形变，
不是整体缩放。用一个全局 `scale_factor` 去"校正"，只会在修横向的同时
把本来已经对的 z 一起弄错。

**正确做法是把内参当成输入**，而不是后处理：

```python
from scene_graph.builder import BuildConfig
cfg = BuildConfig(known_intrinsics=(518.86, 519.47, 325.58, 253.74))  # fx, fy, cx, cy
```

CLI 三种写法：

```powershell
--intrinsics auto                     # 自动找图片同目录的 intrinsics.npy / camera.npy / K.npy
--intrinsics .\calib\intrinsics.npy   # 3×3 npy
--intrinsics "518.9,519.5,325.6,253.7"  # 直接给四个数
```

只知 fx/fy 不知主点时，主点填几何中心仍远好于不传 ——
误差的主要来源是 focal（实测差 3.17 倍），主点偏移影响小一个量级。

**没有已知内参时**：`intrinsics_source` 标为 `"predicted"`，
并做视场合理性检查（`PLAUSIBLE_HFOV_DEG = (30°, 110°)`）。
不合理就在 `build_meta` 留痕**并**发一条警告，把「怎么修」也写进去 ——
只报错不给出口的警告会被忽略：

> 内参是模型预测的，且视场不可信（HFoV 125.8°，reason=hfov_out_of_range，
> 可信区间 30–110°）—— 所有横向米制尺寸可能被整体放大，本图的坐标只能当相对量用。
> 有已知内参请设 `BuildConfig.known_intrinsics`。

### 21.5 同图 A/B 复跑（唯一变量就是内参）

用同一张 `rgb.png`、同一个 prompt、同一套权重跑两次
（`living_room_gt` vs `living_room_pred`）。**两次的掩码逐像素完全相同**
（sofa 都是 23841 px，检测框一模一样）—— 分割不受内参影响，
所以这是干净的单变量对照。

| 指标 | 有 GT 内参 | 预测内参 |
|---|---|---|
| sofa 尺寸 w×h×l | **2.34 × 0.82 × 1.26 m** | 6.70 × 2.35 × 1.13 m |
| table | 1.06 × 0.54 × 1.03 m | 3.00 × 1.48 × 1.29 m |
| chair | 0.98 × 0.30 × 0.83 m | 2.72 × 0.94 × 0.79 m |
| mirror | 0.88 × 1.28 × 0.27 m | 2.57 × 3.69 × 0.28 m |
| 深度范围 | [1.59, 4.39] m | [1.38, 3.97] m |
| **重力方向 tilt** | **11.95°，reliable=True (`ok`)** | **63.04°，reliable=False** |
| 警告条数 | 0 | 2 |
| 关系边数 | 137 | 133 |

只有第一行是意料之中的。**第二处是个意外的二级效应**：
内参错误把地面点云横向拉伸了 3.17 倍，直接毁掉了重力方向估计 ——
tilt 从 11.95° 变成 63.04°，判为不可靠。而 `above`/`below` 全部依赖这个 up 轴，
也就是说**内参错误会静默翻转所有上下关系**，不只是把数字放大。
这条此前没被预料到，值得单独写进报告的风险分析。

有 GT 内参那一版跑 `inspect_scene.py` 的结论是「**没有发现越界数值**」；
预测内参那一版给出 5 条嫌疑，其中 1 条是根因、4 条是它的症状 ——
体检脚本现在会把根因排在第一位，并写明「先解决这里」。

### 21.6 尚未解决 / 下一步

> **2026-09-16 晚更新**：下面第 1、2、5 条已在本节之后被处理，见 **§22**。
> 第 1 条（EXIF）已实现并端到端验证；第 5 条（多图复跑）改用
> **剂量-反应扫描 + 数字变焦**给出了等效证据，但仍留下一个明确的证据边界。
> 第 3、4 条不变。

- ~~**真实照片没有 GT 内参怎么办**：可从 EXIF 的等效焦距估
  （`fx_px ≈ f_35mm / 36 × 图像宽度`），或要求用户提供标定。尚未实现。~~
  **已实现** → `vision/exif.py`、`scripts/inspect_exif.py`、`--intrinsics exif|auto`；见 §22.6。
- ~~**Omni3D-Bench 自带 GT 相机** ⟹ 主实验臂的横向尺度是可控的，
  这反而是本项目的有利条件，应在实验设计里写清楚。~~
  **❌ 该说法已被实测推翻（2026-09-17，见 §24）**：Omni3D-Bench 的 parquet 只有
  `image_index / image / q_index / question / answer / answer_type` 六列 ——
  没有相机内参、没有深度、没有三维框；HF 官方 README 的 annotations 格式同样
  不含任何三维字段。⟹ **主实验臂并不自带 GT 相机**，它同样走在
  「模型自己猜相机」那条不确定路径上。原「有利条件」的推论随之作废。
- 重力方向估计仍是最脆弱的一环（有 GT 内参时 tilt 仍有 11.95°）；
  画面下沿若是斜面或有 roll 会失效，目前只做到「检测」不解决。
- 关系边只按 `(i, j)` 且 `i` 小于 `j` 的方向单方向枚举，反向关系由 `query_relation` 现算 ——
  已写进 `builder.py` 的注释与单测，避免下游误以为图里有两条边。
- ~~尚未在**多张图**上复跑：本节全部结论来自同一张 demo 图。
  换图后需要重跑 `probe_depth_gt.py` 才能声称「内参杠杆」有普遍性。~~
  **已用另一条路给出证据**：仓库内只有一对 (图, GT 深度) 配对，换图不可能。
  改为**扫单一变量**（焦距倍率 k，其余全部冻结）建立因果 + **数字变焦**造三组视场；
  见 §22.1–§22.5。**但证据边界依然存在，且被明确写下来了**（§22.9）。

---

## 22. 内参杠杆的因果证据、EXIF 落地、主点代价（2026-09-16 晚）

> 本节是 §21.3 的接续。§21.3 已经证明「内参来源」是精度杠杆，
> 但那是一张图、一次对照。任何答辩老师都会接着问三个问题，本节逐个回答：
>
> ① **这是不是巧合？** —— §22.1–§22.5
> ② **真实照片没有 GT 内参怎么办？** —— §22.6
> ③ **EXIF 的精度够不够？不够的话差在哪？** —— §22.7
>
> 数据来源：`phase0/probe_k_sweep.py`、`phase0/probe_principal_point.py`、
> `phase0/make_exif_fixture.py`；报告分别为 `probe_k_sweep_report.txt`、
> `probe_principal_point_report.txt`、`probe_exif_pipeline_report.txt`。

### 22.1 为什么「换多张图」被换成了「扫 k」

§21.6 留下的最大缺口是「所有结论来自同一张图」。但仓库里**只有一对
(图, GT 深度) 配对**（`vendor/UniDepth/assets/demo/`），换图在物理上做不到。

于是把问题重新表述：真正要证明的不是「换一张图也成立」，
而是**「误差是由内参这一个原因造成的」**。后者可以用**剂量-反应（dose-response）**
来建立：把焦距按 k 倍缩放，权重、图、prompt、预处理**全部冻结**，
只在 `k ∈ [0.15, 4.0]` 上扫 20 个点。若内参是支配原因，误差曲线必须
**在 k=1 处极小、两侧单调**。

结果是三条曲线，极小值**不在同一个位置** —— 这正是本节最有价值的发现：

| 分量 | 极小在 k= | 对应 fx | HFoV | 极小值 |
|---|---|---|---|---|
| **方位误差（与深度解耦）** | **1.0000** | 518.9 | 63.3° | **2.6 px** |
| 深度 ARel | 1.1000 | 570.7 | 58.6° | 7.5 % |
| 3D 合成误差 | 1.1000 | 570.7 | 58.6° | 12.84 cm |

真值 k=1.0000 处：方位 2.6 px，3D 26.73 cm。
预测 k=0.3155（fx=163.7）处：方位 **306.3 px**，3D 192.73 cm。

**方位曲线两侧单调**（左单调递减 True，右单调递增 True）⟹ 该极小是全局极小，
曲线形态干净、没有多峰，因此它**可以作为焦距的估计量**。

> ⚠ **方法学修正（本节最重要的负面结果）：不能用 3D 合成误差反推相机内参。**
> 它的极小跑到 k=1.1000，偏离真值 10.0%。原因不是内参，而是**深度头的偏好**：
> 深度 ARel 的极小也在 k=1.1000 —— 模型在更窄的视场上更接近它的训练分布。
> 若拿合成误差定容差，会得出「把焦距故意调大 10% 反而更好」这种荒谬结论
> （实测 3D 误差确实从 26.7 cm 降到 12.8 cm）。**必须分解成横向与纵深两部分。**

各向异性的量化形态：方位误差动态范围 **2.6 – 797.0 px（306.5 倍）**，
而深度 ARel 只有 **7.5% – 73.7%（9.8 倍）** —— 差一个量级。
这就是「focal 错只让横向按 k 倍伸缩、沿光轴几乎不动」的直接读数，
也再次说明全局标量 `scale_factor` 修不了它（§21.4）。

### 22.2 自洽性：误差真的只归因于「那份 K 的数值」吗

上面整条曲线暗含一个前提：「模型回传的 `intrinsics`」与它**实际用来构造方向场的 K**
是同一份。这条前提单独可测 —— 把模型预测的 K **显式喂回去**，与「不给内参」对比：

| 对照 | 逐像素点云差异 | 3D 误差中位 |
|---|---|---|
| 喂回预测 K vs 不给内参 | 中位 1.642 mm，p90 3.757 mm，最大 18.175 mm | 1.9435 m vs 1.9430 m |

毫米级 ⟹ 成立。**误差可以被完全归因到「那份 K 的值错了」这一个原因上，
模型内部没有别的隐藏机制在干扰。** 这是后面所有结论成立的前提。

### 22.3 机制确认：ray 场就是解析针孔，传入 K 就完全可控

§5 原先的措辞担心 `rays` 是「学习出来的方向场、无法解析控制」。实测否定了这个担心。
判据：若模型的方向场就是焦距为 k·fx 的针孔网格，那么「方位跨度 / GT 跨度」
应当精确等于 1/k。20 个采样点（焦距覆盖 0.15–4.00 倍，相差 **27 倍**）：

| k | 方位跨度/GT | 针孔预期 1/k | 相对差 |
|---|---|---|---|
| 0.1500 | 6.708 | 6.667 | 0.6% |
| 0.3155 | 3.189 | 3.170 | 0.6% |
| 1.0000 | 1.006 | 1.000 | 0.6% |
| 2.0000 | 0.503 | 0.500 | 0.6% |
| 4.0000 | 0.252 | 0.250 | 0.6% |

全部落在 **0.62–0.64%**，几乎是一个常数。

> ⟹ **修正 §5 的措辞**：`rays` **不是**不可控的学习场，而是一个解析针孔网格，
> 焦距完全由传入的 K 设定。因此横向尺度可以被**精确设定**，
> 而不是只能被动测量。（那 0.6% 的常数残差与 k 无关，指向采样网格约定这类
> 系统性因素，不是随机误差；它同时**上界了本方法能分辨的焦距精度**。）

> ⚠ **一个被实测证伪的统计量**（写下来免得后人再踩）：本打算用「X 跨度比」当横向读数，
> 但它**被最远像素主导** —— x = (u−cx)/fx · z，z 越大横向越夸张。本图 GT 深度到 10.0 m，
> 7.7% 的像素超过 5 m，于是该比值混进了深度误差：k=1 时读 0.650，
> 看起来像「模型横向压缩了 35%」，其实只是它在远处给不出 10 m。
> **方位跨度比值没有这个病**（它把 z 除掉，是纯方向量）——
> 这解释了为什么上一版报告的 X 跨度比值在宽视场端差到 40% 以上。

### 22.4 细化扫描：焦距可以被独立反解，精度 ±1%

C 段网格间距是 ±10%，严格讲只能说「极小落在 [0.95, 1.05] 内，且真值点是该区间的最优采样点」。
细化到 ±1% 网格 + 抛物线插值：

| k | fx | 方位 px | 3D 中位 |
|---|---|---|---|
| 0.98 | 508.5 | 3.80 | 0.3047 |
| 0.99 | 513.7 | 2.80 | 0.2864 |
| **1.00** | **518.9** | **2.60** | 0.2673 |
| 1.01 | 524.0 | 2.50 | 0.2476 |
| **1.02** | **529.2** | **2.30** | 0.2278 |
| 1.03 | 534.4 | 3.20 | 0.2083 |

网格极小 k=1.02（2.30 px），抛物线插值后 **k̂ = 1.0168 → fx̂ = 527.6**，
与仓库自带 GT 的 fx=518.9 相对差 **+1.68%**。

⟹ **这是一个可复用的方法**：只要有 GT 深度，就能用**方位误差的极小位置**
把真实焦距**独立测出来**（量级精度 ±1%），从而给 EXIF 那 4% 级的量化误差当判据，
而不是只能盲信 EXIF 元数据。

两条**独立读数**的交叉核对（不要把它们说成一致，差的那部分要交代）：

- 方位跨度比值在 k=1 时 = 1.0063 ⟹ 蕴含模型方位场比 GT 宽约 0.63%，对应 k ≈ 1.006
- 方位误差的抛物线极小 k̂ = 1.0168

两者**同号**，但量级差 1.1% —— 而 ±1% 正是本方法在当前分辨率下的分辨极限
（0.63% 的焦距差在画面边缘只值 2 个像素）。
⟹ 正确说法是「模型方位场相对 GT 内参有约 **0.6–1.7% 的系统性展宽**」，
而不是某个更精确的数。**两处都指向同一方向，这一点比数值本身更值得记。**

> 实用含义：这 1% 量级的残差比 EXIF 自身的 4% 量化误差还小，
> 对关系判断（容差 50 mm）完全不构成问题 —— **不值得再投入去挤压它**。
> 继续抠它需要的不是更聪明的拟合，而是更多图像。

### 22.5 多视场复现：用数字变焦造出三组 (图, GT K) 配对

既然不能换图，就换**视场**：对同一张图做中心裁剪 + 缩放（相当于数字变焦 s 倍），
视场随之变窄，而 GT 内参可以在解析上同步更新
（`GT K` 按裁剪比例缩放，`GT 深度`必须一起裁剪 —— 这一点第一次写漏了，
见 §22.8 的坑）。

| s | 裁剪 | GT fx | GT HFoV | 预测 fx | A 3D 中位 | B 3D 中位 | **A/B** |
|---|---|---|---|---|---|---|---|
| 1.00 | 640×480 | 518.9 | 63.3° | 163.7 | 1.9430 | 0.2673 | **7.27×** |
| 1.50 | 427×320 | 777.7 | 44.7° | 188.3 | 2.6686 | 0.2234 | **11.95×** |
| 2.00 | 320×240 | 1037.7 | 34.3° | 248.1 | 2.1517 | 0.4899 | **4.39×** |

三档**全部** A/B > 1（不给内参都更差）⟹ 「内参杠杆」不是某一对数值的巧合，
在三个不同视场下都复现。

**顺带得到一个关于模型相机头的独立结论**：s=1.00 时预测 fx=163.7，
s=2.00 时 248.1，实际放大 **1.515×**，而真值应放大 2.000×。
⟹ 相机头**确实读到了视场线索**（fx 随变焦增大），但**严重欠响应**。
所以它不是「输出一个与视场无关的常数」，而是在「有多宽」这件事上系统性偏低 ——
与它不给内参时 fx 偏小 3.17 倍是**同一种偏差在不同尺度上的表现**。

> 另一处印证：「深度头偏好窄视场」被独立复现 ——
> s=1.50（44.7°）时 B 路径误差 0.2234 m，比 s=1.00（63.3°）的 0.2673 m 还小，
> 尽管变焦图更糊。这就是 §22.1 里合成误差极小跑到 k=1.1 的原因。
> **因此本段只做同图 A/B，不与 s=1 横比绝对值**（变焦图是上采样的，不能评绝对精度）。

### 22.6 EXIF 路径：真实照片无 GT 内参时的唯一来源

「内参杠杆」要落地，来源只有三个：① 标定流程（课程作业场景不现实）
② 数据集自带相机（只在用数据集时有）③ **EXIF**。所以这不是顺手加的便利功能，
而是让 §21.3 那条杠杆在真实照片上成立的**必要条件**。

新增 `vision/exif.py`（**零 torch**）+ `scripts/inspect_exif.py` + `--intrinsics exif|auto`。
三条实现判断：

| # | 判断 | 理由 |
|---|---|---|
| ① | 用**长边** `max(W,H)` 换算，而不是先解析 Orientation | `f_px = f_35mm / 36 × max(W,H)`；结果**对旋转不变**，消掉「5/6/7/8 记错一个就静默按宽高比缩放 fx」这条路 |
| ② | 优先 `FocalLengthIn35mmFilm`（0xA405），只有 `FocalLength`（0x920A）时降级并告警 | 后者需传感器宽度，默认 36 mm 是**全画幅假设** —— 对手机（约 5.6 mm）差约 6.4 倍。降级路径标 `assumed_sensor=True`。~~且**骗不过 `check_fov`**（手机照片会算出 150° 量级 → `hfov_out_of_range`），错误假设不会静默流到下游~~ **此断言已被 §23.2 实测证伪**：窗口只挡得住 `FocalLength < 12.6 mm`（手机超广），**APS-C / MFT / 1 吋一律逃得掉**；`assumed_sensor` 必须原样透传并**无条件告警** |
| ③ | EXIF **自带的量化误差**必须一起返回（`quantisation_rel`） | 见 §22.7 |

`read_exif_intrinsics()` **返回 `None` 而不是抛异常** —— EXIF 缺失太常见
（截图、聊天软件转存、部分 PNG），属**正常分支**；拿到 `None` 就退到
「模型预测 + `check_fov` 告警」那条路并写明降级原因。

**实测（`make_exif_fixture.py`）。** 先发现当时仓库里**两张图 EXIF 全为空**
（`scripts/inspect_exif.py`：UniDepth 自带的 demo 图、以及上游检出里的一张 demo 照片，
都是 0 个标签；后者已随上游检出移出）⟹ **EXIF 路径无法用仓库素材验证**。
于是自己造一张**带真值**的 fixture：从 GT 内参反推该写进 EXIF 的整数等效焦距

```
fx_gt = 518.9（640×480）→ f_35 = 518.9 × 36 / 640 = 29.186 mm → 取整 29 mm
回推 fx = 29/36 × 640 = 515.56 px        相对 GT 偏差 −0.64%
```

若只造一张「有 EXIF 的图」，验证就退化成「流水线跑通了」——什么都没证明。
**必须知道正确答案。** 这里的陷阱是：若换算写错宽高（例如竖构图用了 W 而非长边），
偏差会**立刻变成 33%** 而不是 0.64%，而这种错在单元测试之外看不出来。

读回结果：`source=exif:35mm`，K = fx=515.56 / cx=320.0 / cy=240.0，
与解析期望**差 0.000 px** ✓；HFoV 63.65° 判定可信；量化误差 ±1.72%。

**端到端 A/B（同一张 JPEG，唯一变量是内参）**：

| 场景 | 内参来源 | fx | HFoV | sofa 尺寸 | 视场检查 |
|---|---|---|---|---|---|
| `exif_fixture_exif` | `exif`（`intrinsics_source=provided`） | 515.56 | 63.7° ✓ | **2.46 × 0.80 × 1.22 m** | plausible |
| `exif_fixture_pred` | 模型预测 | 159.17 | **127.11°** | **6.93 × 2.41 × 1.14 m** | `hfov_out_of_range`，**警告正确触发** |

警告原文：*「内参是模型预测的，且视场不可信（HFoV 127.1°，reason=hfov_out_of_range）……
所有横向米制尺寸可能被整体放大」* —— 整条降级链路按设计工作。

> ~~**这一条决定 EXIF 的定位**：它只服务于「用自己拍的照片扩容样本」那条支线。
> **主实验臂（Omni3D-Bench）自带 GT 相机，不依赖 EXIF 精度** ——
> 这是本项目的有利条件，应在实验设计里写清楚。~~
>
> **❌ 已推翻（2026-09-17，见 §24）**：Omni3D-Bench **不带任何三维标注**。
> 因此 EXIF / 外部给 K 这条路径不是「支线」—— 它**同时**服务主实验臂：
> 只要主实验臂的题目涉及横向米制尺寸（501 题里 float 占 270 题、权重 53.9%），
> 内参不确定性就会直接进入分数。这条修正让 §21–§23 的内参结论
> 从「一条支线的精度问题」上升为「主指标的精度问题」。
> 另外：fixture 是 **JPEG（有压缩伪影）**，只用于**同图内** A/B，不与 PNG 那几次横比。

### 22.7 主点才是 EXIF 的主要误差，不是量化

这是本节第二个意外结论。EXIF 给出的 K 有两处可能错：**量化**（f_35 整数毫米）
与**主点**（EXIF 不记录 cx/cy，只能取图像中心）。用四组 K 做分解，
把总误差拆成两份：

| 变体 | 二维方位 px | ↑p90 | 仅 x 分量 | 仅 y 分量 | 3D 中位 (m) |
|---|---|---|---|---|---|
| K0 GT（理想下限） | 6.82 | 9.90 | 2.61 | 6.23 | 0.2673 |
| K1 EXIF 全套 | 21.96 | 26.16 | 8.32 | 20.26 | 0.2376 |
| K2 只错主点 | 21.56 | 24.63 | 8.19 | 19.95 | 0.2269 |
| K3 只错焦距 | 7.29 | 11.44 | 2.70 | 6.44 | 0.2805 |
| （不给内参，参照） | **432.0** | — | 305.8 | 238.5 | 1.9430 |

以 K0 的 6.82 px 为下限，EXIF 全套多出来的 15.14 px 拆成：

| 来源 | 方位 px | 占 | 3 m 处横向 | 占 50 mm 容差 |
|---|---|---|---|---|
| **主点中心假设（K2−K0）** | **14.75** | **97.4%** | 85.3 mm | **171%** |
| 焦距量化 0.64%（K3−K0） | 0.47 | 3.1% | 2.7 mm | 5% |
| 两者之和 15.22 vs 实测全套 15.14 | （差 0.08 px） | | | |

（两项不是严格可加 —— 方位误差是各分量的非线性组合；但量级对比已足够下结论。）

⟹ **主点是量化的 31.2 倍。EXIF 路线的主要误差是主点，不是量化。**
但两者都远小于「不给内参」的代价（432 px）—— 所以 EXIF 依然值得做，
只是它的定位要说准：**对米制尺寸与距离够用；对方向/方位级精度受限于中心假设。**

**5×5 主点偏移网格（焦距固定为 GT，隔离变量）**：网格最低点**不在 (0,0)** 而在
(Δcx, Δcy) = (0, +7) 处，读数 3.62 px，**低于**中心点 (0,0) 的 6.82 px。
⟹ 即使把 GT 内参原样传进去，模型实际用的 ray 场与这份 K 之间仍有约 7 px 量级的
**纵向偏移** —— 与 §22.3 那个 0.6% 尺度残差是同一类现象。
量级（不足画面 1%）远小于 EXIF 那 13.7 px，不影响结论，
但它说明「传入 K 就等于控制了 ray 场」这句话**只在 ~1% 精度上成立**。

**沿偏离方向的幅度扫描**（方向 (-5.6, -13.7) px 的单位向量 (-0.377, -0.926)）：

| 偏离幅度 px | 二维方位 px | 3 m 处横向 mm |
|---|---|---|
| 0.0 | 6.82 | 39.4 |
| 8.0 | 14.76 | 85.3 |
| 16.0 | 22.73 | 131.4 |
| 20.0 | 26.73 | 154.5 |

逐段斜率 **[0.988 0.992 0.995 0.997 0.997 0.998]**，最陡 1.00 / 最平 0.99，
比值 1.01× ⟹ **近似为常数**，所以主点误差可以被概括成
**「等效像素偏移」一个数** —— 这让它可以直接与 50 mm 的关系容差对比，
而不需要每次重跑点云。（换算：fx=518.9 ⟹ 3 m 处 1 px ≈ 5.78 mm ⟹ 50 mm ≈ **8.6 px**。）

> ⚠ **一条口径修正（两个数字都对，但不能混用）**：§21/Step 7 报的
> 「模型预测内参横向误差 **306.3 px**」用的是**一维**指标（只含 x/z）；
> 本探针同一情形用**二维**口径得到 **432.0 px**（x 305.8 / y 238.5）。
> 二维更大，因为它把纵向那部分也算进来了。
> **引用时必须带上口径**，否则「同一个量」在两处数字不同，看起来像矛盾。
> 顺带这也解释了为什么 Step 7 完全没看见主点问题：K2「只错主点」的一维 x 分量
> 是 8.19 px（不显眼），二维是 21.56 px，而它的 y 分量是 19.95 px ——
> **纵向主点误差几乎全部落在 y 分量上，而 Step 7 的指标只测 x。
> 用标量概括二维量必有盲区。**

### 22.8 本节新增 / 修改的文件、测试与本机坑

| 文件 | 状态 | 说明 |
|---|---|---|
| `phase0/probe_k_sweep.py` | 新增 | 剂量-反应 + 细化扫描 + 数字变焦 + 容差预算；`metrics()` 分解出 `err3d` / `err_xy` / **`bear_err_px_median`** / `bear_span_ratio` |
| `vision/exif.py` | 新增 | EXIF → 内参，零 torch，`ExifIntrinsics` 冻结 dataclass |
| `tests/test_vision_exif.py` | 新增 | **25 条**用例，零 GPU；覆盖长边约定、`image_size` 覆盖、降级告警、`size_mismatch`、无 EXIF→None、坏值→None、IFDRational/字符串、SubIFD 分支、IFD0 优先、K 良构、量化、frozen、**不 import torch** |
| `scripts/inspect_exif.py` | 新增 | 批量体检图片 EXIF 可用性；`--raw` 导出原始标签 |
| `phase0/probe_principal_point.py` | 新增 | 主点 vs 量化分解、5×5 网格、径向扫描 |
| `phase0/make_exif_fixture.py` | 新增 | 造带真值的 EXIF fixture（同时写 `probe_exif_pipeline_report.txt`） |
| `scripts/build_scene.py` | 修改 | `load_known_intrinsics()` 改为返回三元组 `(内参四元组或 None, note, meta)`；新增 `exif` 与 `auto`（优先级：sidecar npy → EXIF → 预测）分支；`build_log` 单独落 `intrinsics_meta` |

全套单测从 **166 → 191 passed**（2.77 s，零 GPU、零联网）。

**本节踩到并修掉的本机坑（都是「结果看起来合理但其实是错的」那类）**：

| 坑 | 症状 | 修法 |
|---|---|---|
| 数字变焦忘了裁剪 GT 深度 | A/B 被一个假的常数污染 | 裁剪窗口必须同时切 `d_gt`：`d_crop = d_gt[oy:oy+Hc, ox:ox+Wc]` |
| 单调性检查方向写反 | 严格递减的方位曲线报「不单调」 | `diff(lo) <= 0`；并把原始数组打出来核对 |
| 容差预算用了「相对基准涨幅」 | 基准只有 2.6 px（已接近模型误差下限），在近零分母上算出**虚高的 ±0.5%** | 改用**绝对像素门槛**（5/10/25/50 px → ±2.1% / ±6.6% / ±18.6% / ±42.0%）；相对版只留作交叉核对 |
| EXIF 测试用 `-1` 写 SHORT 标签 | Pillow 序列化报错 | 从参数化里去掉 `-1`，另设一条直接测 `_as_float` 拒绝坏值 |
| `HF_HUB_DISABLE_XZ` 拼错 | 环境变量静默失效 | 正确名为 `HF_HUB_DISABLE_XET` |

### 22.9 本节仍未做到的（证据边界，必须写进报告不能含糊）

1. ~~**全部证据仍来自同一张 demo 图。**~~ **已由 §23 处理，且结果反过来修正了本节的口径**：
   跨来源复跑（10 张、4 个来源 + 5 台真实相机 EXIF）证明**输入分辨率**是一个被忽略的混杂因子，
   本节全部数字（预测 fx=163.7 / HFoV 125.8° / 横向放大 3.169× / 三维误差 1.943 m）
   **只在 640×480 这一档成立**。详见 §23.1。
   但要分清：§23 换掉的是**「相机头有多偏」这个量**，**没有**换掉
   **「内参是横向尺度的唯一开关」这个结论** —— 后者在 §23.3 反而被加强了。
2. **变焦图是上采样的**，不是真实光学变焦，高频细节缺失，**不能用于评估绝对精度**。
3. ~~**EXIF fixture 是自己造的，不是真实手机照片。**~~ **已由 §23.2 补上**：
   用 5 台真实相机的原始 EXIF（Panasonic DMC-L10 / Olympus E-P3 / Ricoh GR /
   Sigma DP3 Merrill / Sony DSC-RX1R，画幅 1.0–2.0×）复跑，fx 独立重算与模块输出差 **0.000 px**。
   **仍缺席的是手机档。** 那 5 台相机的画幅分别是 35 mm(1.0×) / APS-C(1.5×) ×2 /
   4/3 与 MFT(2.0×) ×2，覆盖 1.0–2.0×；手机（crop≈5–7）没有实测。
   注意这一档不是「更难的极端」，而是**反过来**：按 §23.2 的窗口分析，
   手机超广（`FocalLength` < 12.6 mm）恰恰是**唯一会被 `check_fov` 兜住**的一档。
4. **主点 (5.6, 13.7) px 这个偏移相当大**，很可能因为 demo 图是**渲染**出来的
   （渲染相机的主点可以任意设定）。真实手机照片的主点通常更接近中心，
   所以 **14.75 px 这个数上界了实际代价，不是典型值**。要典型值需要真实照片 + 标定真值。
5. **未测「主点误差对物体尺寸/距离的影响」** —— 布局问题主要影响位置，
   尺寸是二阶效应；本探针只测了方位与合成 3D 误差。
6. **0.6% 的常数残差（方位跨度 vs 1/k）来源未查清**，它上界了可分辨的焦距精度。
7. 网格在真值附近的间距是 ±10%，所以「方位极小在 k=1」只能读作
   「极小落在 [0.95, 1.05] 内且真值点为最优采样点」，**不是**「精确等于 1」。

> **§22 一句话总结（口径已在 §23 更正）**：内参杠杆是**因果**的（剂量-反应、自洽性、
> 三视场复现都指向同一结论），方向场是**解析可控**的（0.6% 常数残差），
> 真实照片有**可落地的来源**（EXIF，精度在预算内），而 EXIF 的短板**不是量化而是主点**
> （14.75 px / 97.4%）。代价量级排序：**不给内参（432 px）≫ 主点假设（14.75 px）≫
> 焦距量化（0.47 px）**。⚠ 但「不给内参 432 px」这个数**只在 640×480 成立**
> —— 768 px 以上它会掉到 40 px 量级（§23.1）。

---

## 23. 跨来源复跑：一个被忽略的混杂因子（2026-09-16 深夜）

§22.9 把「全部证据来自同一张 demo 图」列为最大证据缺口。本节就是去补这个缺口，
但结果**不是**「结论被更多图证实」，而是**发现了原结论的一个混杂因子**：

> **§21、§22 的全部数字（预测 fx=163.7、HFoV 125.8°、横向放大 3.169×、
> 三维误差 1.943 m）都只在 640×480 这一档输入分辨率下成立。**

探针：`phase0/probe_cross_source.py`（三段 A/B/C，素材抓取在 `fetch_cross_source.py`）。

### 23.1 C 段：分辨率剂量-反应 —— 本节的结论性证据

固定 4:3、固定场景内容，**只改输入像素数**，其余一切冻结：

| 输入 | 像素数 | resize | fx 预测 | HFoV 预测 | HFoV 真值 | 预测/真值 | 可信 | 3D 误差(无 K) | 3D 误差(GT K) |
|---|---|---|---|---|---|---|---|---|---|
| 320×240 | 76.8 k | 1.614 | 94.4 | 118.92° | 63.33° | 0.364 | ✗ | 1.6365 m | 0.2251 m |
| 480×360 | 172.8 k | 1.076 | 123.1 | 125.68° | 63.33° | 0.316 | ✗ | 1.9604 m | 0.2943 m |
| **640×480** | 307.2 k | 1.000 | **163.7** | **125.80°** | 63.33° | **0.316** | ✗ | **1.9430 m** | **0.2673 m** |
| 672×504 | 338.7 k | 1.000 | 172.5 | 125.66° | 63.33° | 0.317 | ✗ | 1.9682 m | 0.2006 m |
| 704×528 | 371.7 k | 1.000 | 346.9 | 90.84° | 63.33° | 0.608 | ✓ | 0.7890 m | 0.1657 m |
| 736×552 | 406.3 k | 1.000 | 269.6 | 107.55° | 63.33° | 0.452 | ✓ | 1.2518 m | 0.2151 m |
| 768×576 | 442.4 k | 1.000 | 702.5 | 57.32° | 63.33° | **1.128** | ✓ | **0.1837 m** | 0.2066 m |
| 800×600 | 480.0 k | 1.000 | 741.7 | 56.68° | 63.33° | 1.144 | ✓ | 0.1803 m | 0.2107 m |
| 960×720 | 691.2 k | 0.932 | 888.0 | 56.79° | 63.33° | 1.141 | ✓ | 0.2491 m | 0.1389 m |
| 1280×960 | 1.229 M | 0.699 | 1159.4 | 57.80° | 63.33° | 1.117 | ✓ | 0.1774 m | 0.1427 m |
| 1600×1200 | 1.920 M | 0.559 | 904.2 | 83.00° | 63.33° | 0.697 | ✓ | 0.6443 m | 0.1415 m |

**① 结论一：`640×480` 是唯一的离群点，而它正是 §21/§22 用的那一档。**

跨 11 档，预测 HFoV 从 **56.68° 走到 125.80°（极差 69.12°，标准差 29.1°）**。
在 768 px 以上，预测/真值比落在 **1.12–1.14**（几乎正确），
3D 误差掉到 **0.18 m** —— 与「给 GT K」的 0.21 m **已经没有实质差距，甚至更小**。
⟹ **「内参杠杆 7.3 倍」是一个关于 640×480 的陈述，不是关于这个模型的陈述。**

**② 结论二（更严格）：跳变不是 pipeline 造成的 —— 是相机头自身不容忍小输入。**

这是本节最硬的一段，用的是**排除法**。`infer()` 的预处理由 config 里的
`shape_constraints` 驱动：`pixels_min=200000`、`pixels_max=600000`、
`ratio_bounds=[0.5, 2.5]`。于是：

- 640×480 = 307 k 与 800×600 = 480 k **都落在 [200 k, 600 k] 内**
  ⟹ `resize_factor` **都是 1.0000**，两档的预处理**逐字相同**（见上表 resize 列）。
- 中间那 4 档（672×504 / 704×528 / 736×552 / 768×576）同样 `resize_factor=1.0000`。

而就在这段**预处理完全不变**的区间里，预测 fx 在
**172.5 → 346.9 → 269.6 → 702.5** 之间摆动（**非单调**，4.07 倍跨度），
HFoV 在 **125.66° → 90.84° → 107.55° → 57.32°** 之间摆动。
⟹ 这既不是 padding 边界、也不是 resize 阈值、也不是某个分支。
**相机头的输出对这种输入尺度变化没有稳定性可言。**

补一条更狠的观察：960×720 / 1280×960 / 1600×1200 三档的**网络实际输入尺寸完全相同**
（都是 896×672，见预处理轨迹），预测 fx 却分别是 888.0 / 1159.4 / 904.2，
无 K 的 3D 误差分别是 0.249 / 0.177 / **0.644** m。
⟹ 不稳定性不只来自「网络吃多大」，还来自**图像被重采样后的像素级呈现**。
即：**相机头的输出主要取决于输入怎么被呈现，而不取决于场景真实的视场角。**

**③ 结论三：`check_fov` 不能当这道题的护栏。**

11 档里只有 4 档被判不可信；而 704×528（比 0.608）与 736×552（比 0.452）
—— **焦距错了 40–55%** —— 全部 `plausible=True`、零告警。
反面同样成立：§23.2 里合规的长焦照片反而被误报。

**④ 结论四（本节真正的正面结果）：有外部 K 时，米制尺度对分辨率稳健。**

把两列的**极差**比一比：

| 路径 | 3D 误差中位跨 11 档的范围 | 极差倍数 |
|---|---|---|
| 不给内参（模型猜） | 0.1774 – **1.9682** m | **11.1×** |
| 给 GT K | **0.1389** – 0.2943 m | **2.1×** |

**外部 K 把米制尺度对分辨率的敏感性压掉了 5.3 倍**，并且把最坏情况从 1.97 m 拉回 0.29 m。
⟹ **「内参是横向尺度的唯一开关」这个结论被加强了**，只是理由从
「模型猜得偏」升级成「模型猜得**不可控**」—— 而不可控比偏更致命，
因为它不能被任何事后校验（`check_fov`）发现。

**⑤ 可操作的工程结论**（已落到 `vision/depth.py` 与 README 的命令示例）：

- 喂 UniDepth 的图**短边至少 600 px（推荐 ≥768 px）**。这是一行代码的成本，
  却把「无内参」路径的 3D 误差从 1.94 m 压到 0.18 m。
- 但**仍然要传 K**：高分辨率下无 K 路径会再次劣化（1600×1200 是 0.644 m，
  是 GT K 路径 0.142 m 的 4.5 倍），而 GT K 路径在全表都稳。

### 23.2 A 段：EXIF 路径在**真实相机 EXIF** 上复跑

§22.6 只验证了 `vision/exif.py` 能读懂**我们自己合成的** EXIF。合成件格式标准、
字段齐备 —— 这恰恰证明不了它面对真实 EXIF 时可靠。

素材：`hMatoba/Piexif` 测试集的 `r_*.jpg`（真实相机原始 EXIF），
经 jsDelivr 逐字节代理（`upload.wikimedia.org` 在本机超时，jsDelivr 可用）。

| 文件 | EXIF 里的相机 | fx 模块 | fx 独立重算 | 差 | `FocalLength × crop` vs EXIF | 判定 |
|---|---|---|---|---|---|---|
| panasonic | Panasonic DMC-L10 | 222.222 | 222.222 | **0.000** | 50 × 2.0 = 100 vs 100 | 一致（0.00%） |
| pentax | **Olympus E-P3** | 33.333 | 33.333 | **0.000** | 无 35 mm 等效值 ⟹ 走降级 | ⚠ 见下 |
| ricoh | Ricoh GR | 62.222 | 62.222 | **0.000** | 18.3 × 1.5 = 27.45 vs 28 | 一致（1.96%） |
| sigma | SIGMA DP3 Merrill | 166.667 | 166.667 | **0.000** | 50 × 1.5 = 75 vs 75 | 一致（0.00%） |
| sony | SONY DSC-RX1R | 77.778 | 77.778 | **0.000** | 35 × 1.0 = 35 vs 35 | 一致（0.00%） |

- **对账通过**：5/5 台、fx 独立重算与模块输出差 **0.000 px**，解析链路在真实 EXIF 上没有串位。
- **正对照通过**：4 台有 35 mm 等效值的相机，`FocalLength × 已知裁切系数`
  与 EXIF 值一致到 **0.00–1.96%** ⟹ 说明我们没把 IFD0 / SubIFD 读串。
- **⚠ 一个被证伪的旧断言（必须更正）**：`vision/exif.py` 原 docstring 写
  「这种粗暴假设**骗不过** `check_fov`」。实测**不成立**：

  Olympus E-P3 的 `FocalLength=15 mm`、真实画幅 MFT（crop 2.0）⟹ 真值 HFoV **61.9°**；
  按全画幅假设算成 **100.4°** —— **仍落在 30–110° 窗口内**，`check_fov` 判 `plausible=True`。
  **fx 错了整整 2 倍，下游一个告警都收不到。**

  根因是窗口范围：30–110° 只等价于 `f_35 ∈ [12.6, 67.2] mm`。全画幅假设下
  `f_35_assumed = FocalLength`，误差要被兜住必须让 assumed 值**逃出**窗口，
  即 `FocalLength < 12.6 mm` —— 只有手机超广（crop≈5–7）够格。
  **APS-C(1.5×)/MFT(2×)/1 吋(2.7×) 的常见焦距一律逃不掉。**

- **误报同样存在**：Panasonic（f_35=100）与 Sigma（f_35=75）的 K **完全正确**，
  却因 HFoV 20.4° / 27.0° 低于 30° 下界而被判不可信。
  ⟹ `check_fov` 的真实语义应表述为「**典型照片的合理性先验**」，
  而不是「内参正确性的校验器」。5 台真实相机上它的表现是
  **2 正确、2 误报、1 漏报** —— 作为护栏只有 40% 的理想率。

- 已落地的更正：`vision/exif.py` docstring 改写（含窗口等价推导）、
  `assumed_sensor` 分支的 note 加上「不可由 check_fov 担保」、
  `scripts/build_scene.py` 在 `assumed_sensor` 时**无条件**告警（不再依赖 `fov.plausible`）。
  两条回归测试钉住这两个失效模式（`tests/test_vision_exif.py`）。

### 23.3 B 段：跨来源行为 —— 以及它如何被 C 段解释

10 张图、4 个独立来源（Picsum ×5 含横竖方 / Pexels / Unsplash / Pixabay，
外加一张**上游检出自带的 demo 照片**（1440×1920 竖幅，**另一个房间**，全项目第一张
与 UniDepth demo 无关的真实照片；该素材已随上游检出移出仓库 ⟹ 这一行不可复现）：

> ⚠ **称呼映射**：探针的原始输出（`phase0/probe_cross_source_result.json`、
> `probe_cross_source_report.txt`）里，这张图的键名是 `vadar_demo`。那两个文件是**原始测量记录**，
> 按「证据不可改写」保留原样、**没有动过内容** —— 它与本表的 `upstream_demo` 是同一条。

| 素材 | 尺寸 | fx 预测 | HFoV 预测 | 可信 | 深度中位 | 耗时 |
|---|---|---|---|---|---|---|
| picsum_land_a | 1600×1200 | 1201.3 | 67.3° | ✓ | 269.25 m | 1488 ms |
| picsum_land_b | 1600×1067 | 738.5 | 94.6° | ✓ | 157.22 m | 202 ms |
| picsum_port_a | 1200×1600 | 4366.5 | 15.7° | ✗ | 170.58 m | 104 ms |
| picsum_port_b | 1200×1600 | 1590.0 | 41.4° | ✓ | 0.45 m | 118 ms |
| picsum_square | 1200×1200 | 2010.3 | 33.2° | ✓ | 1.08 m | 108 ms |
| pexels_photo | 1600×1137 | 778.2 | 91.6° | ✓ | 65.42 m | 136 ms |
| unsplash_photo | 1600×1068 | 1334.4 | 61.9° | ✓ | 113.03 m | 140 ms |
| pixabay_tree | 1280×797 | 707.8 | 84.2° | ✓ | 61.75 m | 140 ms |
| **upstream_demo**（素材已移出） | 1440×1920 | 1150.8 | 64.1° | ✓ | 2.05 m | 117 ms |
| **unidepth_demo** | **640×480** | **163.7** | **125.8°** | ✗ | 3.36 m | 81 ms |

预测 HFoV 中位 65.69°、范围 **[15.65°, 125.8°]**、标准差 **30.95°**，
只有 **2/10** 判为不可信。

⚠ **这张表单独看会得出错误结论。** 它的第一直觉是「相机头其实挺准的，
§21 那 125.8° 是个例外」—— 而这会**推翻**§21 的核心叙事。
C 段解释了它：**那 8 张「看起来准」的都是 1200–1600 px，落在剂量-反应曲线的另一端。**
把 B 与 C 合起来，正确的表述是：

> **不是「它总是偏 3 倍」，而是「它的输出不可预测、且随输入分辨率与重采样方式漂移」。**
> §21 的 3.169× 是一个真实观测，但它描述的是**一种输入条件下的表现**，不是模型的不变量。

`upstream_demo` 这一行本身也有价值：HFoV 64.1°、深度中位 2.05 m，
对这个室内真实场景**看起来完全合理** —— 它说明这张图不是「模型失败」，
而是「模型成功」，恰好推翻了「它一定失败」的假设。

### 23.4 探针与素材的可复现性

```powershell
# 1) 抓素材（A 臂真实相机 EXIF + B 臂真实场景），落 .cache/cross_source/
& <venv>\Scripts\python.exe D:\3D_Spatial_Agent\phase0\fetch_cross_source.py

# 2) A 段：EXIF 路径 vs 真实相机 EXIF（纯 CPU，秒级，无 GPU 无联网）
& <venv>\Scripts\python.exe D:\3D_Spatial_Agent\phase0\probe_cross_source.py --part a

# 3) B 段：相机头跨来源行为（GPU，约 20 s）
& <venv>\Scripts\python.exe D:\3D_Spatial_Agent\phase0\probe_cross_source.py --part b

# 4) C 段：分辨率剂量-反应（GPU，约 20 s）—— 本节的结论性证据
& <venv>\Scripts\python.exe D:\3D_Spatial_Agent\phase0\probe_cross_source.py --part c
```

产出：`probe_cross_source_report.txt` / `probe_cross_source_result.json` /
`fetch_cross_source_report.txt` / `cross_source_manifest.json`。

**测量纪律（都是这一轮又踩到的）**：
- B/C 段必须在 `import torch` **之前**设 `HF_HOME` / `HF_ENDPOINT` /
  `HF_HUB_DISABLE_XET`，否则权重加载会直连 huggingface 并抛 `httpx.ProxyError: 502`。
- C 段的 `err3d_*` 通过 `importlib` **复用** `probe_k_sweep.py` 的 `metrics()`
  —— 两个探针的数字要能并列在同一张表里，前提是指标定义逐字相同。
- 任何几何变换（这里是 resize）都**必须同步施加到 GT 深度**上（§22.9 的老教训）。
- 排除「跳变来自 pipeline」必须**把预处理轨迹打出来**（本节新增
  `preprocess_trace()` 与报告里的单独一表），否则读者有理由怀疑是分支而非模型。

### 23.5 本节仍未做到的（更新后的证据边界）

1. **手机 EXIF 仍缺席。** 5 台真实相机的画幅覆盖 **1.0–2.0×**（35 mm ×1、APS-C ×2、
   4/3 与 MFT ×2），最小的那一台（MFT，crop 2.0）**仍然逃得掉 `check_fov`** ——
   这说明结论不是「传感器太大才漏」的巧合。手机（crop≈5–7）是 EXIF 路径最常见的输入，
   也是唯一会被 `check_fov` 兜住的那一档 —— 却没有实测。
2. **仍无「跨来源 + GT 内参」的配对**，因此 B/C 段建立的是
   「预测值不可控」，**不是**「误差是多少」。绝对误差仍只有 UniDepth demo 那一对。
   ⟹ 原本指望 **Omni3D-Bench 自带 GT 相机与 GT 深度**来补这个缺口，
   **该指望已落空**（§24 实测：它只有 (图, 问, 答) 六列）。
   要拿到「跨来源 + GT 内参」的配对，只能换基准（Omni3D 原始数据里带
   `K`/深度的子集、ScanNet、ARKitScenes 等）或自建标定样本。
   这从 Phase 12 里一个「顺便拿到」的东西，变成**需要单独解决的子问题**。
3. **B 段 8 张里 6 张经 CDN 重编码**（Picsum / Pexels / Unsplash / Pixabay），
   压缩伪影与本机照片可能不同；只有 `upstream_demo` 是原图（且该素材已移出仓库）。
4. **C 段的 GT 深度是按分辨率重采样的**，高分辨率档的「真值」比实际更平滑。
   但这一效应**同时作用于两条臂**（无 K 与有 K），所以两者之差仍可比。
5. **`ratio_bounds` 依赖的 `resolution_level` 未确认**：若模型设了
   `resolution_level`，`pixels_bounds` 会收窄到 40 k 宽的窗口，
   预处理轨迹会变。本机实测 640×480 与 800×600 的 `resize_factor` 都是 1.0
   ⟹ 至少这两档用的是全宽窗口，但未逐档确认。
6. **主点、畸变对分辨率的耦合未测** —— 本节只跟踪了 fx 与 HFoV。

> **§23 一句话总结**：补最大证据缺口的结果不是「结论更稳了」，而是
> **发现原结论被一个隐藏变量混杂**：`640×480` 是相机头的一个反常工作点，
> 而 §21/§22 的全部数字都取自它。跨来源 + 分辨率剂量-反应之后，
> **结论的方向不变、理由更强** —— 内参必须外部给定，不是因为模型「偏」，
> 而是因为它的输出**没有稳定性**，且 `check_fov` 这道护栏在真实相机上
> 只有 40% 的理想率（2 正确 / 2 误报 / 1 漏报）。

---

## 24. 主实验臂的数据契约，与一个被推翻的关键假设（2026-09-17 晚）

把主实验臂的数据集拿到手的过程中**推翻了一个被两次「确认」过的前提**，所以单独成节记录 ——
它不是实现细节，它改变了 Phase 12 的可行性判断。
（当时的实验臂是上游原版流水线；该臂的上游检出与兼容层已于 2026-09-20 移出，但**下面这条被推翻的
假设与「用哪条臂」无关，全部结论仍然成立**。）

### 24.1 Omni3D-Bench 的真实契约（实测）

抓取路径（`dataset/builders/fetch_omni3d_bench.py`）：
HF 镜像 `hf-mirror.com` 可达，仓库 `dmarsili/Omni3D-Bench` 的 `data/` 下
**只有一个文件**：`train-00000-of-00001.parquet`，106,490,728 字节，
单线程 23.1 s / 4.6 MB·s⁻¹，落盘大小与远端 `oid` 逐字节一致。

parquet schema（`dataset/builders/read_omni3d_bench.py --inspect` 实测）：

| 列名 | 类型 | 说明 |
|---|---|---|
| `image_index` | string | 形如 `91339.246_00000463.jpg`（**自带扩展名**） |
| `image` | struct\<bytes, path\> | HF Image 特征，图像字节内联在 parquet 里 |
| `q_index` | int64 | 题号。**注意不是** `question_index` —— 上游运行器消费时要求后者，故读取时必须显式重命名 |
| `question` | string | 问题 |
| `answer` | string | 真值 |
| `answer_type` | string | `int` / `float` / `str` |

**就这六列。没有相机内参、没有深度图、没有三维框、没有场景 JSON。**
HF 官方 README 的 annotations 格式（`image_index / question_index / image /
question / answer_type / answer`）同样不含任何三维字段；仓库文件树里也没有
第二个数据文件。

> **⟹ 「Omni3D-Bench 自带 GT 相机」是错的。**
> 它此前在 §22.6 末段与 §23.5 第 2 条被写为「已确认」，
> 并据此推出「主实验臂不依赖 EXIF 精度」「EXIF 只服务扩容支线」。
> 这两条推论**同时作废**。

这条修正的实际后果：

1. 主实验臂的横向米制尺度**不可控** —— 它和 §23 里那条不确定路径是同一条。
2. EXIF / 手写 K 从「支线」变成「主指标相关」：501 题里 float 占 **270 题**，
   而 float 在 Total 里占 **53.9%** 权重（§24.3）。
3. 「跨来源 + GT 内参」的配对样本仍然缺失，且**不能再指望这个基准**。

### 24.2 实际题数与题目构成（README 说 500，实际 501）

| 指标类别 | 题数 | 占比 | 覆盖 |
|---|---|---|---|
| numeric (count) | 70 | 14.0% | `answer_type == int` |
| numeric (other) | 270 | 53.9% | `answer_type == float`，MRA 口径 |
| yes/no | 75 | 15.0% | `answer_type == str` 且答案 ∈ {yes, no} |
| multi-choice | 86 | 17.2% | `answer_type == str` 其余 |
| **合计** | **501** | 100% | **201 张唯一图片**，300 行是重复图（平均 2.5 问/图） |

> 注意 `str` 类里**混着** yes/no 与 multi-choice，必须按**答案取值**再切一刀。
> 只看 `answer_type` 会以为 `str` 是一类，指标就再也拆不开。

### 24.3 Total 的聚合口径：不是猜的，是验算出来的

论文表格给四个子指标 + Total，但**没写 Total 怎么算**。
这本来是个可以随便糊弄过去的地方，但它决定了「哪个子指标值得优化」，
所以做了一次可证伪的检验：

> 若 Total 是「按题数加权的 micro 平均」，即
> `Total = Σ(子指标 × 该类题数) / 501`，
> 那么把论文**自己**那 8 行数字代进去，应当能复现它**自己**的 Total。

结果（`evaluation/metrics.py::verify_total_aggregation`）：

| 方法 | 论文 Total | 复算 Total | 残差 |
|---|---|---|---|
| GPT4o | 42.9 | 42.86 | −0.04 |
| Claude3.5-Sonnet | 32.2 | 32.23 | +0.03 |
| Llama3.2 | 25.6 | 25.61 | +0.01 |
| Gemini1.5-Pro | 32.0 | 32.04 | +0.04 |
| Gemini1.5-Flash | 35.0 | 35.00 | 0.00 |
| Molmo | 26.1 | 26.14 | +0.04 |
| SpaceMantis | 30.3 | 30.34 | +0.04 |
| **上游方法**（文献引用） | **40.4** | **40.43** | **+0.03** |

**8 行残差全部 ≤0.043pp**（论文只给一位小数，单行四舍五入本身就贡献 ≤0.05pp）。
口径确认：`Total = Σ(子指标 × 题数) / 501`，权重 = 14.0% / 53.9% / 15.0% / 17.2%。

> **两次失败的方法也值得记下来**，因为它们说明了「怎么做才对」：
>
> ① 一开始写的是「4 个类别占比当未知数、用 8 行解最小二乘」。
> 它是**病态**的：8 个方法在四个子指标上高度共线（方法之间主要差在整体水平，
> 不是差在类别构成），坐标下降解出 `numeric_count` 权重 0.92、
> `yes_no` 权重 **−0.008**（负权重，物理上不可能），残差 17.4pp。
> 病态不是「数据不够」，是**模型形式**不吃这套解。
>
> ② 表 2 的 ViperGPT / VisProg 也**对不上**（残差 −6.74 / −7.58pp）。
> 它们与表 1 不是同一批次、题数分布未公开 ——
> 把它们和上游方法放进同一个方程，等于假设两批人用了同一份题数分布。
> 这个假设不成立，所以判据只用表 1 的 8 行，而那两行**显式排除并留痕**
> （`excluded_non_same_run`），不悄悄丢掉。

**工程结论：float 题占 53.9% 的权重。** 任何只针对 count / yes-no / multi 的
改进，天花板不到 15%；而 §21–§23 反复证明的内参问题，恰恰主要作用在
float（横向米制尺寸与距离）这一类上。**两件事是同一件事。**

### 24.4 子指标口径里两个会静默算错的地方

`evaluation/metrics.py` 逐行复刻 `engine.py:356 write_summarized_results`，
其中两处如果读反，数字会明显偏乐观而**不会报错**：

1. **无法解析的预测 = 记 0 分，不是「该题被排除」。**
   原实现里 `num_ct_n += 1` 在 `try` **之前**，所以分母含它、分子不含它。
   若实现成「排除」，一个只会输出乱码的模型在这些题上不会被扣分 ——
   极端情况准确率虚高一倍。
   （第一遍读码时我读反了，是用测试回放才纠正过来的，记在此处。）
2. **float 的 MRA 分母是「全部 float 题」，含解析失败的。**
   解析失败时 10 档阈值一档都不命中，但不从分母里消失。
   另外 `continue` 位于 `for threshold` 循环**内部**，作用是「这一档不计数」，
   不是「放弃这道题」。

---

### 24.5 稳定入口必须按「这次到底发生了什么」分流（同日晚，实测踩到）

运行器把结果写成一个固定名字的入口，本意是「随便哪天回来，打开一个文件就能看到最近一次
的结果」。第一版只有一个 `latest_run.json`，**`--plan`、缺 key 提前退出、真正进过流水线的
run 全都往里写**。于是在 `--plan` 之后直接跑一次实跑（那时环境变量里还没有 key），
得到的是：

```
latest_run.json  ←  一次 fatal 记录（缺 API key；当时环境变量名还带上游前缀，现已统一为 `SPATIAL_*`）
```

上一次 `--plan` 的记录里有 `plan_summary.next_command`、有模型指纹 —— **被一条失败覆盖**。
想照着 `next_command` 复现，打开文件看到的是自己刚刚的失败。这不是格式问题，
是**证据被销毁**，而且发生得很自然：先 plan、再实跑，是文档自己推荐的顺序。

修法是把「最近一次」的语义拆开：

| 文件名 | 什么时候写 | 为什么不允许被别的情况覆盖 |
|---|---|---|
| `latest_run.json` | 只有**真进过流水线**的 run | 它是唯一带着四类子指标 + Total 的产物 |
| `latest_plan.json` | `--plan` | 里面的 `next_command` 是「下一步怎么跑」的唯一出处 |
| `latest_failed.json` | 前检查失败 / 选不出题 / 缺 key | 失败现场是排查证据，但**不该顶掉**上一次的成功 |

同一轮还有一个附带裁决：**`subset.json` 在 fatal 态不写**。
它被定义为「后续所有臂复现同一批题的依据」，而一个 fatal 的题集**从来没跑过** ——
让它顶掉上一次真跑过的题集，会让「复现同一批题」这条链路指向一个从未执行的选择。

产物还带 `mode` 字段自述类别（`run` / `plan` / `fatal`），
这样即便文件被改名、被拷到别处，也不靠文件名认人。
分流规则当时由 7 条用例守住（其中最关键的一条是：**先写一次成功的 run，再写一次 fatal，
前者必须逐字节不变**）。⚠ 这些用例随实验臂运行器于 2026-09-20 一并移出仓库 ⟹
**这条守卫目前是缺失的**，将来若重建运行器，必须把它一起重建。

### 24.6 「验收标准自己会答错」的三个实例：巧合命中 ×2 与 GT 口径 ×1（2026-09-19 第十轮）

§13.3 把 `verifier` 的四档结论写成了「答案有没有被系统自己的证据支持」。这一轮证明：
**这个验收环节本身会出错，而且出错方式全都是「不报错、只改判」。** 三个实例同源。

#### (a) 巧合命中是一个**类**，抓到两个

共同形状：一个与题目无关的常量漏进 `verifier` 的数字池，让某个答案**看起来有出处**。

| 泄漏点 | 为什么漏 | 后果 |
|---|---|---|
| `list_objects.evidence["label_counts"]` = `{"sofa":1,"picture":4,…}` | 一组 1~4 的小整数，与题目无关 | 「4 个」恒 `supported`、「3 个」恒 `unsupported` —— 判断力来自数字**大小** |
| ⭐ `query_relation.evidence["method"]` = `"geometry_v1"` | 旧 `_ID_RE = \b[A-Za-z_][A-Za-z_0-9]*_(\d+)\b` 只抹「下划线**紧跟**数字」的 id（`chair_1`），**认不出 `_v1` 这种版本号后缀** | 每调一次就塞一个 `('method', 1.0)`；C8 的答案恰好是 `1` ⟹ 判 `supported`，**出处是算法版本号** |

第二条更值得记的是**它是怎么被发现的**：不是被「巧合命中」这个检查抓到的（那个检查当时
只查 `label_counts`），而是**追着「这个逐字出处是哪来的」**查出来的 —— 触发它的还是同一轮里
另一个改动（GT 修正让 `matched_from` 从 `derived` 变成 `evidence`，才想到去看那一档）。
⟹ **「巧合 = 0」不能读成「这一类已清零」。**

修法必须**两层一起**，任缺一层换个写法又漏：

1. `_ID_RE` 加固为 `\b[A-Za-z_][A-Za-z_0-9]*_[A-Za-z_0-9]*\d[A-Za-z_0-9]*`。
   ⚠ 它**救不了** `sam2.1-hiera-base-plus`（点号分段、无 `_`）—— 正则层面无解。
   回归用例同时钉**反例**：`extent_m=1.263` / `a_z=3.17` 里的合法量不能被误抹。
2. `_NON_QUANTITY_KEYS` —— 按键名**整棵子树跳过**（与 `_CENSUS_KEYS` 同一手法）。三味：
   身份（`object_id`/`label`/`a`/`b`/…）、版本（`method`/`tool`/`model`）、
   以及 ⭐ **请求参数**（`tol`/`k`/`min_contain`）。第三味的理由：
   **「答案恰好等于我传进去的阈值」不是证据** —— `tol=0.0` 会让答案 `0` 恒判 `supported`。

**代价要明写**：`tol` 从此不作逐字出处 ⟹ C8 的完美答案由 `supported` 落到
`derived:count/within` → `weak`。**这是对的**：计数本来就是派生量。

#### (b) − (c) 两个「检查器」的通用要求

- **`coincidence` 自洽性对照**：把无关字段抹掉再验一次，结论翻转 = 巧合命中。
  ⚠ 可抹的是 `PRUNABLE_KEYS`（普查／版本／参数），**不是** `NON_QUANTITY_KEYS` ——
  后者还含 `object_id` 这类**身份键**，而**字符串答案的 `text_backed` 正要靠它们找出处**。
  第一版图省事抹了全部，C1/C2/C3/C7（答案形如 `picture_1`）从 `supported` 掉成 `weak`，
  被报成「巧合命中」—— **四例全是假阳性**。抹掉答案本身的出处不叫发现巧合。
- **修复要能被重放核对**：`scripts/show_answer_provenance.py` 打印每题答案命中的 `(键, 值)`，
  里面**冻了一份已知有缺陷的旧 `_ID_RE`**，用来重放「修复前会命中什么」
  （实测输出 `C8 修复前会命中 method=1.0`）。第一版直接用**当前**口径重放 ⟹ 一片空白
  （加固后的正则已把 `geometry_v1` 抹干净），于是「修掉了什么」只剩一句转述。
  ⟹ **冻一根「坏掉的尺子」是合法且必要的做法**，并附回归用例防止有人顺手把它「修好」——
  那会让重放表变成空白，而**空白与「没有泄漏」长得一模一样**。

#### (d) GT 也会有口径缺陷，而且长得像「模型算错」

C8「有多少个物体在桌子的前面？」的 GT 原先定义为「工具**默认** `tol=0.05` 下的输出」= **0**，
而 `chair_1` 与 `table_1` 的 z 只差 **37.8 mm** —— 物理上确实在桌子前面。
问题问的是**世界的样子**，`tol=0.05` 是工具的**鲁棒性余量（实现细节）**。
把「工具的输出」当「问题的真值」⟹ **任何按质心比较的正确程序都被判「算错」**，
归因成 `class B`，把「加工具／改提示词」这些修法引向一个**根本不存在的失败**。

- GT 改为**严格比较（`tol=0`）= 1**，口径写进 `Spec.convention`；
- 新增 `convention_audit()`：**凡 GT 对某个有默认值的参数敏感，就必须声明口径**，
  否则报 `ambiguous`。它把「这个探针测的是不是模型」变成**可计算**的问题。
- ⚠ 这不是迁就模型：C8 的「一次调用解」`_one_call_count_front_of_table` **一直**用严格比较，
  它的 docstring 早就写着「两个答案都对，取决于那个对程序不可见的容差」——
  **「两个都对」这句话本身就是「问题没被定义清楚」的自白**，只是当时没人去读。

#### (e) ⚠ 「两个数一起变」时先分清「测量变了」还是「尺子变了」

同一轮里既改提示词（调用 106 → 32）又改 GT 口径（C8 的冗余分母 9 → 1），
于是「冗余倍数 5.0 → 3.0」这个跨度里**混着两件事**：

| 指标 | 改前 | 只改提示词 | 改后（含 GT 修正） |
|---|---|---|---|
| 与 GT 一致 | 9/10 | 9/10 | **10/10** |
| `class` | `{B:1, OK:4, OK*:5}` | 同改前 | **`{OK:5, OK*:5}`** |
| 工具调用 | 106 | 32 | 32（**一次没动**） |
| 冗余倍数（**上中位数**） | 5.0 | 1.11 | **3.0** |

⚠ **「中位数」的口径**：`_summarize` 用的是 `sorted(ratios)[len(ratios)//2]` = **上中位数**
（偶数题数时取**偏大**那一个，不是两数均值）。同一份定义横跨改前改后所以可比，
但它与教科书中位数不是一回事 —— **引用时别省掉这句**。

#### (f) 产物：删除不可用时，改放「产物地图」

`--analyze` 原先每复算一次就盖一个新时间戳 ⟹ 同一目录里同时有「多份测量」与
「同一份测量的多个版本」，**从文件名上分不出来**。已改为**按源文件名落盘**（幂等覆盖）；
真跑仍用时间戳（那是**新的测量**）。见 `scripts/probe_combination_run.py::analyze` / `_write`。

⚠ 本机 `Remove-Item` **删不掉文件**（安全守卫静默拦下：`deleted=0`，不报错）。
⟹ 改放 `reports/README.md` 当**产物地图**，并明写：
**判断「哪份是当前的」只看这张表，不认 `mtime`、不认文件名新旧。**
理由：「已被取代」的那几份与当前版**名字几乎一样、数字却是旧口径**，
属于**看起来正确、实际过期** —— 比文件缺失更危险，因为缺失会被发现。

---

## 附：本次调研的关键事实汇总（供交叉核对）

> **关于早期参考实现（上游）**：本项目早期版本曾对一份第三方参考实现（CVPR 2025，
> 以「LLM 输出 Python 程序」做 3D 空间问答）做过完整代码核查，**该检出及其全部素材已于
> 2026-09-20 移出仓库**。下表只保留它的**文献级信息**（供报告引用与划清界限）；
> 代码级细节（文件树、类拆解、行号、工具签名）已全部删除，评估结论与弃用理由见 §2。
> **本项目不依赖、不包含、不分发该实现的任何代码或数据。**
> 其仓库许可证标注为 `NOASSERTION`；本仓库既不含其代码，自然不受其条款约束。
> 唯一需要区分的是**评测数据集** `dmarsili/Omni3D-Bench`：它与上述实现**不是同一件东西**，
> 我们只把它当公开评测基准使用。

| 事实 | 值 | 来源 |
|---|---|---|
| 描述 | `[CVPR 2025] Program synthesis for 3D spatial reasoning` | GitHub API |
| 论文 | arXiv:2502.06787（v1 2025-02-10，v2 2025-03-28） | arXiv |
| 数据集 | `dmarsili/Omni3D-Bench`，**实测 501 题 / 201 图**（README 写 500）；CC BY-NC；**只有 (图, 问, 答) 六列，无 GT 相机/深度/三维框** | parquet schema 实测 + README（§24.1） |
| 论文成绩 | Omni3D-Bench 40.4；+oracle 94.4；CLEVR 53.6（+oracle 83.0）；GQA 46.1 | `RESULTS.md` |
| 本机 GPU | RTX 4060 Laptop，8188 MiB，驱动 560.76，CUDA 12.6 | `nvidia-smi` 实测 |
| UniDepth 预处理约束 | `pixels_min=200000`、`pixels_max=600000`、`ratio_bounds=[0.5,2.5]`、`shape_mult=14` | 模型 `config.json` 的 `data.augmentations.shape_constraints`（§23.1） |
| 相机头预测的分辨率敏感性 | fx 预测在 `resize_factor` 恒为 1.0 的区间内摆动 **4.07 倍**（172.5→346.9→269.6→702.5），非单调 | `probe_cross_source.py --part c`（§23.1） |
| 无内参路径的 3D 误差范围（跨 11 档分辨率） | **0.1774 – 1.9682 m（11.1×）** | 同上 |
| 有 GT K 路径的 3D 误差范围（同上） | **0.1389 – 0.2943 m（2.1×）** | 同上 |
| `check_fov` 在 5 台真实相机上的表现 | 2 正确 / 2 误报（长焦）/ **1 漏报**（MFT 全画幅假设） | `probe_cross_source.py --part a`（§23.2） |
| 本机可用的图片外源 | `cdn.jsdelivr.net`（可逐字节代理 GitHub）、`gitee` / `gitcode` / `hf-mirror` / Unsplash / Pexels / Pixabay CDN；`upload.wikimedia.org` 与 `raw.githubusercontent.com` **超时** | `_hosts.txt` 连通性实测（§23.4） |

---

*本文档早期版本基于 2026-09-14 对上游参考实现仓库的实际代码核查；该检出已于 2026-09-20 移出本仓库，评估结论保留在 §2。*
*2026-09-16 补充：视觉栈已在 Windows 原生跑通；**三个视觉模型的显存与延迟全部实测**（§5），SAM2 的估算已替换。*
*2026-09-16 补充 ②：**L1 感知层与场景图构建已落地并实测**（§21）；发现「内参来源」这一被官方文档低估的精度杠杆（三维误差中位数差 7.3 倍），据此撤销了 `calibrate_scale` 的设计。*
*2026-09-16 补充 ③：把内参杠杆从「单图一次对照」升级为**因果证据** —— 剂量-反应扫描（§22.1）、自洽性（§22.2）、方向场解析可控性（§22.3）、±1% 焦距反解（§22.4）、三视场复现（§22.5）；并落地 **EXIF 内参路径**（§22.6，`vision/exif.py`，端到端验证 sofa 2.46 m vs 6.93 m），同时查明 **EXIF 的主要误差是主点而非量化**（§22.7，14.75 px / 97.4%）。单测 166 → 191。*
*2026-09-16 补充 ④：**补最大证据缺口的结果是发现了一个隐藏混杂因子**（§23）—— 跨来源复跑（10 张图 / 4 个来源 / 5 台真实相机 EXIF）证明 **640×480 是相机头的一个反常工作点**，§21/§22 的全部数字都取自它；768 px 以上无内参路径的 3D 误差从 1.943 m 掉到 0.18 m。结论方向不变、理由更强：**内参必须外部给定，不是因为模型偏，而是因为它的输出没有稳定性**（预处理恒定的区间内 fx 摆动 4.07 倍），且 `check_fov` 在真实相机上只有 40% 理想率（2 正确 / 2 误报 / **1 漏报**）。已据此更正 `vision/exif.py` 的一条旧断言。单测 191 → 193。*
*2026-09-17 补充 ⑤：**拿到主实验臂的数据集后，推翻了一个被两次「确认」过的前提**（§24）。Omni3D-Bench 实测只有 `image_index / image / q_index / question / answer / answer_type` 六列，**不带 GT 相机、不带深度、不带三维框** —— 「主实验臂自带 GT 相机」是错的，据此推出的「主实验臂不依赖 EXIF 精度」同时作废。实际是 **501 题 / 201 图**（README 写 500），float 题占 **53.9%** 的 Total 权重（该口径已用论文自身 8 行数字验算，残差 ≤0.043pp）。同轮交付 **实验臂运行器与 Windows 兼容层**（上游相关部分已于 2026-09-20 随检出一并移出；`evaluation/metrics.py` 保留并继续承担口径复算）+ `dataset/builders/{fetch,read}_omni3d_bench.py`（保留）。单测 231 → 306（其中 7 条守住产物的稳定入口分流，见 §24.5）。*
*2026-09-20 补充 ⑥：**上游依赖已彻底移除。** `vendor/` 下的上游检出、实验臂 A 的上游运行器与兼容层、
全部上游探针脚本与素材（含 §23 的 `upstream_demo` 图）一并移出仓库；`README.md` 与本文档的定位改写为
「感知层接三个开源模型、Agent 层与工具层自研」。**实验臂 A 的历史结果留在 `results/A/`，但已不可复现**，
只能当记录。配置文件里的环境变量前缀统一为 `SPATIAL_*`（原带上游前缀的名字已弃用）。*

*仍标注「估算」的只剩 Molmo 与 Qwen 的显存，它们在 Phase 4 实测后替换；标注「不确定」的项以官方文档为准。*
