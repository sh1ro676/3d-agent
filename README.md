# 3D Spatial Agent

基于 **VADAR**（CVPR 2025, `damianomarsili/VADAR`）技术路线重写的三维空间智能体项目。
课程：3D 视觉 / 三维视觉算法。用途：课程大作业 + 答辩 + 简历。

> **项目根目录：`D:\3D_Spatial_Agent`**
> 2026-09-15 从 C 盘 WorkBuddy 工作区迁出，此后一律在 D 盘工作。

**一句话定位**：让 Agent 不再「看图猜空间关系」，而是「调工具算空间关系」。

---

## 两类输出（项目的核心契约）

| # | 输出类 | 内容 | 面向 |
|---|---|---|---|
| 1 | **问答输出** | 自然语言答案 + **米制数值** + 三维高亮 + 工具调用 trace | 单道题 |
| 2 | **场景级输出** | 结构化三维场景描述：物体清单 + 米制尺寸 + 两两几何关系（`scene_graph.json`） | 整张图 |

**输入**：1 张 RGB 图片（任意分辨率）+ 1 句自然语言空间问题。批量评测时用 `annotations.json`。

> 第 2 类**不依赖任何提问**，场景构建完成即可导出。它把项目从「三维问答工具」抬到
> 「三维场景理解系统」——这才是标题里 `3D Spatial` 二字应有的产出。详见方案文档 §1、§11.1（L5）。

**典型任务**：

> 用户：「哪把椅子离门最近？」
> Agent：`find_object("door")` → `find_object("chair")` → `get_3d_position()` →
> `calculate_distance()` → `argmin` → 回答「Chair 2，1.42 m」→ Viewer 高亮 → 可选 `move_camera()`

---

## 文档阅读顺序

1. **`docs/VADAR可借鉴性评估与路径选择.md`** ← **先读这份**，它取代主方案的部分结论（WSL 与「基于 VADAR 改造」两条）
2. `docs/3D_Spatial_Agent_技术调研与实施方案.md`（20 章主方案，`.html` 为阅读版）

所有关于 VADAR 的陈述都标注了 `文件:行号`；凡是估算值都显式标注「估算」，实测值才算数。

---

## 当前进度

| Phase | 状态 |
|---|---|
| **0 环境** | ✅ **已完成** —— 视觉栈 Windows 原生跑通，**未装 WSL** |
| 0.x 核心验证 | ✅ `probe3d.py` A–D 四段全 PASS；`probe_sam2.py` 补完最后一批估算值；`probe_depth_gt.py` 量化内参杠杆 |
| **1b 自建骨架** | ✅ **已完成（2026-09-16）** —— 信封 + 场景图 + 11 个工具 |
| **1c 感知层 + builder** | ✅ **已完成（2026-09-16）** —— `vision/` 七个模块 + `builder.py` + `store.py`；**193 用例全绿（2.8 s、零 GPU）**，已产出真实照片的场景图 |
| **1c.2 内参因果证据 + EXIF** | ✅ **已完成（2026-09-16 晚）** —— 剂量-反应扫描 + 数字变焦复现 + **EXIF 内参路径端到端跑通**；见下文「内参杠杆（二）」 |
| **1c.3 跨来源复跑** | ✅ **已完成（2026-09-16 深夜）** —— 10 张图 / 4 个来源 / 5 台真实相机 EXIF。**结论是发现了一个隐藏混杂因子**：`640×480` 是相机头的一个反常工作点，§21/§22 的数字都取自它。见下文「内参杠杆（三）」 |
| 1 VADAR 原版跑通 | ⬜ 对照臂 A，只需一个 cheap API key |
| 2 吃透 VADAR 架构 | ✅ 已完成（产出是设计决策，见方案文档 §4、§6） |
| 3–13 | ⬜ 见主方案文档 §18 |

**下一步**：把 `scene.json` 交给 L5 报告层（`describe_scene` / `diagnose_failure`）。
**手机 EXIF 仍是缺口**（5 台真机画幅覆盖 **1.0–2.0×**：35 mm ×1、APS-C ×2、4/3 与 MFT ×2；
最小的那台 MFT **仍然逃得掉 `check_fov`**，所以不是「传感器太大才漏」的巧合），
以及「跨来源 + GT 内参」的配对 —— 正解是 Omni3D-Bench，落在 Phase 12（方案文档 §23.5）。

> 已写进代码的两条硬约束：① `get_3d_position` **必须用 SAM2 掩码质心**，不能用检测框内中位数 ——
> 两者相差**均值 83 mm / 最大 208 mm**，而关系判断的容差是 50 mm。
> ② **内参必须作为输入传进来**，有就一定传 —— 它决定全部横向米制尺度（见下文）。
> 这两条都不是注释，而是 `Node.centroid_3d` 的语义与 `BuildConfig.known_intrinsics` 的存在理由。

---

## 三条硬约束（2026-09-16 修订）

| 约束 | 说明 | 何时相关 |
|---|---|---|
| `vendor/VADAR/` 目录名**必须**叫 `VADAR` | `predefined_modules.py:17` 硬编码 `from VADAR.prompts...` | 只在跑**实验臂 A**（原版基线）时 |
| VADAR 本体需 Unix（`signal.alarm`） | `engine.py:594`、`agents.py:572` | 同上；**我们自己的实现完全不需要** |
| 主模型必须多模态 | `vqa()` 内联 base64 图片发给模型 | 只在复用 VADAR 的 `vqa()` 时；自建架构已把视觉语义拆给独立小 VLM |

> **WSL2 已降为可选**：过去认为"必须 WSL2"的三处阻塞（`signal.SIGALRM` / GroundingDINO 编译 / SAM2 编译）
> 中，前两处随「不基于 VADAR 代码实现」而消失。**已实测整条视觉栈在 Windows 原生跑通。**

---

## 环境事实（2026-09-16 实测）

**硬件**：RTX 4060 Laptop **8188 MiB**｜驱动 560.76｜CUDA 12.6｜Ryzen 9 7945HX 16C/32T
**磁盘**：C 剩 21.8 GB｜**D 剩 381 GB（工作盘）**
**环境**：`venvs\vision`（Python 3.12.5）｜torch 2.6.0+cu124｜torchvision 0.21.0+cu124

**关键实测数字**：

| 模型 | 权重 | 峰值显存 | 延迟 |
|---|---|---|---|
| UniDepth V2 (vits14) | 130.4 MB | 486 / 604 MB | 739–1120 ms |
| GroundingDINO tiny | 657 MiB | 1761–1903 / 2152–2306 MB | 586–814 ms |
| SAM2.1 hiera-base-plus | 279.7 MiB | 647 / 928 MB | 282 ms（单框）· **179 ms（9 框批量）** |

**⭐ 三模型同时驻留只有 1200 MB = 8188 MiB 的 14.7%。** 注意区分两个量：
**常驻只是权重，峰值含推理激活**（算完即释放）。剩下约 6.9 GB 留给 LLM ——
这就是「本地 LLM 可选」这条路在显存上成立的原因。
（真实流水线复测：常驻 **1146.7 MB = 14.0%**，峰值 2177.2 MB，与探针值一致。）

**9 个框必须一次调用**：批量 200 ms vs 逐个 1512 ms，**快 7.57×** —— 这是调用形态而非优化项。

**`points` 是真几何（创新点 1 的地基）**：`infer()` 返回 7 个键，`points[2]` 与 `depth` **完全相同**
（差 0.000e+00），真实照片深度落在 **[1.376, 3.974] m**（单位就是米）。
→ **VADAR 拿到完整 XYZ 却只取 z 列**；升级到真三维不需要任何新模型、任何新显存。

---

## ⭐ 内参杠杆（2026-09-16 实测，本阶段最重要的发现）

第一次串起三模型时场景图里出现了 **6.70 m 宽的沙发**。一路查到源头：
**UniDepth V2 自己的相机头给错了内参**。

用仓库自带的配对 GT 深度图（`assets/demo/depth.png`）逐像素对比两条路径：

| 路径 | 深度 ARel | δ<1.25 | **三维误差中位** | 三维相对中位 |
|---|---|---|---|---|
| `infer(rgb)` —— 让模型猜相机 | 19.8% | 57.1% | **1.943 m** | 59.0% |
| `infer(rgb, camera=GT K)` —— 把相机告诉它 | 11.7% | 93.2% | **0.267 m** | 9.1% |

同一张图、同一份权重，**只因为内参来源不同，三维误差中位数差 7.3 倍**。
（真值 fx=518.9 → HFoV 63.3°；预测 fx=163.7 → HFoV 125.8°；横向放大 **3.169×**。）

**两点必须在报告里说清楚**：

1. 官方 README 把 GT 内参写成 "as **well**"（锦上添花），但官方 `scripts/demo.py`
   **自己就是传 camera 的** —— 作者验证效果时从不用纯 RGB 路径。
2. 传进去的 camera **不会**改写回传的 `intrinsics`（那是个独立预测头，
   传 GT 后仍返回 fx=163.7）。它走的是 `unidepthv2.py:361-362`，
   把 camera 转成 `rays` 喂进 decoder 当条件。所以 `DepthField.intrinsics`
   记的必须是**实际生效的那一份**。

**这不是「尺度校正」。** 内参错是**各向异性**的：fx 小 k 倍 ⟹ 横向坐标大 k 倍，
而 z 几乎不动（`depth` 就是 `points` 的 z 列）。一个全局 `scale_factor` 修不了它，
只会在修横向的同时把已经对的 z 一起弄错 —— 原方案里的 `calibrate_scale` 据此**撤销**。

正确做法是把内参当**输入**：

```powershell
& ...python.exe scripts\build_scene.py --image rgb.png --intrinsics auto `
      --scene-id living_room --prompt "sofa. chair. table. picture. mirror."
```

`--intrinsics` 接受 `auto`｜`.npy 路径`｜`"fx,fy,cx,cy"` 三种写法。
没有已知内参时会标 `intrinsics_source="predicted"` + 做视场合理性检查
（可信区间 30°–110°），不合理就**发警告**，让下游知道这批数字只能当相对量用。

**一个意外的二级效应**：内参错误还直接毁掉重力方向估计 ——
同一张图，tilt 从 **11.95°（reliable）** 变成 **63.04°（不可靠）**。
也就是说内参错会**静默翻转所有 `above`/`below` 关系**，不只是把尺寸放大。

> 详细对比表、分区统计与因果验证见方案文档 §21。探针：`phase0/probe_depth_gt.py`。

---

## ⭐ 内参杠杆（二）：从「一次对照」升级为「因果证据」（2026-09-16 晚）

上一节的结论来自**一张图、一次对照**。仓库里只有一对 (图, GT 深度) 配对，换图物理上做不到。
于是改用**剂量-反应**：把焦距按 k 倍缩放，权重/图/prompt/预处理**全部冻结**，扫 20 个 k。
若内参是支配原因，误差曲线必须在 k=1 处极小、两侧单调 —— 结果确实如此，但**三条曲线的极小不在同一处**：

| 分量 | 极小在 k= | fx | HFoV | 极小值 | 预测点 (k=0.3155) |
|---|---|---|---|---|---|
| **方位误差（与深度解耦）** | **1.0000** | 518.9 | 63.3° | **2.6 px** | **306.3 px** |
| 深度 ARel | 1.1000 | 570.7 | 58.6° | 7.5 % | 20.0 % |
| 3D 合成误差 | 1.1000 | 570.7 | 58.6° | 12.84 cm | 192.73 cm |

> ⚠ **最重要的是一条负面结果：不能用 3D 合成误差反推相机内参。**
> 它的极小跑到 k=1.1000（偏 10%），因为**深度头偏好更窄的视场**（训练分布），与内参真值无关。
> 拿它定容差会得出「把焦距故意调大 10% 反而更好」这种荒谬结论。**必须分解成横向与纵深两部分。**
> 各向异性因此有了量化形态：方位误差动态范围 **306.5 倍**，深度 ARel 只有 **9.8 倍**。

**四条支撑证据**（细节见方案文档 §22）：

1. **自洽性**：把模型预测的 K 显式喂回去 vs 不给内参，点云逐像素差中位 **1.642 mm**
   ⟹ 误差可完全归因到「那份 K 的值错了」，模型内部没有别的机制。
2. **方向场是解析可控的**（修正了原先的担心）：方位跨度 / (1/k) 在 20 个采样点
   （焦距差 **27 倍**）全部落在 **0.62–0.64%** ⟹ `rays` 就是针孔网格，
   焦距完全由传入的 K 设定，**横向尺度可以被精确设定**而非只能被动测量。
3. **焦距可独立反解，精度 ±1%**：±1% 网格 + 抛物线插值 → k̂=1.0168、fx̂=527.6，
   与 GT 518.9 只差 **1.68%**。⟹ 有 GT 深度就能**独立测出真实焦距**，可用来校验 EXIF。
4. **三视场复现**：数字变焦造出 s=1 / 1.5 / 2 三组 (图, GT K) 配对，
   A/B 倍数 **7.27× / 11.95× / 4.39×，全部 > 1**。
   顺带查明模型的相机头**读到视场但严重欠响应**（真值应放大 2.000×，它只放大 1.515×）。

### EXIF 内参路径：真实照片无 GT 内参时的唯一来源

新增 `vision/exif.py`（**零 torch**）+ `scripts/inspect_exif.py` + `--intrinsics exif`。
`f_px = f_35mm / 36 × max(W,H)` —— 用**长边**使结果对旋转不变，消掉 Orientation 记错就静默缩放 fx 那条路。
优先 `FocalLengthIn35mmFilm`，只有 `FocalLength` 时降级并标 `assumed_sensor=True`
（全画幅假设对手机差约 6.4 倍，但**骗不过 `check_fov`**，不会静默流到下游）。
读不到 EXIF 时返回 `None` 而非抛异常 —— 缺失是**正常分支**。

**实测**：仓库两张图 EXIF **全为空** ⟹ 无法用仓库素材验证，于是**自己造一张带真值的 fixture**
（从 GT K 反推 `f_35 = 518.9×36/640 = 29.19 → 29 mm → 回推 fx=515.56`）。
若只造「有 EXIF 的图」，验证就退化成「流水线跑通了」——什么都没证明。

端到端同图 A/B（唯一变量是内参）：

| 场景 | fx | HFoV | sofa 尺寸 |
|---|---|---|---|
| `--intrinsics exif` | 515.56 | 63.7° ✓ | **2.46 × 0.80 × 1.22 m** |
| 不给内参 | 159.17 | **127.1°** | **6.93 × 2.41 × 1.14 m**（警告正确触发） |

### ⭐ EXIF 的主要误差是主点，不是量化

EXIF 不记录主点，只能取图像中心。把误差拆开：

| 来源 | 方位 px | 占 | 3 m 处横向 | 占 50 mm 容差 |
|---|---|---|---|---|
| **主点中心假设** | **14.75** | **97.4%** | 85.3 mm | **171%** |
| 焦距量化 0.64% | 0.47 | 3.1% | 2.7 mm | 5% |
| （不给内参，二维口径） | 432.0 | — | 2458 mm | 远超 |

⟹ **主点是量化的 31.2 倍**。但两者都远小于「不给内参」的代价 ⟹ EXIF **依然值得做**，
定位要说准：**对米制尺寸与距离够用；对方向级精度受限于中心假设。**
主点误差近似线性（斜率 0.99–1.00）⟹ 可概括成「等效像素」一个数
（3 m 处 1 px ≈ 5.78 mm ⟹ 50 mm 容差 ≈ **8.6 px**）。

> ⚠ **口径警告**：方案文档 §21 报的 306.3 px 是**一维**（只含 x/z），本节的 432.0 px 是**二维**。
> 两个数都对，**引用时必须带口径**，否则看起来像矛盾。
> 这也解释了为什么前面完全没发现主点问题：纵向主点误差几乎全落在 **y 分量**，而旧指标只测 x。

---

## ⭐ 内参杠杆（三）：跨来源复跑**推翻了口径**（2026-09-16 深夜）

「全部证据来自同一张图」是之前列出的最大缺口。补它的结果是发现一个**隐藏混杂因子**：

> **§21、§22 的全部数字（预测 fx=163.7、HFoV 125.8°、横向放大 3.169×、三维误差 1.943 m）
> 都只在 `640×480` 这一档输入分辨率下成立。**

**同一张图、同一 4:3，只改输入像素数**（`probe_cross_source.py --part c`）：

| 输入 | resize | fx 预测 | HFoV 预测 | 预测/真值 | 可信 | 3D 误差(无 K) | 3D 误差(GT K) |
|---|---|---|---|---|---|---|---|
| 480×360 | 1.076 | 123.1 | 125.68° | 0.316 | ✗ | 1.960 m | 0.294 m |
| **640×480** | 1.000 | **163.7** | **125.80°** | **0.316** | ✗ | **1.943 m** | 0.267 m |
| 672×504 | 1.000 | 172.5 | 125.66° | 0.317 | ✗ | 1.968 m | 0.201 m |
| 704×528 | 1.000 | 346.9 | 90.84° | 0.608 | ✓ | 0.789 m | 0.166 m |
| 736×552 | 1.000 | 269.6 | 107.55° | 0.452 | ✓ | 1.252 m | 0.215 m |
| **768×576** | 1.000 | 702.5 | 57.32° | **1.128** | ✓ | **0.184 m** | 0.207 m |
| 800×600 | 1.000 | 741.7 | 56.68° | 1.144 | ✓ | 0.180 m | 0.211 m |
| 1600×1200 | 0.559 | 904.2 | 83.00° | 0.697 | ✓ | 0.644 m | 0.142 m |

**三条结论：**

1. **`640×480` 是反常工作点，而它正是 §21/§22 用的那一档。** 768 px 以上比值回到
   **1.12–1.14**（几乎正确），3D 误差掉到 **0.18 m** —— 与「给 GT K」的 0.21 m 已无实质差距。
2. **跳变不是 pipeline 造成的**（本节最硬的一段，用排除法）：640×480 与 800×600
   **都落在 `pixels_bounds=[200000,600000]` 内 ⟹ `resize_factor` 都是 1.0**，
   预处理逐字相同。而就在这段**预处理不变**的区间里，预测 fx 在
   172.5 → 346.9 → 269.6 → 702.5 之间**非单调摆动（4.07 倍）**。
   ⟹ 相机头的输出对输入尺度没有稳定性可言。
3. **`check_fov` 当不了护栏**：真实相机上只有 **40% 理想率**（5 台：2 正确 / 2 误报 / **1 漏报**）。
   漏报那例是 Olympus E-P3：MFT 画幅按全画幅假设算，fx 错 2 倍、HFoV 100.4°（真值 61.9°），
   **仍落在 30–110° 窗口内、零告警**。误报那两例是合规长焦（100 mm / 75 mm 等效）。

> ⚠ **结论方向不变，理由更强**：内参必须外部给定，**不是**因为模型「偏 3 倍」，
> 而是因为它的输出**没有稳定性**，且这件事**没有任何事后校验能发现**。
> 反过来：有外部 K 时，3D 误差跨 11 档分辨率的范围是 **0.139–0.294 m（2.1×）**，
> 而无 K 是 **0.177–1.968 m（11.1×）** —— **外部 K 把分辨率敏感性压掉 5.3 倍**。

**工程结论（已写进代码与命令示例）**：喂图**短边 ≥600 px（推荐 ≥768 px）**，
一行代码把无内参路径的 3D 误差从 1.94 m 压到 0.18 m；**但仍然要传 K**
（1600×1200 时无 K 是 0.644 m，是有 K 的 4.5 倍）。

**附带补上的两个缺口**：
- **EXIF 路径在真实相机 EXIF 上复跑**（5 台：Panasonic DMC-L10 / Olympus E-P3 / Ricoh GR /
  Sigma DP3 Merrill / Sony DSC-RX1R，画幅 1.0–2.0×）—— fx 独立重算与模块输出差 **0.000 px**，
  `FocalLength × 已知裁切系数` 与 EXIF 值一致到 0.00–1.96%。
  ⟹ 解析链路可靠，但**手机仍缺席**（而手机恰好是唯一会被 `check_fov` 兜住的一档）。
- **全项目第一张与 UniDepth demo 无关的真实照片**：`vendor/VADAR/demo-notebook/resources/demo.jpg`
  （1440×1920 竖幅、**另一个房间**），预测 HFoV 64.1°、深度中位 2.05 m —— 看起来完全合理。

> 完整表格、预处理轨迹、素材来源与可复现命令见方案文档 **§23**。
> 探针：`phase0/fetch_cross_source.py` + `phase0/probe_cross_source.py --part a|b|c`。

**网络**：❌ `huggingface.co` 超时（**必须** `HF_ENDPOINT=https://hf-mirror.com`）｜❌ `api.openai.com` 超时
｜✅ `download.pytorch.org` 实测 4.19 MB/s（最快）｜✅ `pypi.tuna.tsinghua.edu.cn`、`hf-mirror.com`
｜✅ `cdn.jsdelivr.net`（**可逐字节代理 GitHub 仓库文件 ⟹ 能拿到真实相机 EXIF 素材**）、
`gitee` / `gitcode` / Unsplash / Pexels / Pixabay CDN
｜❌ `upload.wikimedia.org`、`raw.githubusercontent.com` 超时｜⚠️ `github.com` git 端点间歇超时

---

## 目录结构

```
D:\3D_Spatial_Agent\
├── README.md                  ← 你在这里
├── docs\
│   ├── VADAR可借鉴性评估与路径选择.md        ★ 先读
│   └── 3D_Spatial_Agent_技术调研与实施方案.md / .html
├── vendor\
│   ├── VADAR\                 原始源码（HEAD 56018ebc），**一行不改**，只作参考
│   └── UniDepth\              git clone，editable 安装
├── venvs\vision\              Python 3.12.5
├── .cache\                    pip / wheels / huggingface / models（全部引到 D 盘）
├── phase0\                    Phase 0 可执行脚本与探针
│   ├── 01_setup_windows.ps1      建 venv + 装依赖 + 自检 + 出锁文件
│   ├── 01b_fetch_torch.ps1       curl 预下 torch 轮子（断点续传）
│   ├── 01c_fetch_gdino.ps1       curl 分段续传 GroundingDINO 权重
│   ├── verify_env.py             环境自检（CUDA / 缺件 / import UniDepthV2）
│   ├── probe3d.py                ★ 核心探针：A–D 四段端到端
│   ├── probe3d_result.json       机器可读实测结果
│   ├── requirements-vision.lock.txt  74 个固定版本
│   ├── 02_probe_llm_api.py       LLM 端点探针
│   ├── 03_vadar_llm_bridge.py    不改 VADAR 源码换 LLM 后端
│   ├── LLM_BACKEND.md            后端切换说明与 5 个坑
│   ├── probe_depth_gt.py         ★ 内参杠杆探针（GT 深度逐像素对照，见上文）
│   ├── probe_k_sweep.py          ★ 内参剂量-反应 + 数字变焦三视场复现（见「内参杠杆（二）」）
│   ├── probe_principal_point.py  ★ 主点 vs 量化的误差分解（发现主点才是 EXIF 的主要误差）
│   ├── make_exif_fixture.py      ★ 造带真值的 EXIF fixture（同时写 probe_exif_pipeline_report.txt）
│   ├── fetch_cross_source.py     ★ 跨来源素材抓取（真实相机 EXIF + 真实场景，见「内参杠杆（三）」）
│   ├── probe_cross_source.py     ★ 跨来源复跑：A=真实 EXIF / B=相机头行为 / C=分辨率剂量-反应
│   └── 00_install_wsl.ps1 / 01_setup_ubuntu.sh   旧 WSL 路线，保留作对照
├── vision\                    ★ L1 感知层（三模型的原生包装）
│   ├── types.py               Detection / DepthField / PerceptionLike 协议  ← 零 torch
│   ├── geometry.py            掩码质心 / 稳健尺寸 / 重力方向 / 视场检查      ← 零 torch
│   ├── exif.py                ★ EXIF → 内参（长边约定 + 量化误差随 K 一起记录）← 零 torch
│   ├── grounding.py           GroundingDINO（transformers 原生）
│   ├── segmentation.py        SAM2（transformers 原生，强制批量）
│   ├── depth.py               UniDepth V2 → points / depth / K（含内参覆盖规则）
│   └── registry.py            懒加载 + 显存记账 + 卸载
├── tools\                     ★ 暴露给 LLM 的 API（契约见方案文档 §13.3）
│   ├── result.py              ToolResult 信封 + 6 个错误码 + 恢复动作枚举
│   ├── version.py             TOOLS_VERSION / RELATIONS_METHOD（版本化单一来源）
│   ├── registry.py            @tool 装饰器 + ToolContext + trace + 能力开关
│   ├── guards.py              NOT_IN_SCENE（幻觉捕获点）/ NOT_FOUND
│   ├── geometry.py            get_3d_position / get_3d_extent / calculate_distance / calculate_angle
│   ├── spatial.py             list_objects / find_object / find_nearest / query_relation …
│   └── build_doc_html.py      Markdown → HTML（构建脚本，不属于工具库）
├── scene_graph\               ★ 3D 场景图中间表示
│   ├── schema.py              Node / Edge / SceneGraph（坐标系约定写在文件头）
│   ├── relations.py           纯几何关系 + UpAxis 归一（零依赖、可脱离场景图单测）
│   ├── builder.py             ★ 单遍构建：升维→检测→去重→分割→节点→重力→关系
│   ├── store.py               落盘（格式版本信封）+ 掩码 1-bit PNG 往返
│   └── tests\                 test_relations.py / test_builder.py / test_store.py
├── tests\                     test_tools.py / test_vision_geometry.py / test_vision_depth.py / test_vision_exif.py
├── scripts\
│   ├── smoke_tools.py         端到端冒烟（不需要 GPU、不需要联网）
│   ├── build_scene.py         ★ 真实跑通入口（加载三模型 → scene.json + masks + build_log）
│   ├── inspect_scene.py       场景图体检：内参/视场 → 掩码泄漏 → 点云离散 → 尺寸越界
│   └── inspect_exif.py        ★ 批量体检图片 EXIF 可用性（不加载模型、不需要 GPU）
├── dataset\scenes\            场景图 JSON 缓存
│                              living_room_gt / living_room_pred 是同图内参 A/B；
│                              exif_fixture_exif / exif_fixture_pred 是 EXIF 路径同图 A/B
└── logs\
```

---

## 常用命令（Windows 原生）

```powershell
# 全部单元测试（193 用例，约 2.8 s，不需要 GPU 与联网）
cd D:\3D_Spatial_Agent
& D:\3D_Spatial_Agent\venvs\vision\Scripts\python.exe -m pytest -q

# 端到端冒烟：场景图 → 工具 → 答案 + 证据链 + trace（不需要 GPU 与联网）
& D:\3D_Spatial_Agent\venvs\vision\Scripts\python.exe D:\3D_Spatial_Agent\scripts\smoke_tools.py

# 先看看手上的照片有没有可用的 EXIF（不加载任何模型、不需要 GPU）
& D:\3D_Spatial_Agent\venvs\vision\Scripts\python.exe D:\3D_Spatial_Agent\scripts\inspect_exif.py `
    D:\photos\*.jpg --raw

# ★ 真实照片建场景图（加载三模型，约 2 s/图 + 13 s 冷启动）
#   --intrinsics auto：sidecar npy → EXIF → 模型预测，逐级降级；强烈建议提供
#   ⚠ 图片短边 ≥600 px（推荐 ≥768 px）—— 见下文「内参杠杆（三）」的分辨率表
& D:\3D_Spatial_Agent\venvs\vision\Scripts\python.exe D:\3D_Spatial_Agent\scripts\build_scene.py `
    --image D:\3D_Spatial_Agent\vendor\UniDepth\assets\demo\rgb.png `
    --scene-id living_room_gt --intrinsics auto `
    --prompt "sofa. chair. table. picture. mirror."

# 场景图体检（不加载任何模型、不需要 GPU）
& D:\3D_Spatial_Agent\venvs\vision\Scripts\python.exe D:\3D_Spatial_Agent\scripts\inspect_scene.py `
    --scene D:\3D_Spatial_Agent\dataset\scenes\living_room_gt --scene-id living_room_gt

# 内参杠杆的量化证据（需要 GPU，约 30 s）
& D:\3D_Spatial_Agent\venvs\vision\Scripts\python.exe D:\3D_Spatial_Agent\phase0\probe_depth_gt.py

# 内参的剂量-反应曲线 + 数字变焦三视场复现（需要 GPU，约 5 min）
& D:\3D_Spatial_Agent\venvs\vision\Scripts\python.exe D:\3D_Spatial_Agent\phase0\probe_k_sweep.py

# 造一张带真值的 EXIF fixture（写 .cache\exif_fixture\rgb_exif.jpg）并核对读数
& D:\3D_Spatial_Agent\venvs\vision\Scripts\python.exe D:\3D_Spatial_Agent\phase0\make_exif_fixture.py

# 主点假设的代价（需要 GPU，约 4 min）
& D:\3D_Spatial_Agent\venvs\vision\Scripts\python.exe D:\3D_Spatial_Agent\phase0\probe_principal_point.py

# ★ 跨来源复跑（方案文档 §23）。先抓素材，再跑三段
& D:\3D_Spatial_Agent\venvs\vision\Scripts\python.exe D:\3D_Spatial_Agent\phase0\fetch_cross_source.py
& D:\3D_Spatial_Agent\venvs\vision\Scripts\python.exe D:\3D_Spatial_Agent\phase0\probe_cross_source.py --part a   # 纯 CPU，秒级
& D:\3D_Spatial_Agent\venvs\vision\Scripts\python.exe D:\3D_Spatial_Agent\phase0\probe_cross_source.py --part b   # GPU，约 20 s
& D:\3D_Spatial_Agent\venvs\vision\Scripts\python.exe D:\3D_Spatial_Agent\phase0\probe_cross_source.py --part c   # GPU，约 20 s（决定性证据）

# 环境自检
& D:\3D_Spatial_Agent\venvs\vision\Scripts\python.exe D:\3D_Spatial_Agent\phase0\verify_env.py

# 核心探针（A–D 四段：UniDepth → points 自洽 → GroundingDINO → 三维距离）
& D:\3D_Spatial_Agent\venvs\vision\Scripts\python.exe D:\3D_Spatial_Agent\phase0\probe3d.py

# 改完方案文档后重建 HTML
D:\Users\ROG\anaconda3\python.exe D:\3D_Spatial_Agent\tools\build_doc_html.py `
    "D:\3D_Spatial_Agent\docs\3D_Spatial_Agent_技术调研与实施方案.md"
```

---

## 本机环境金律（每次都踩，先看这里）

| 现象 | 对策 |
|---|---|
| **单条命令约 121 秒被杀**（前台后台一视同仁，日志无 ERROR） | 每条命令 ≤110 s；安装/下载一律拆片 |
| pip 下大文件静默卡死（2.5 GB 轮子 18 min 0 字节） | 大文件用 **`curl -C -`**，别交给 pip |
| 大文件下载中途被杀 | `curl -C - --max-time 100` 分轮续传 |
| **`.ps1` 报「命令未找到」** | PowerShell **没有 `B` 数值后缀**（只有 KB/MB/GB/TB）—— `MinBytes = 100B` 会被当成命令名。小文件阈值写**纯字节数** |
| `.ps1` 并非"无法执行"（**旧结论已推翻**） | `powershell -NoProfile -ExecutionPolicy Bypass -File x.ps1` 可正常执行，exit code 与 stderr 都读得到。此前「6 秒静默退出」的真因就是上一条的语法错 |
| `curl -C -` 对已完整文件返回 **http=416** | 加分支识别，否则白跑满重试次数 |
| PowerShell stdout 不回传；bash 不可用 | 一律「命令写文件 → Read 读文件」；日志中文乱码时先设 `[Console]::OutputEncoding = [Text.Encoding]::UTF8`，并给子进程加 `PYTHONIOENCODING=utf-8`。**更好的办法是让 Python 脚本自己写报告文件**（UTF-8 由 Python 控制，绕开整条控制台编码链）—— `probe_depth_gt.py` / `build_log.txt` 就是这么做的 |
| **`*>` 会让成功的命令返回 exit=1** | PowerShell 把子进程 stderr（timm/xformers 的 warning）当成 NativeCommandError。判断成败要看脚本自己写的产物，不要只看 exit code |
| **CUDA 计时不用 `synchronize` 会差一个量级** | 实测同一模型：同步 47 ms、不同步 28 ms，而不同步时「第二条路径」总显得更快 —— 那是 kernel 排队顺序造出来的假结论。测 GPU 一律 `torch.cuda.synchronize()` 夹住，并先跑一次丢弃的预热 |
| HF 下载慢/下两份等价权重 | `HF_HUB_DISABLE_XET=1` + `allow_patterns=["config.json","*.safetensors"]` |
| **跑 GPU 探针时忘了设 HF 变量 ⟹ `httpx.ProxyError: 502`** | `HF_HOME` / `HF_ENDPOINT` / `HF_HUB_DISABLE_XET` 必须在 **`import torch` 之前**设好（`huggingface_hub` 在 import 期就读它们）。权重已缓存也一样会去连 |
| **结论被一个没注意到的输入维度混杂** | 本轮实例：`640×480` 是 UniDepth 相机头的反常工作点，而全部旧结论都取自它。**换来源前先扫一遍「输入尺度」这个自变量**，别只换内容 |
| **想证明「跳变不是 pipeline 造成的」** | 不能只靠推理：**把 pipeline 的预处理轨迹打出来**（`probe_cross_source.preprocess_trace()` 打印 padding / resize_factor / 网络实际输入尺寸）。否则读者有理由怀疑是分支 |
| **按字节原样取 GitHub 上的测试素材** | `raw.githubusercontent.com` 超时，但 **`cdn.jsdelivr.net/gh/<owner>/<repo>@<ref>/<path>` 可用**（逐字节代理，不重编码 ⟹ EXIF 保真）。`data.jsdelivr.com/v1/packages/gh/<owner>/<repo>@<ref>?structure=flat` 可列文件 |
| **想造「真实相机 EXIF」素材** | 别自己合成（合成的证明不了真实标签分布）：`hMatoba/Piexif` 的 `tests/images/r_*.jpg` 是 5 台真实相机的原始 EXIF。注意**文件名不可信**（`r_pen.jpg` 里装的是 **Olympus E-P3**，`r_pana.jpg` 是 DMC-L10），一律以 EXIF 里的 `Make`/`Model` 为准 |
| **方案文档改了但 HTML 像是只渲染了一半** | Markdown 里的字面量尖括号（如写「i 小于 j」时用了 `<j`）会被当成**未闭合标签**，**吞掉文档剩余部分** —— 症状是正文在中途断掉、后面的章节和附录整块消失。改完务必重建 HTML **并 grep 一次末尾章节标题**；正文里不要用尖括号写小于号 |
| **造验证素材时「有 EXIF」不等于「有真值」** | 若只造一张「带 EXIF 的图」，验证就退化成「流水线跑通了」——什么都没证明。必须从 GT 内参**反推出该写进 EXIF 的值**，于是素材自带正确答案与已知偏差；否则换算写错宽高（偏差会从 0.64% 变成 33%）在单测之外看不出来 |
| **裁剪/变焦类操作必须同步裁剪真值** | 数字变焦时忘了把 GT 深度一起裁到同一窗口，A/B 对比被一个假常数污染，结论全错 |
| **单调性检查写反方向** | `diff(lo) >= 0` 会让严格递减的曲线报「不单调」→ 断言直接失效。正确是 `<= 0`（先降后升）；并**把原始数组打出来核对**，别只信布尔量 |
| **相对阈值不要建立在近零基准上** | 基准 2.6 px 已接近模型自身误差下限，在近零分母上算「涨 X%」会得出虚高的 ±0.5%，读起来像苛刻工程要求，其实只是小分母假象。**改用绝对门槛**（5/10/25/50 px） |
| **一维与二维指标不能混引** | 同一情形：一维（只含 x/z）306.3 px vs 二维 432.0 px。两个都对，但**引用必须带口径**，否则「同一个量」在两处数字不同，看起来像自相矛盾。（纵向误差几乎全落在 y 分量 —— 用标量概括二维量必有盲区） |

---

## 方法论约定（贯穿全项目）

1. **空间关系必须由几何计算得出**，不由 LLM/VLM 判断。VLM 只用于颜色/材质等语义属性。
2. **一切数字必须实测**。方案文档里每个估算值都标了「需实测替换」，实测值才算数。
3. **不编造 VADAR 中不存在的函数**。所有代码引用都带真实文件行号。
4. **`vendor/VADAR/` 一行不改**，所有改动通过适配器 + 补丁记录，保证实验臂 A 随时可跑。
5. **创新点必须能出定量实验**。「换了个 LLM」不算创新点。

> **许可提示**：UniDepth 与 Omni3D-Bench 均为 **CC BY-NC 4.0**，本项目**不可商用**；报告与答辩中须注明。
