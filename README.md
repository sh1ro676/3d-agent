# 3D Spatial Agent

自建的三维空间智能体项目：**感知层**接三个开源模型（GroundingDINO + SAM2 + UniDepth），
**Agent 层与工具层全部自研**。
课程：3D 视觉 / 三维视觉算法。用途：课程大作业 + 答辩。

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

1. `docs/3D_Spatial_Agent_技术调研与实施方案.md`（主方案，20 余章；`.html` 为阅读版）

**凡是估算值都显式标注「估算」，实测值才算数。**

---

## 当前进度

| Phase | 状态 |
|---|---|
| **0 环境** | ✅ 视觉栈 **Windows 原生**跑通，未装 WSL |
| 0.x 核心验证 | ✅ `probe3d.py` A–D 四段全 PASS；`probe_sam2.py` 补完最后一批估算值；`probe_depth_gt.py` 量化内参杠杆 |
| **1b 自建骨架** | ✅（2026-09-16）信封 + 场景图 + 工具库 |
| **1c 感知层 + builder** | ✅（2026-09-16）`vision/` 八个模块 + `builder.py` + `store.py`；**零 GPU 可单测**，已产出真实照片的场景图 |
| **1c.2 内参因果证据 + EXIF** | ✅（2026-09-16 晚）剂量-反应扫描 + 数字变焦复现 + **EXIF 内参路径端到端跑通**；见下文「内参杠杆（二）」 |
| **1c.3 跨来源复跑** | ✅（2026-09-16 深夜）10 张图 / 4 个来源 / 5 台真实相机 EXIF。**发现了一个隐藏混杂因子**：`640×480` 是相机头的一个反常工作点，§21/§22 的数字都取自它。见下文「内参杠杆（三）」 |
| **L5 场景级报告层** | ✅ `describe_scene` / `summarize_scene` / `diagnose_failure` / `counterfactual`（分析工具，不在问答动作空间内） |
| **自研 Agent** | ✅ `agents/`：程序合成 + 执行 + 记忆 + 前置规划 + **答案证据校验** |
| **前端演示台** | ✅ `demo/` 四区界面 + 四种布局 + 现场真跑 `/api/ask` + 反事实 + 上传建图 |
| 4 显存工程 / 8 数据集合成 / 9 QLoRA / 10 评估 / 12 完整实验 / 13 答辩 Demo | ⬜ |

**下一步**：接外部给定的对照方法、补 `ablation.py` / `stats.py`、进入完整实验（路线全图见方案文档 §18）。
**手机 EXIF 仍是缺口**（5 台真机画幅覆盖 **1.0–2.0×**：35 mm ×1、APS-C ×2、4/3 与 MFT ×2；
最小的那台 MFT **仍然逃得掉 `check_fov`**，所以不是「传感器太大才漏」的巧合），
以及「跨来源 + GT 内参」的配对 —— 原以为 Omni3D-Bench 能补上，**实测它没有 GT 相机/深度/三维框**，该基准上无法验证内参杠杆，需另找带标定的数据源（方案文档 §23.5）。

> 已写进代码的两条硬约束：① `get_3d_position` **必须用 SAM2 掩码质心**，不能用检测框内中位数 ——
> 两者相差**均值 83 mm / 最大 208 mm**，而关系判断的容差是 50 mm。
> ② **内参必须作为输入传进来**，有就一定传 —— 它决定全部横向米制尺度（见下文）。
> 这两条都不是注释，而是 `Node.centroid_3d` 的语义与 `BuildConfig.known_intrinsics` 的存在理由。

---

> **全程 Windows 原生，不需要 WSL2。** 整条视觉栈（GroundingDINO + SAM2 + UniDepth）已在 Windows 上
> 实测跑通；延迟与峰值显存见下。

---

## 环境事实（2026-09-16 实测）

**参考硬件**：单卡 RTX 4060 Laptop，显存上限 **8188 MiB**（下文显存数字均在这一档实测）｜CUDA 12.6
**环境**：`venvs\vision`（Python 3.12.5）｜torch 2.6.0+cu124｜torchvision 0.21.0+cu124
**国内网络**：`huggingface.co` 不可达 ⟹ 需设 `HF_ENDPOINT=https://hf-mirror.com`；
`raw.githubusercontent.com` 超时时，可用 `cdn.jsdelivr.net/gh/<owner>/<repo>@<ref>/<path>`
逐字节取文件（**不重编码 ⟹ 真实相机 EXIF 保真**）。

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
→ **完整 XYZ 已经在手里，`depth` 只是它的 z 列** —— 升级到真三维不需要任何新模型、任何新显存。

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
- **全项目第一张与 UniDepth demo 无关的真实照片**（1440×1920 竖幅、**另一个房间**）：
  预测 HFoV 64.1°、深度中位 2.05 m —— 看起来完全合理。
  ⚠ 该素材随早期基线检出一起移出本仓库，**结论保留但当前不可复现**。

> 完整表格、预处理轨迹、素材来源与可复现命令见方案文档 **§23**。
> 探针：`phase0/fetch_cross_source.py` + `phase0/probe_cross_source.py --part a|b|c`。

---

## 目录结构

```
3d-agent\
├── README.md                  ← 你在这里
├── docs\
│   └── 3D_Spatial_Agent_技术调研与实施方案.md / .html       ★ 主方案
├── configs\
│   └── llm_backend.env.template   LLM 后端配置模板（含真 key 的 llm_backend.env 不入库）
├── llm_env.py                 .env 加载器：解析 / 冲突检测 / 密钥掩码
├── vendor\
│   └── UniDepth\              git clone，editable 安装（VENDOR.md 记固定 HEAD 与获取命令）
├── venvs\vision\              本地虚拟环境（不入库，按文档自建）
├── .cache\                    模型权重与 pip 缓存（不入库）
├── phase0\                    Phase 0 可执行脚本与探针
│   ├── 01_setup_windows.ps1      建 venv + 装依赖 + 自检 + 出锁文件
│   ├── 01b/01c/01d_fetch_*.ps1   curl 预下 torch 轮子 / GroundingDINO / SAM2 权重
│   ├── 04_install_ollama.ps1     可选的本地 LLM 后端
│   ├── verify_env.py             环境自检（CUDA / 缺件 / import UniDepthV2）
│   ├── probe3d.py                ★ 核心探针：A–D 四段端到端
│   ├── probe3d_result.json       机器可读实测结果
│   ├── probe_sam2.py             批量 vs 逐个的调用形态对照（7.57×）
│   ├── requirements-vision.lock.txt  74 个固定版本
│   ├── probe_depth_gt.py         ★ 内参杠杆探针（GT 深度逐像素对照，见上文）
│   ├── probe_k_sweep.py          ★ 内参剂量-反应 + 数字变焦三视场复现（见「内参杠杆（二）」）
│   ├── probe_principal_point.py  ★ 主点 vs 量化的误差分解（发现主点才是 EXIF 的主要误差）
│   ├── make_exif_fixture.py      ★ 造带真值的 EXIF fixture（同时写 probe_exif_pipeline_report.txt）
│   ├── fetch_cross_source.py     ★ 跨来源素材抓取（真实相机 EXIF + 真实场景，见「内参杠杆（三）」）
│   ├── probe_cross_source.py     ★ 跨来源复跑：A=真实 EXIF / B=相机头行为 / C=分辨率剂量-反应
│   └── probe_intrinsics.py       内参口径对照（各探针的 *_report.txt / *_result.json 同目录）
├── vision\                    ★ L1 感知层（三模型的原生包装）
│   ├── types.py               Detection / DepthField / PerceptionLike 协议  ← 零 torch
│   ├── geometry.py            掩码质心 / 稳健尺寸 / 重力方向 / 视场检查      ← 零 torch
│   ├── exif.py                ★ EXIF → 内参（长边约定 + 量化误差随 K 一起记录）← 零 torch
│   ├── semantics.py           颜色 / 材质判定（VLM 侧语义的落点）            ← 零 torch
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
│   ├── attributes.py          get_attributes（关掉 VLM 时返回 CAPABILITY_DISABLED）
│   ├── scene_report.py        describe_scene / summarize_scene / diagnose_failure / counterfactual
│   └── build_doc_html.py      Markdown → HTML（构建脚本，不属于工具库）
├── scene_graph\               ★ 3D 场景图中间表示
│   ├── schema.py              Node / Edge / SceneGraph（坐标系约定写在文件头）
│   ├── relations.py           纯几何关系 + UpAxis 归一（零依赖、可脱离场景图单测）
│   ├── builder.py             ★ 单遍构建：升维→检测→去重→分割→节点→重力→关系
│   ├── store.py               落盘（格式版本信封）+ 掩码 1-bit PNG 往返
│   └── tests\                 test_relations.py / test_builder.py / test_store.py
├── llm\                       LLM 后端适配层
│   ├── adapter.py             OpenAI 兼容客户端（环境变量键名表 `ENV_ALIASES`）
│   ├── render.py              工具文档 → 提示词（只取 docstring 第一段）
│   ├── schema.py              程序与提交的结构化输出约束
│   └── vlm.py                 视觉语义的小模型封装
├── agents\                    ★ 自研 Agent
│   ├── synthesizer.py         单轮程序合成（主路径，1 次 LLM 调用）
│   ├── executor.py            12 个问答工具的显式白名单 + 静态检查 + 子进程执行 + 超时强杀
│   ├── verifier.py            答案证据校验器（supported / weak / unsupported / abstained）
│   ├── planner.py             前置规划臂（只改上下文，不改控制流）
│   ├── memory.py              多轮轨迹记忆
│   ├── loop.py                多轮 tool-calling（消融臂 E′）
│   └── prompts\
├── evaluation\                评估口径层（阈值与分母口径写在 `metrics.py` 里）
├── tests\                     跨模块单测（零 GPU、不需要联网）
├── scripts\
│   ├── build_scene.py         ★ 真实跑通入口（加载三模型 → scene.json + masks + build_log）
│   ├── run_agent.py           ★ Agent 端到端入口（程序合成 → 执行 → 证据校验）
│   ├── serve_demo.py          起前端演示台
│   ├── smoke_tools.py         端到端冒烟（不需要 GPU、不需要联网）
│   ├── inspect_scene.py       场景图体检：内参/视场 → 掩码泄漏 → 点云离散 → 尺寸越界
│   ├── inspect_exif.py        ★ 批量体检图片 EXIF 可用性（不加载模型、不需要 GPU）
│   ├── mine_glue_patterns.py  ★ 从已落盘的真跑程序里挖重复形态（轨迹驱动的算子挖掘）
│   └── …                      report_scene / export_demo / verify_points / probe_combination …
├── dataset\                   数据集与场景图缓存
│   ├── scenes\                场景图 JSON（living_room_gt / _pred 是同图内参 A/B；
│   │                          exif_fixture_exif / _pred 是 EXIF 路径同图 A/B）
│   └── builders\              Omni3D-Bench 等数据集的抓取与读取
├── demo\                      前端演示台（`?layout=` 四种布局，静态页 + 现场真跑）
├── reports\                   实验产物与分析报告 ← **判断「当前是哪一版」只看 `reports/README.md`，不认 mtime**
├── results\                   实验臂真跑产物
└── logs\                      LLM 调用日志
```

---

## 环境准备与常用命令（Windows 原生）

**首次使用三步**：

1. 按 **`vendor/VENDOR.md`** 拉回第三方检出（`vendor/UniDepth` 本身是 git clone，
   不入库，该文件记录了固定 HEAD 与重新获取命令）；
2. 跑 **`phase0/01_setup_windows.ps1`** —— 建 venv + 装依赖 + 自检 + 出锁文件；
3. 依赖版本固定在 `phase0/requirements-vision.lock.txt`，环境须为 Python 3.12。

以下命令请先 `cd` 到仓库根目录再执行。

```powershell
# 解释器简写（下文一律用 $PY）
$PY = "venvs\vision\Scripts\python.exe"

# 全部单元测试（不需要 GPU 与联网）
# ⚠ 必须从仓库根目录运行 —— 用例分布在 tests\、scene_graph\tests\、evaluation\tests\ 三处
& $PY -m pytest -q

# 端到端冒烟：场景图 → 工具 → 答案 + 证据链 + trace（不需要 GPU 与联网）
& $PY scripts\smoke_tools.py

# ★ 自研 Agent 端到端（按成本从低到高，先看提示词再花钱）
& $PY scripts\run_agent.py --scene living_room --question "哪把椅子离门最近？" --dry-run
& $PY scripts\run_agent.py --scene living_room --program-file my_prog.py
& $PY scripts\run_agent.py --scene living_room --question "哪把椅子离门最近？"

# ★ 前端演示台（默认 http://127.0.0.1:8770）
& $PY scripts\serve_demo.py

# 先看看手上的照片有没有可用的 EXIF（不加载任何模型、不需要 GPU）
& $PY scripts\inspect_exif.py `
    <你的照片目录>\*.jpg --raw

# ★ 真实照片建场景图（加载三模型，约 2 s/图 + 13 s 冷启动）
#   --intrinsics auto：sidecar npy → EXIF → 模型预测，逐级降级；强烈建议提供
#   ⚠ 图片短边 ≥600 px（推荐 ≥768 px）—— 见下文「内参杠杆（三）」的分辨率表
& $PY scripts\build_scene.py `
    --image vendor\UniDepth\assets\demo\rgb.png `
    --scene-id living_room_gt --intrinsics auto `
    --prompt "sofa. chair. table. picture. mirror."

# 场景图体检（不加载任何模型、不需要 GPU）
& $PY scripts\inspect_scene.py `
    --scene dataset\scenes\living_room_gt --scene-id living_room_gt

# 内参杠杆的量化证据（需要 GPU，约 30 s）
& $PY phase0\probe_depth_gt.py

# 内参的剂量-反应曲线 + 数字变焦三视场复现（需要 GPU，约 5 min）
& $PY phase0\probe_k_sweep.py

# 造一张带真值的 EXIF fixture（写 .cache\exif_fixture\rgb_exif.jpg）并核对读数
& $PY phase0\make_exif_fixture.py

# 主点假设的代价（需要 GPU，约 4 min）
& $PY phase0\probe_principal_point.py

# ★ 跨来源复跑（方案文档 §23）。先抓素材，再跑三段
& $PY phase0\fetch_cross_source.py
& $PY phase0\probe_cross_source.py --part a   # 纯 CPU，秒级
& $PY phase0\probe_cross_source.py --part b   # GPU，约 20 s
& $PY phase0\probe_cross_source.py --part c   # GPU，约 20 s（决定性证据）

# 环境自检
& $PY phase0\verify_env.py

# 核心探针（A–D 四段：UniDepth → points 自洽 → GroundingDINO → 三维距离）
& $PY phase0\probe3d.py

# 改完方案文档后重建 HTML
python tools\build_doc_html.py `
    "docs\3D_Spatial_Agent_技术调研与实施方案.md"
```

---

## 方法论约定（贯穿全项目）

1. **空间关系必须由几何计算得出**，不由 LLM/VLM 判断。VLM 只用于颜色/材质等语义属性。
2. **一切数字必须实测**。方案文档里每个估算值都标了「需实测替换」，实测值才算数。
3. **不编造第三方库里不存在的函数**。引用外部代码一律带真实文件行号；若该检出已移出仓库，
   改用行为描述而不是行号指针。
4. **`vendor/` 下的第三方检出一行不改**，自研侧只通过适配层调用它。
5. **创新点必须能出定量实验**。「换了个 LLM」不算创新点。

> **许可提示**：UniDepth 与 Omni3D-Bench 均为 **CC BY-NC 4.0**，本项目**不可商用**；报告与答辩中须注明。
