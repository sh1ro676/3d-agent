# `evaluation/` —— 实验臂运行与评估

这个目录回答一个问题：**「跑一个实验臂」到底要跑什么。**
它把 `vendor/VADAR` 的四段流水线接上本项目的基础设施，
并且把 VADAR 原版缺的三样东西补上：可复现的题集、归一化的结果、不致命的失败处理。

---

## 1. 先跑 `--plan`（不需要 API key、不加载模型）

```bash
cd D:/3D_Spatial_Agent
venvs/vision/Scripts/python.exe evaluation/runner.py --plan --num-questions 20 --num-api-questions 3
```

它做四件事，每一件都能立刻暴露配置错误：

| 步骤 | 验什么 | 典型失败 |
|---|---|---|
| 前检查 | `annotations.json` 的字段是否是 VADAR 硬约定的那 6 个；`images/` 里图片是否齐 | `question_index` 写成 `q_index` 没映射 → `KeyError` 在很晚才炸 |
| 选子集 | 题集是否确定、四类指标各有多少题、覆盖几张图 | 取前 20 题只有 **1 道 numeric_count** —— 小样本偏差必须看得见 |
| 装适配层 | 4 个命名空间是否都替换成桥接的 `Generator`；Windows alarm 是否就位；模型指纹 | 漏一个命名空间 → 一部分代码还在走 OpenAI，静默 |
| 模型指纹 | `base_url + model + temperature + max_tokens + extra_body` | 少记一项，换模型后的对比无法归因（§16.2） |

`--plan` 的产物是 `results/<arm>/latest_plan.json` 与 `results/<arm>/subset.json`。
**`subset.json` 是后续所有臂复现同一批题的依据**，不要删。

---

## 2. 真跑

### 2.1 唯一的手工步骤：把 key 填进 `configs/llm_backend.env`

```bash
notepad "D:\3D_Spatial_Agent\configs\llm_backend.env"     # 或用任意编辑器
```

`VADAR_API_KEY=` 就在里面（当前是**第 30 行**；拿不准就别数 ——
运行器缺 key 时的报错会直接给出「把 key 填进 `<路径>` 的第 N 行」）。
在 `=` 后面填值，**不要加引号**：

```ini
VADAR_API_KEY=sk-0123456789abcdef
```

为什么要有这么一个文件：`phase0/06_deepseek_setup.ps1` 设的是**会话级**环境
变量 —— 只在那一个 PowerShell 窗口里有效、**不落盘**、也传不进任何别的进程。
2026-09-17 实跑 arm A 就卡在这里：运行器完整、`--plan` 全绿，真跑时读不到 key，
`return 2`。配置落到文件之后，「这次实验用的是哪套后端」变成一行可检查、
可归档的事实，而不是「某个窗口现在还开着吗」。

**加载顺序**（后面的只补缺、不覆盖前面的）：

```
configs/llm_backend.env  →  进程环境变量  →  代码里的默认值
```

四条语义 —— 每一条都是为了让**读错配置这件事必须浮出来**（完整版见仓库根
`vadar_env.py` 的模块注释）：

| 语义 | 为什么这么定 |
|---|---|
| **文件优先于进程环境**，冲突记进 `llm_env.conflicts` | 文件是你刚编辑过的；进程环境可能是几天前忘掉的残留。两者不同必须可见 |
| **空值视同未设** | 模板里 `VADAR_API_KEY=` 是常态，它不该顶掉一个真实存在的环境变量 |
| **`#` 只在行首（可前置空格）才是注释**，不支持行尾注释 | 值里带 `#` 时，行尾注释会**静默截断**配置 |
| **同一个键写两次直接报错** | 改配置忘了注释旧行 = 静默用错值，这是最难查的一类 |

`--env-file <路径>` 换一个文件（换文件时上一份独有的键会被撤销回原值）；
`--no-env-file` 完全不用文件。

产物里会留两份归因材料：`llm_env`（脱敏快照：路径 / 键数 / `config_sha256` /
每个密钥的 8 位指纹）与 `api_key_source`（`env_file` / `process_env` / `missing`）。
**密钥明文永远不会出现在任何产物里** —— 这由 9 条测试守着（见 §6）。

### 2.2 跑

```bash
# 小样本 smoke（2 题，消耗极小）
venvs/vision/Scripts/python.exe evaluation/runner.py --arm A --num-questions 2 --num-api-questions 2

# 全量（501 题）
venvs/vision/Scripts/python.exe evaluation/runner.py --arm A --num-questions 501 --num-api-questions 10
```

产物：

```
results/A/
├── subset.json           ← 本次用到的题目清单（含四类指标计数）★复现依据
├── latest_run.json       ← 最近一次**真进过流水线**的 run（模型指纹/环境/指标/成本）
├── latest_plan.json      ← 最近一次 --plan（里面有 next_command）
├── latest_failed.json    ← 最近一次提前退出（前检查失败 / 选不出题 / 缺 key）
├── <时间戳>/             ← VADAR 原生产物
│   ├── signature_generator/signatures.json
│   ├── api_generator/
│   ├── program_generator/
│   ├── program_execution/execution.json  execution.csv  results.txt
│   └── arm_report.json
└── ...
```

**这三份 json 不要合并回一个 `latest_run.json`** —— 理由见 §3.4。

---

## 3. 三个必须知道的设计决定

### 3.1 相对原版的**唯一**偏离：题集选择改成确定性

原版 `evaluate.py:42`：

```python
api_questions = random.sample(questions, args.num_api_questions)   # 无种子
```

同数据两次跑会抽到不同题 → 数字无法归因。这是 bug ③，
也是「原版论文结果不可复现」的直接原因之一。

本运行器改用 `random.Random(seed).sample(...)`，并把**被选中的清单落盘**。
`questions` 本身的选取默认仍是原版的 `questions[:N]`（`--select vadarspec`）。

三种模式：

| `--select` | 行为 | 什么时候用 |
|---|---|---|
| `vadarspec`（默认） | `questions[:N]`，原版行为 | 要和原版对话、追求保真 |
| `seeded-sample` | 种子化均匀抽样 | 需要跨图片覆盖 |
| `stratify` | 按四类指标的实际占比配比 | 子集 < 40 题时，避免某类为 0 → 指标为 `None` 被误读成 0 分 |

> **为什么 `stratify` 存在**：`--plan` 实测取前 20 题只有 1 道 `numeric_count`。
> 子集太小时分项指标会大面积失真，而这件事**不会报错**，只会让报告里的
> 某个格子变成 `None` 或一个基于 1 道题的百分比。

### 3.2 VADAR 的 bug 默认**保留**，用开关隔离

`SignatureAgent.__init__` 走的是 `Agent.__init__(model_name, write_results)`
两参数版本，`dataset` 拿到默认值 `"clevr"`，于是
`get_signatures` 永远用 `SIGNATURE_PROMPT_CLEVR`（`agents.py:123-126`）——
即使你传的是 Omni3D 的 signatures（bug ①）。

臂 A 的定义是「VADAR 原始流水线」，所以**默认复现这个 bug**：

```bash
--fix-signature-prompt     # 打开它，把 dataset 改成真实值
```

这样 bug 的代价就变成一个可量化的数字（同一批题跑两次的差），
而不是一段口说无凭的「我们认为这里有 bug」。

### 3.3 Windows 上的 `signal.SIGALRM`

VADAR 在 `agents.py:572` 与 `engine.py:594` 用 `signal.alarm` 设执行超时，
Windows 没有 `SIGALRM`。`win_alarm.py` 用两条互补路径替代：

1. **顺路搭 VADAR 自己的 `sys.settrace`**（它在开 alarm 前本来就装了 tracer），
   在 tracer 里比墙上时钟 → 判定精确到行，无额外开销。
2. **后台定时器 + `PyThreadState_SetAsyncExc`**，补路径 ① 的盲区
   （`sys.settrace` 只对之后**新建的帧**生效，当前帧不被跟踪）。

**明确的降级**：两条路径都**不能可靠打断不返回的长 C 调用**
（超大图的模型推理）。Linux 上真信号可以。所以本机测出的
「程序超时」数字与 WSL/原版环境**不完全可比**。

### 3.4 稳定入口按「这次到底发生了什么」分流

三个入口文件，各管一类：

| 文件 | 什么时候写 | 关键性质 |
|---|---|---|
| `latest_run.json` | 只有真进过流水线的 run | 唯一带四类子指标 + Total 的产物 |
| `latest_plan.json` | `--plan` | 唯一带 `plan_summary.next_command` 的产物 |
| `latest_failed.json` | 前检查失败 / 选不出题 / 缺 key | 失败现场，但**不许顶掉**上一次成功 |

理由是一次实测：`--plan` 之后直接实跑（当时环境里还没有 key），运行器在缺 key 处
`return 2`，却**照旧写了 `latest_run.json`** —— 上一次 plan 里的 `next_command`
和模型指纹被一条 fatal 覆盖。**先 plan 再实跑正是本文档推荐的顺序**，
所以这个坑很容易踩到，而且踩到的时候你正需要那个文件。

另外：**fatal 态不写 `subset.json`**。一个 fatal 的题集从来没跑过，
顶掉上一次真跑过的题集会让「复现同一批题」指向一个从未执行的选择。

产物都带 `mode` 字段（`run` / `plan` / `fatal`）自述类别，不靠文件名认人。

---

## 4. 指标口径（`metrics.py`）

四个子指标逐行对齐 `engine.py:356 write_summarized_results`：

| 类别 | 题数 | 口径 |
|---|---|---|
| numeric (count) | 70 | `int(预测) == int(真值)`；`int("3.7")` 抛异常 → **记 0 分** |
| numeric (other) | 270 | MRA：相对误差 < 阈值，10 档取平均 |
| yes/no | 75 | 字符串相等（预测侧小写化） |
| multi-choice | 86 | 字符串相等（预测侧小写化） |

**Total = Σ(子指标 × 该类题数) / 501**（按题数加权的 micro 平均）。
这条**不是猜的**：用论文自己 8 行数字验算，残差 ≤ **0.043pp**
（`metrics.verify_total_aggregation()`，8 行同行数据全部命中）。

两个容易读反的地方，已经用测试钉住：

* **无法解析的预测是「记 0 分」，不是「该题被排除」**。
  `num_xxx += 1` 在 `try` 之前 —— 分母含它。
  如果实现成「排除」，准确率会系统性虚高（极端情况虚高一倍）。
* `str` 要按**答案取值**再切成 yes/no 与 multi-choice 两类，
  只按 `answer_type` 分会把两类合成一类。

> **权重分布的工程含义**：`numeric_other` 占 **53.9%** 的权重。
> 也就是说这个基准的分数**过半**来自对 float 题的相对误差。
> 任何针对 count / yes-no 的优化，天花板都不到 15%。

---

## 5. 已知的、会影响结论边界的事实

* **Omni3D-Bench 不提供 GT 相机、也不提供深度或三维框。**
  parquet 只有 6 列：`image_index / image / q_index / question / answer / answer_type`；
  HF README 的 annotations 格式也没有任何三维字段（2026-09-17 实测 + 官方说明核对）。
  ⟹ 主实验臂的横向米制尺度**仍然**走在「模型自己猜相机」那条不确定路径上，
  不能宣称被基准兜住。这条推翻了此前「主实验臂自带 GT 相机」的记录。
* **实际是 501 题、201 张图**（README 写 500）。
  300 行是同一张图的重复出现（平均每图 2.5 问）。
* `results.txt` 里 VADAR 自己的 `Accuracy` 是**逐题字符串相等**，
  float 题几乎不可能命中。它是原版遗留字段，**不是主指标**。

---

## 6. 测试

```bash
# 跑仓库根：只跑 evaluation/tests 会漏掉另外两套（tests/ 164 + scene_graph/tests/ 67）
venvs/vision/Scripts/python.exe -m pytest -q \
  --basetemp=D:/3D_Spatial_Agent/.pytest_bt -p no:cacheprovider
```

`--basetemp` 不能省：Windows 上 pytest 回收系统临时目录时会被安全钩子拦下，
进程以 `EXIT=1` 退出且**不打印汇总行**（看起来像失败，其实全过）。详见 `logs/README.md`。

`evaluation/tests/` 现有 **85** 个用例（其中 10 条是配置文件接线：
优先级、换文件撤销、写坏文件不生效、以及**整份产物序列化后搜不到 key 明文**），
`tests/test_vadar_env.py` 另有 **45** 条守解析语义与模板卫生。
全部零 torch、零 GPU、零联网。

> 值得单说一条：`tests/test_vadar_env.py::TestNoLeak` 不是"断言某个字段被掩码"，
> 而是**把整份报告序列化后逐字符搜 key 明文**。前者是点、后者是面 ——
> 将来任何人往报告里加字段，都会被它自动拦住。
