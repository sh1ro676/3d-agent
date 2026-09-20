# VADAR LLM 后端接入说明

> 结论先行：**可行。而且比我上一条消息里判断的更可行。**
> 我之前说"没有 baseline"，那个结论**只对 OpenAI 成立**，不对整条链路成立。

---

## 1. 先修正上一条消息里的一个错误判断

我说 `api.openai.com` 不通 → 没有 baseline → 实验臂 A 不存在。这个推理里混了两件事。
实测（本机 PowerShell，HEAD/GET，12s 超时）：

| 端点 | 结果 | 含义 |
|---|---|---|
| `api.openai.com` | 超时 12s+ | **网络层不可达**（真正的问题） |
| `api.deepseek.com` | 188 ms, HTTP 401 | 服务器已应答，只是没带 key → **可用** |
| `dashscope.aliyuncs.com`（通义） | 211 ms, HTTP 401 | 可用 |
| `open.bigmodel.cn`（智谱） | 171 ms, HTTP 401 | 可用 |
| `ark.cn-beijing.volces.com`（火山） | 154 ms, HTTP 401 | 可用 |
| `api.moonshot.cn` | 638 ms, HTTP 401 | 可用 |
| `api.siliconflow.cn` | 239 ms, HTTP 401 | 可用 |
| `api.hunyuan.cloud.tencent.com` | 1158 ms, HTTP 401 | 可用 |
| `openrouter.ai` | 200 | 可用（中转） |
| `hf-mirror.com` | 可用（`huggingface.co` 不可用） | 模型权重走镜像 |

`401 未授权` 说明 TCP + TLS + HTTP 全通。**所以国产端点全都能用，只有 OpenAI 不行。**

---

## 2. 这个链路唯一的硬条件：主模型必须多模态

这是**必须知道的一件事**，否则会踩坑。

`VQAModule`（`vqa()`）**不是可选的 CLEVR/GQA 专属模块**。它同时出现在两个分支里：

```
engine/predefined_modules.py:655  get_module_list()
    ├─ :656  if dataset in ["clevr","gqa"]:  -> LocateModule, VQAModule(:664), DepthModule, ...
    └─ :686  else (omni3d):                  -> LocateModule, VQAModule(:693), DepthModule, ...
```

它在 `Omni3D` 的模块清单里（`:693`）。而 `vqa()` 的实现是**内联 base64 图片**：

```
engine/predefined_modules.py:334  predict()
    :337  img.save(buffered, format="PNG")
    :338  base64.b64encode(...)
    :346  {"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}
```

而且 `vqa()` 被写进了给模型看的 API 文档里，ProgramAgent 会主动调用它
（`prompts/api_prompt.py:113`：`To determine if an object is in another object - use VQA`）。

**所以：纯文本模型当主模型 → 程序一跑到 `vqa()` 就 400。** 两种解法：

### 方案甲：单端点（推荐起步）
DeepSeek 的多模态能力现已并入 **`deepseek-flash`（DeepSeek-V4.1-Flash）**：
它**支持图片输入**，且与纯文本同价。一个端点同时干「程序合成」和「vqa」，
链路最简单，实验对照也最干净 —— 本机已实测 **视觉问答 4/4 全对**（见 §4.6）。

> ⚠ **模型名口径 2026-09-17 有变，与本文上一版不同**。官方 `Models & Pricing` 页
> 明写「请使用 `deepseek-flash` 作为模型名」，并把 `deepseek-v4-flash` 与
> `deepseek-v4-flash-vision-exp` 列为**遗留别名** —— 原话是「对应模型已退役，
> 请求由 DeepSeek-V4.1-Flash 承接，并按 Flash 价计费」。
>
> 本机实测印证了这一条：请求写的是 `deepseek-v4-flash-vision-exp`，
> 响应里 `model_returned` 回的是 **`deepseek-flash`** —— 名字不一致，
> 说明是端点做了规范化，而不是原样回显请求名。别名目前仍可用，
> 但规范名更耐久（别名有被摘掉的风险）。

### 方案乙：双端点（追求成本时）
文本走 DeepSeek，视觉走便宜得多的 Qwen3-VL 系列。
`vqa()` 的调用次数在整条链路里可能最多（每道题 N 次），所以视觉单价是成本主项。

| 模型 | 提供方 | base_url | 备注 |
|---|---|---|---|
| **`deepseek-flash`** | DeepSeek | `https://api.deepseek.com` | **规范名**（= V4.1-Flash）。1M 上下文、384K 输出、**支持 Vision**、Tool Calls |
| `deepseek-v4-flash` / `-vision-exp` | DeepSeek | 同上 | **遗留别名**，对应模型已退役，路由到 V4.1-Flash，同价 |
| ~~`deepseek-v4-pro`~~ | DeepSeek | 同上 | **正在退役**：2026-09-14 12:00 起，请求全部路由到 V4.1-Flash |
| `qwen3-vl-flash` | 阿里百炼 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | 便宜，支持 Function Calling |
| `qwen3-vl-plus` | 阿里百炼 | 同上 | 视觉更强 |

DeepSeek 价格（元/百万 token，**高峰 / 空闲**；V4 Pro 已并入 Flash，故只剩一档）：

| 模型 | 输入(未命中) | 输入(命中) | 输出 |
|---|---|---|---|
| `deepseek-flash`（含全部遗留别名） | **3.0 / 1.5** | **0.10 / 0.05** | **9.0 / 4.5** |

官方英文页同价（USD/百万）：未命中 `$0.30 / $0.15`、命中 `$0.006 / $0.003`、
输出 `$1.20 / $0.60`。两处口径一致，可互相校对。

> ⚠ **`deepseek-v4-pro` 已经不能当「强模型上界臂」用了**。官方称 V4.1 Flash
> 在性能、成本、速度上已全面超越 V4 Pro，故**有序退役 V4 Pro**：自
> 2026-09-14 12:00（北京时间）起，所有 `deepseek-v4-pro` 请求全部路由到
> V4.1-Flash 并按 Flash 价计费。**换了名字不会换到更强的模型。**
> 要强模型上界臂得换服务商（如 `qwen3-vl-plus`）。

高峰时段 = 北京时间 9:00–12:00 与 14:00–18:00，其余为空闲（半价）。
**跑全量 benchmark 放到空闲时段，账单直接砍半。**
图片按尺寸折算 token，单张上限 **384 tokens**（一张图不到一厘钱）。

> 不确定项：`qwen3-vl-flash` 的价格在不同阿里云页面上口径不一致
> （见过 `0.18/1.8` 与 `1.2/12` 两种，元/百万 token）。所以**没有内置进默认价格表**，
> 请以百炼控制台实际账单为准，用 `VADAR_PRICE_TABLE` 显式配置。

---

## 3. 接入方式：运行时适配器，不动 `vendor/VADAR/` 一行

**为什么不能直接改源码**：项目要跑实验臂 A（原始基线）。一旦编辑 `vendor/VADAR/`
里的文件，基线就永久丢失了。

**为什么适配器可行**：VADAR 每个模块都是**模块级绑定**，只要在导入 `agents`/`engine`
之前把这些命名空间里的 `Generator` 符号换掉即可：

```
engine/engine_utils.py:51        <- Generator 的定义处
engine/predefined_modules.py:23  <- `from .engine_utils import *`（星号导入）
engine/engine.py:20-25           <- 显式 from-import
agents/agents.py:19-26           <- 显式 from-import
```

**四个命名空间缺一个，就有一部分代码还在走 OpenAI。**

### 补丁点清单（都已核实行号）

| # | 位置 | 原始行为 | 适配器处理 |
|---|---|---|---|
| 1 | `engine/engine_utils.py:56-57` | `OpenAI(api_key=file.read())`，无 `base_url`，**必须有 `./api.key` 文件** | 改由环境变量驱动，不依赖文件 |
| 2 | `engine/predefined_modules.py:226` | `Generator("gpt-4o", ...)` 硬编码 | 被 `Generator` 替换覆盖 |
| 3 | `agents/agents.py:39` | `model_name="gpt-4o"` 默认值 | 同上 |
| 4 | `engine/engine.py:33` | `api_key_path="./api.key"` | 同上 |
| 5 | `demo-notebook/notebook_imports.py:44` | **独立的一份复制**（自己 new 了 `OpenAI`，`:84/:91` 硬编码 `gpt-4o`） | 未覆盖。若用 notebook 需单独处理 |

> 第 5 条要注意：`demo-notebook/notebook_imports.py` 不是 import 引擎，而是把
> engine + agents 的逻辑**复制了一份**。所以它不会跟着适配器走。

---

## 4. 换模型后才暴露的五个坑（都已实测验证）

这些是纯 gpt-4o 环境下**永远不会出现**、换成任何其他模型都可能踩到的问题。

### 坑 1：`<answer>` 缺失 → `IndexError`，整道题作废

```
engine/predefined_modules.py:354
    answer = re.findall(r"<answer>(.*?)</answer>", output, re.DOTALL)[0].lower()
                                                                        ^^^ 列表取 [0]
```

实测（本机 Python 3.13）：`<answer>blue</answer>` → `blue`；
但 `The color is blue.` / `answer: blue` / `blue` → 全部 `IndexError`。

gpt-4o 守规矩不代表别的模型守规矩。一旦偶发不吐标签，这**一道题直接消失**，
而且报错位置离原因很远（在模块内部，不是模型输出处），排查很费时。

→ 适配器改为容错兜底：取标签失败时回退到「最后一个非空行 + 去掉 `answer:` 前缀 + 去标点」，
实测能把 `"The color is blue."` 正确兜底为 `blue`，并把原始输出写进日志。
需要严格模式时设 `VADAR_STRICT_TAGS=1` 改成抛异常。

### 坑 2：方法名正则排斥返回值类型注解 → `AttributeError`

```
agents/agents.py:89   (同类写法还在 :352 / :483 / :668)
    re.compile(r"def (\w+)\s*\(.*\):").search(sig).group(1)
```

正则要求出现**字面量 `):`**。实测：

| 签名 | 结果 |
|---|---|
| `def object_height(image, bbox):` | ✅ 提取到 `object_height` |
| `def object_height(image, bbox) -> float:` | ❌ 匹配失败 → `None.group(1)` → `AttributeError` |
| `def estimate_height(image: Image, bbox: list) -> float:` | ❌ 失败 |

**只要模型给签名加返回值注解，`SignatureAgent` 就崩。** 这是本项目里
最容易在答辩时被问到的"换模型副作用"，值得写进报告。

→ 探针脚本会把这个指标单独测出来；如果命中率高，需要在 SignatureAgent 提示词里
显式禁止返回值注解（属 Phase 3 的提示词工程，不是 Phase 0）。

### 坑 3：异常分支是无限递归 + 60 秒睡眠

```
engine/engine_utils.py:88-96
    try:
        response = self.client.chat.completions.create(...)
    except Exception as e:
        time.sleep(60)
        return self.generate(prompt, messages)     # 无限递归，无上限
```

任何一次 4xx 都会变成**静默卡死**：不报错、不退出，每 60 秒重发一次。
最危险的触发方式就是坑 2 的变体 —— **模型名写错 / 把图发给纯文本模型**。

→ 适配器改为：4xx 立刻抛 `LLMError`（快速失败）；只有 429 和 5xx 才重试，
指数退避、上限默认 2 次。评估脚本逐题 catch，把单题失败记为错答而不是整轮挂掉。

### 坑 4：没传 `max_tokens`，程序可能被静默截断

```
engine/engine_utils.py:89-93
    self.client.chat.completions.create(model=..., messages=..., temperature=...)
                                                            # 没有 max_tokens
```

ProgramAgent 输出的是**完整 Python 程序**，很长。靠服务端默认上限，
有概率被截断成一个语法不完整的程序，然后 Engine 执行失败、再触发重试、
再浪费钱。而且 `finish_reason` 里写着 `length`，但没人看。

→ 适配器显式设 `VADAR_MAX_TOKENS`（默认 8192），并检测 `finish_reason=="length"`
时打 `[TRUNCATED]` 标记写进日志。

### 坑 5：思考模式（2026-09-17 已从 DeepSeek 官方文档核实）

DeepSeek V4 全系**默认开启思考模式**（默认 effort = `high`）。对程序合成来说这通常是好事，
但会显著拉长单次延迟，而 `engine/engine.py:594` 给程序执行套了 **200 秒 SIGALRM 超时** —— 
`vqa()` 是在程序**内部**被调用的，多次思考 + 多次视觉调用叠加，有撞超时的风险。

**参数名已核实**（来源：`api-docs.deepseek.com/guides/thinking_mode`），不必再猜：

| 作用 | OpenAI 格式字段 |
|---|---|
| 开关 | `{"thinking": {"type": "enabled"\|"disabled"}}` |
| 力度 | `reasoning_effort`: `"low"\|"high"\|"max"`（默认 `high`） |

所以适配器的 `VADAR_EXTRA_BODY` 填这个就是对的：

```bash
export VADAR_EXTRA_BODY='{"thinking": {"type": "disabled"}}'
```

> ⚠⚠ **一个会静默毁掉可复现性的坑（本节最重要的一条）**：官方明写
> **思考模式下 `temperature` / `top_p` / `presence_penalty` / `frequency_penalty` 全部无效**
> —— 原话是「为了兼容既有软件，设置这些参数**不会报错，但也不会有任何效果**」。
> 也就是说：**只要思考模式还开着，第 7 节里的 `VADAR_TEMPERATURE=0.2` 就是空操作**，
> 「程序合成建议低温，保证可复现」这句话**不成立**，而且**不会有任何报错或提示**。
>
> **二选一，并把这个选择写进报告**：
> ① **关思考 + 温度 0.2** → 真正拿到低温采样，但放弃思考带来的质量；
> ② **留思考 + 接受采样不可控** → 那就**不能**再声称「低温可复现」，
>    复现性只能靠 `n>1` 重复采样 + 报告方差。
>
> **不要既不关思考、又宣称低温可复现** —— 这在答辩时是会被追问的点。

探针脚本（`02_probe_llm_api.py`）用两条独立证据回答这个问题：

- `t1` 报告响应里是否出现 `reasoning_content` —— 只能说明「当前配置下开没开」；
- **`t1b` 思考模式 A/B**（2026-09-17 新增）—— 同一个提示跑两臂：
  `disabled` 臂显式带 `{"thinking":{"type":"disabled"}}`，`default` 臂什么都不带，
  对比两臂的 `reasoning_content` / `reasoning_tokens` / 中位延迟。

> ⚠ **必须实测，不能引用文档下结论**：文档说参数有效，但只有本机跑出
> 「disabled 臂无 reasoning、default 臂有 reasoning」才算证据。
> 若两臂都有 reasoning → 关闭参数被忽略，那就只能走上面第 ② 条路。
>
> ⚠⚠ **探针与 bridge 必须读同一个 `VADAR_EXTRA_BODY`。这两个脚本曾经不同步**：
> 探针的 `chat_completion()` 早就支持 `extra_body`，但 argparse 既没有 `--extra-body`
> 也不读该环境变量，5 个调用点也全都没传 —— 于是「关思考」这个选择**只对 bridge 生效、
> 对探针完全无效**。后果是首次自检报告里出现了 `reasoning_tokens`，却**无法归因**
> （分不清是「关闭参数失效」还是「探针根本没发这个参数」），报告 `cfg` 里也没记
> `extra_body`，连事后追溯都做不到。现已修复：参数透传 + 报告抬头打印实际发送的
> `extra_body` + `thinking_observed` 汇总字段。

#### 实测结论（2026-09-17，本机真实调用 15 次）

**`{"thinking": {"type": "disabled"}}` 确实有效，走第 ① 条路。**

| 臂 | 是否出现 `reasoning_content` | 中位延迟 | 输出 tokens（2 次合计） |
|---|---|---|---|
| `disabled`（带关闭参数） | **否**（2/2） | 0.98 s | **4** |
| `default`（不带） | **是**（2/2） | 1.22 s | **51**（其中思考 45） |

两臂分得很干净：带参数就没有思考痕迹，不带就有。所以：

- **主路径（T1–T4，13 次调用）思考痕迹为 0 次** ⟹ `temperature=0.2` **真正生效**，
  「低温可复现」这句话**站得住**，前提是实验章节写明「本臂关闭思考模式」。
- 代价侧：同一句「只回一个词」的请求，开思考时**输出 token 是关闭时的 12.75 倍**
  （51 vs 4），延迟 1.24 倍。程序合成的输出长达数百 token，**全量实验必须记账**。

> ⚠⚠ **这里踩过一个坑，值得记住它长什么样。** 首轮报告的 `thinking_observed`
> 输出了「思考模式生效 → `temperature` 被静默忽略 → **不得**声称低温可复现」，
> 而同一份报告的 `thinking_ab` 却说「关闭参数有效」—— **同一份报告里两个相反结论**。
> 根因不是测量错，是**派生错**：`observed_thinking()` 把 T1b 的 `default` 臂
> 也算进了「主路径」统计，而那一臂是**故意**不带关闭参数的对照组，按定义就带思考。
> 于是「整轮跑的是开还是关」恒判为「开」。
>
> **教训：一个统计量如果混入了它自己的对照组，就会给出与本实验相反的结论。**
> 修法是把 T1b **按臂拆开**，只把与本次配置同臂的那一半并入主路径，
> 对照组单独记字段。已修，并用原始记录离线复算验证。
>
> 同时修掉第二处派生错：`t1b_thinking_ab()` 只挑出 `completion_tokens` 两个字段、
> 没保留整份 `usage`，而 `token_totals()` 只认 `r["usage"]` —— 于是 **T1b 在记账里
> 恒为 0**。偏偏 T1b 就是「思考多花多少 token」那个实验，最不该漏。
> 修正后总账 `1897 / 915` → **`1945 / 970`**（T1b 的 prompt 按 T1 同句提示词估算）。

---

## 5. 这对实验章节的叙事意味着什么

**必须修正的说法：不能说"我们复现了 VADAR 的 40.4"。**
`api.openai.com` 在你的网络下不可达，gpt-4o 那条路走不通，所以论文报告值**无法本地复现**。

**但这不影响项目的成立**，反而让消融表更干净：

- 论文里的 40.4 是 **gpt-4o + Omni3D** 的数字。只能作为**论文报告值引用并加脚注**，
  说明服务商与模型不同、不可直接比较。
- 你的消融基线换成：**VADAR 原始流水线架构 + 一个强开源模型**（实验臂 B）。
  然后逐项加：3D 工具库（D）、Scene Graph（E）、QLoRA（C）、多步规划（F）。
- 这样 A→F 全部跑在**同一家服务商、同一套参数、同一份数据**上，
  内部一致性反而比跨服务商对比更好 —— 跨服务商对比本来就是混淆的。
- 想保留一个「强模型上界臂」，**DeepSeek 这边已经没有了**（V4 Pro 已路由到 Flash）。
  需要换服务商，例如 `qwen3-vl-plus`。不需要 OpenAI。

**成本：探针实测的每次调用 token 数（这是下限，不是估计）**

| 探针项 | 对应 VADAR 环节 | prompt/次 | completion/次 |
|---|---|---|---|
| T2 | ProgramAgent 程序合成 | 152 | 183 |
| T3 | SignatureAgent 签名 | 131 | 80 |
| T4 | `vqa()`（含一张图） | 259 | 31 |
| T1 | 连通性 | 12 | 2 |

**整轮探针（15 次调用）合计 1945 prompt + 970 completion = 2915 tokens，
空闲档 ¥0.0073，高峰档 ¥0.0146。** 一次不到一分钱。

> ⚠ **不要把上表直接乘题数外推。** 探针用的是**极简提示词**，
> 而真实 VADAR 的 prompt 要带上完整 API 文档 + few-shot 示例，
> 输入长度会高一个量级；`vqa()` 的图片 token 也随分辨率上升
> （探针用的是合成小图）。**真实 per-call 数只能由「跑臂」运行器量出来** ——
> 探针能证明的只是「端点机制与计价口径没问题」。

---

## 6. 今天可以做的事（不需要 WSL、不需要 GPU）

```bash
# 1) 自检 LLM 链路（Windows 原生 Python 就能跑，零第三方依赖）
#    推荐直接跑 06_deepseek_setup.ps1 —— 它设好 VADAR_EXTRA_BODY，
#    探针自己从**同一个环境变量**读，这才是真正的「与 bridge 同源」。
#
#    ⚠⚠ 不要把 extra_body 当命令行参数传！（2026-09-17 实测，真探针 exit=2）
#       a) 本宿主的 shell 向原生 exe 传含双引号的字符串会**剥掉内层双引号**：
#          '{"thinking": {"type": "disabled"}}' -> {thinking: {type: disabled}}
#          -> json.loads 失败 -> 探针 return 2，整轮自检在第一步就死。
#       b) 给原生 exe 传空/未定义变量时，它会**整个丢掉这个参数**，于是 argv 变成
#          `--extra-body --price-in 0 ...`，argparse 判成
#          "argument --extra-body: expected one argument" -> 同样 return 2。
#       ⟹ 一律「只设环境变量，不在命令行传」。探针 argparse 的默认值
#          本来就是 os.environ["VADAR_EXTRA_BODY"]，与 bridge 读的是同一处。
#    权威证据 = 报告抬头那行 `extra_body : ...`，它打印的是实际解析成功、真正发出去的内容。
cd phase0
export VADAR_EXTRA_BODY='{"thinking": {"type": "disabled"}}'   # bash
# Windows PowerShell:  $env:VADAR_EXTRA_BODY = '{"thinking": {"type": "disabled"}}'
python 02_probe_llm_api.py --base-url https://api.deepseek.com \
                           --model deepseek-flash \
                           --api-key sk-xxxx \
                           --price-in 3.0 --price-out 9.0   # 高峰价；填了才给费用外推

# 2) 验证适配器自身的调用层（同样不需要 VADAR / torch）
export VADAR_API_KEY=sk-xxxx
python 03_vadar_llm_bridge.py --selftest

# 3) 看配置是否如预期
python 03_vadar_llm_bridge.py --show-config
```

第 1 步会输出四类标签的合规率、真实延迟、`vqa()` 的答对率、**思考模式 A/B 结论**，
以及**逐项 token 记账**（T6，给出每项的 per-call token 数）。
外推全量实验时用 T6 的 per-call 数乘真实调用次数；样本太小就加 `--runs 20` 压方差。
**报告里的每个数字都是你机器上实测出来的**，可以直接进实验章节。

等 WSL + 视觉栈就绪后，在 `evaluate.py` 之前插一行 `bridge.install()` 即可切到国产模型：

```python
import sys; sys.path.insert(0, "/path/to/phase0")
import vadar_llm_bridge as bridge
bridge.install()          # 必须在 from agents... / from engine... 之前
from agents.agents import SignatureAgent, APIAgent, ProgramAgent
from engine.engine import Engine
```

---

## 7. 环境变量速查

```bash
# --- 必需 ---
export VADAR_BASE_URL=https://api.deepseek.com
export VADAR_API_KEY=sk-xxxx
export VADAR_MODEL=deepseek-flash          # 规范名；遗留别名 v4-flash / vision-exp 仍可用

# --- 常用可调 ---
export VADAR_TEMPERATURE=0.2          # 程序合成建议低温，保证可复现
export VADAR_MAX_TOKENS=8192          # 别依赖服务端默认值
export VADAR_MAX_RETRIES=2            # 4xx 不重试，只有 429/5xx 走退避
export VADAR_CALL_LOG=logs/vadar_llm_calls.jsonl
export VADAR_STRICT_TAGS=0            # 1 = 标签缺失时抛异常而不是兜底

# --- 可选：视觉独立端点 ---
export VADAR_VISION_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export VADAR_VISION_MODEL=qwen3-vl-flash
export VADAR_VISION_API_KEY=sk-yyyy

# --- 可选：透传不认识的参数（如关闭思考模式）---
# export VADAR_EXTRA_BODY='{"thinking": {"type": "disabled"}}'

# --- 可选：价格表（元/百万 token）---
# export VADAR_PRICE_TABLE='{"qwen3-vl-flash": [0.18, 1.8]}'
```

调用日志汇总（直接产出评估指标）：

```bash
python 03_vadar_llm_bridge.py --summary logs/vadar_llm_calls.jsonl
# -> 平均/p50/p95 延迟、总成本、截断数、重试数、<answer> 兜底次数、各角色分布
```
