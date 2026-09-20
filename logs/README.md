# logs/ 目录约定

本目录只放**过程性产物**（命令输出、探针记录、归档包），不放源码、不放数据、不放交付物。

## 留下来的东西（长期保留）

| 文件 | 为什么留 |
|---|---|
| `README.md` | 本文件，目录约定 |
| `archive_*.zip` | 已归档的历史过程输出。**归档 = 可恢复的删除**，需要时解压回查 |
| `before_*.zip` | **改动前的源码快照**（含当时的补丁脚本）。本项目**没有 git**，「改之前长什么样」只能靠它 —— 不改动前归档，事后就无法区分「本来就这样」与「被我改过」 |
| `_pytest_final.txt` | 最近一次全量测试的**权威记录**（见下方「测试口径」）。⚠ **2026-09-20 被一次中断的运行覆盖过，见下** |
| `phase1_probe_*.json` | LLM 探针报告。`first_run` = 首轮（含**已验证有误的派生字段**）、`raw` = 打补丁前的原始记录；**两者都是证据，别删** |
| `phase1d_k_sweep.txt` | 内参剂量-反应原始输出，方案文档 §21 的数字**由它背书** |
| `phase1e_principal_point.txt` | 主点偏移探针原始输出，方案文档 §22 的数字**由它背书** |
| `vadar_llm_calls.jsonl` | VADAR LLM 桥接调用的真实 trace（`phase0/03_vadar_llm_bridge.py` 产物） |
| `_ls_reports.txt`、`_ls2.txt`、`_collect*.txt`、`_mem_sync.txt`、`_cleanup.txt` | 一次性命令输出（`_` 前缀）。留着的价值：`_cleanup.txt` 是「本机删不掉文件」的**证据** |
| `glue_programs.txt` | ⛔ **已被取代** —— 同一份转储的正本现在在 `reports/glue_programs.txt`（它被 `reports/glue_mining.md` 引用，按目录约定该放 reports/）。本机删不掉，所以留个标记：**别引用这一份** |
| `probe_combination.log`、`mine_glue.log`、`probe_analyze.log`、`_pytest_final.txt` | 最近一次运行的输出。可覆盖，但别当历史证据引用 |

## 测试口径（重要，别记错）

**全量测试必须跑仓库根目录，不能只跑 `tests/`：**

```bash
D:/3D_Spatial_Agent/venvs/vision/Scripts/python.exe -m pytest -q
```

原因：本项目有**三套**测试，`tests/` 只覆盖其中一套：

| 套件 | 数量 | 覆盖 |
|---|---|---|
| `tests/` | 556 | vision 层（geometry/exif/depth）+ 工具层（tools/、L5 报告层）+ 自研 Agent 层（契约 / 校验器 / 提示词预算 / 静态检查） |
| `scene_graph/tests/` | 85 | 场景图构建（builder）、关系计算（relations）、存储（store） |
| `evaluation/tests/` | 114 | 实验臂运行器（题集选择/指标口径/兼容层/alarm 替代/产物分流） |
| **合计** | **755** | — |

⚠ 上面这张表是 **2026-09-19** 的数（原表 306 是 09-17 的，已过期）。
**别把「上次记的数」当成「现在的数」**：验证一句
`python -m pytest --collect-only -q -p no:cacheprovider | find /c "::"` 就够
（本机没有 `find`/`grep`，用 PowerShell 数含 `::` 的行，见下方 Windows 坑 2）。
只跑其中一层不会有任何提示 —— `pytest tests/` 会安静地少收 199 个用例。

只跑 `pytest tests/` 会**少收 199 个用例**且不会报错，看起来"绿色"却漏了两整层——这是本项目最容易踩的假绿陷阱。
（2026-09-20 勘误：此处原写「142」，与上一段的 199 自相矛盾。**同一份文件里两个数互相打架，
比两个数都错更容易蒙混过去** —— 因为读的人会挑一个相信。）

### ⚠ `_pytest_final.txt` 目前**不是**一份完整记录（2026-09-20）

2026-09-20 曾想再跑一次全量来复验，命令**在约 121 秒处被宿主 shell 杀掉**
（`Set-Location … | Out-File` 这条链没跑完），而 `Out-File` 是**覆盖**写 ⟹
那份「**755 passed, 0 skipped**」的权威记录**被 28% 的半截输出覆盖掉了**。
现在这个文件里只有三行进度点和一段失败标记，**没有 `EXIT=` 行**。

- **755 这个数字仍然成立**（来源是 2026-09-19 那次完整运行，见上表），只是**不在这个文件里了**。
- 想恢复这份权威记录，**必须再跑一次完整全量**（后台跑，避开 121 秒上限）。
  按用户口径（「先不用测试，只要功能框架」）**没有自动重跑**。
- ⭐ **教训**：把「权威记录」交给一个**会被覆盖**的文件时，那个文件必须
  ①要么只在成功结束时写，②要么写成带时间戳的新文件。
  现在这个写法，一次失败就能**静默销毁**上一次的成功证据 —— 和 `reports/README.md`
  说的「已被取代的文件看起来正确、实际过期」是同一类风险。

### 一条会「静默变绿」的坑（2026-09-17 踩过）

`tests/test_vision_exif.py` 里有一条哨兵用例，靠「本进程是否加载过 torch」
验证某模块没有在模块级 import torch。**它的失败方式是 SKIP，不是 FAIL** ——
也就是说只要有任何测试在跑的过程中把 torch 拉进进程，那条断言就会安静地消失，
而汇总行只会从「306 passed」变成「305 passed, 1 skipped」。

所以：**看到 `skipped` 要当成失败处理**，先 `-rs` 查是哪一条、为什么。
`evaluation/tests/test_runner.py::TestEvaluationModulesStayTorchFreeAtImport`
用 AST 静态检查守住同一件事（不依赖进程状态、不依赖测试顺序）。

### Windows 上的两个坑

1. **必须给 `--basetemp`**（指定到项目内）：

   ```bash
   python -m pytest -q --basetemp=D:/3D_Spatial_Agent/.pytest_bt -p no:cacheprovider
   ```

   否则 pytest 在会话末尾回收系统临时目录里的 `garbage-*` 时会被安全钩子拦下，进程以 **EXIT=1** 退出，**且不打印 `N passed` 汇总行**——看起来像失败，其实全部用例已通过。

2. **输出重定向到文件再读**：本机 Git Bash 没有 `head`/`tail`/`grep`，直接管道会报 `command not found`。

## 命名约定

- `_xxx.txt` — 临时命令输出（`_` 前缀 = 一次性草稿）
- `phaseN*_xxx.txt` — 某阶段跑出来的过程记录
- `probe_*` / `*_probe.txt` — 探针实验输出
- `archive_<主题>_<日期>.zip` — 归档包（历史过程输出）
- `before_<主题>_<日期>.zip` — 改动前快照（**无 git 时的 undo 依据**）

## 什么不该进 logs/

- 交付物（报告、图、文档）→ 放 `dataset/scenes/*/`、`docs/`
- 可复现的实验配置 → 放 `phase0/`、`scripts/`
- 任何被文档正文引用的**结论**，其原始输出应升格为上面「长期保留」区的一员，而不是躺在草稿堆里
