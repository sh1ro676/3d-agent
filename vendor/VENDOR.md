# vendor/ — 第三方只读检出

本目录下的仓库**不纳入版本控制**（见根目录 `.gitignore` §4）。它们本身就是 git clone，
体积大且非本项目代码。此处记录**固定版本**，需要时按下表重新获取即可。

记录日期：2026-09-20（本项目首次纳入 git 时实测）

| 目录 | 远端 | 分支 | HEAD | 本地改动 |
|---|---|---|---|---|
| `vendor/VADAR` | https://github.com/damianomarsili/VADAR.git | `main` | `56018eb` | **无** |
| `vendor/UniDepth` | https://github.com/lpiccinelli-eth/UniDepth.git | — | `8d8cfe4` | **无** |

「本地改动：无」是执行 `git status --porcelain` 得到的**空输出**，即两者都是纯净检出。
⟹ 忽略它们不会丢失任何本项目的工作。

## 重新获取

```bash
git clone https://github.com/damianomarsili/VADAR.git vendor/VADAR
git -C vendor/VADAR checkout 56018eb

git clone https://github.com/lpiccinelli-eth/UniDepth.git vendor/UniDepth
git -C vendor/UniDepth checkout 8d8cfe4
```

## 各自的许可证（引用时须注明）

- **UniDepth**：CC BY-NC 4.0 — **非商用**。课程 / 学术可用，报告须标注。
- **VADAR**：本项目的「借」是有限度的——只借它「动作空间 = LLM 输出 Python 程序」这个形式，
  弃用其 agent 层 / Engine 层 / 运行时随机生成 API。已核实的 bug 清单见
  `docs/3D_Spatial_Agent_技术调研与实施方案.md`。

## 本项目对 vendor 的用法

`vadar_env.py`（仓库根）负责在导入前替换 VADAR 的**模块级绑定**，绕开
`Generator.__init__` 构造即 `open(api.key)`、且无 `base_url` 的 blocker。
