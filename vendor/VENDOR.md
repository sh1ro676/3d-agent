# vendor/ — 第三方只读检出

本目录下的仓库**不纳入版本控制**（见根目录 `.gitignore` §4）。它们本身就是 git clone，
体积大且非本项目代码。此处记录**固定版本**，需要时按下表重新获取即可。

记录日期：2026-09-20（本项目首次纳入 git 时实测）

| 目录 | 远端 | HEAD | 本地改动 |
|---|---|---|---|
| `vendor/UniDepth` | https://github.com/lpiccinelli-eth/UniDepth.git | `8d8cfe4` | **无** |

「本地改动：无」是执行 `git status --porcelain` 得到的**空输出**，即纯净检出。
⟹ 忽略它不会丢失任何本项目的工作。

> **变更记录（2026-09-20）**：本目录原先还有一个**早期基线检出**，同日本项目
> 与其解耦，该检出已从磁盘和仓库中移除。代码里若出现「已移除的早期基线检出」
> 这类说法，指的就是它；出处见 README 的「参考与致谢」，需要时按那里的链接取回。

## 重新获取

```bash
git clone https://github.com/lpiccinelli-eth/UniDepth.git vendor/UniDepth
git -C vendor/UniDepth checkout 8d8cfe4
```

## 许可证（引用时须注明）

- **UniDepth**：CC BY-NC 4.0 — **非商用**。课程 / 学术可用，报告须标注。

## 本项目对 vendor 的用法

`vision/` 通过 `PerceptionStack` 调用 UniDepth 的模型定义与权重。
权重不在本目录，而是在 `.cache/models/`（检测 / 分割）与 `.cache/huggingface/`（深度），
两者同样不进版本控制，按 README 的环境准备一节重建。
