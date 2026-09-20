#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""⛔ 已被取代 —— 请用 `scripts/show_answer_provenance.py`。

这个文件是一次性诊断脚本，用途是查出「C8 的答案 `1` 到底命中了 trace 里的哪个数」——
它确实查出结果了（`('method', 1.0)`，来自算法版本号 `"geometry_v1"` 里的那个 `1`），
但它是**针对一道题写死的**。

那份能力已经**升格成常驻工具**：`scripts/show_answer_provenance.py`
（用法：`python scripts/show_answer_provenance.py [--only C8] [--key method,tol]`，
产物 `reports/answer_provenance.md`）。它做了两件这个一次性脚本做不到的事：

1. **遍历全部题目**，不是只看 C8；
2. ⭐ **冻了一份已知有缺陷的旧 `_ID_RE`**，用来**重放**「修复前会命中什么」——
   否则修完之后，「修复到底修掉了什么」就只剩一句转述（我自己第一版就是这样，
   用**当前**口径重放，结果一片空白）。

⚠ 本文件本该删掉，但**本机 `Remove-Item` 删不掉任何文件**
（安全守卫静默拦下：`deleted=0`，不报错）。于是改为留这个占位说明，
以免下次有人以为它是活的。**不要在这里继续加东西。**
"""

from __future__ import annotations

if __name__ == "__main__":
    raise SystemExit(
        "已被 scripts/show_answer_provenance.py 取代 —— 见本文件 docstring。"
    )
