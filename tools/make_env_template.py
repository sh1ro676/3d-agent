#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从工作副本生成可分发模板：`configs/llm_backend.env` → `.template`。

为什么需要两个文件
------------------
`configs/llm_backend.env` 是**工作副本**：它含真 key，不进版本库、不进报告、不进截图。
`configs/llm_backend.env.template` 是**可分发模板**：key 位留空，随仓库走。

合成一个文件会掉进一个很隐蔽的坑：测试若断言「这个文件的 key 必须是空的」，
那么**用户一旦正确填了 key，测试就永久变红**。红灯于是失去信号，
而失去信号比没有测试更危险 —— 真正的 key 泄露将不会再被人注意到。
所以「谁的 key 必须空」这件事，必须由**文件名**回答，不能由**内容**回答。

用法
----
    venvs/vision/Scripts/python.exe tools/make_env_template.py

改过 `llm_backend.env`（加/改选项）之后跑一次即可。模板与工作副本的
**键集合**必须一致（`evaluation/tests` 里有断言守着），否则新用户会缺选项。
"""

from __future__ import annotations

import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import vadar_env  # noqa: E402

LIVE = os.path.join(ROOT, "configs", "llm_backend.env")
TEMPLATE = os.path.join(ROOT, "configs", "llm_backend.env.template")

#: `KEY=` / `KEY = ` —— 值部分整个抹掉
_LINE_RE = re.compile(r"^(?P<indent>\s*)(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=(?P<val>.*)$")

#: 像真 key 的串：`sk-` 后跟足够长的随机段。文档里的占位符（`sk-xxxxxxxx`，8 个 x）
#: 不会被命中 —— 阈值故意卡在 16，就是为了让「文档提到前缀」和「真泄露」可分。
_KEYLIKE_RE = re.compile(r"sk-[A-Za-z0-9_\-]{16,}")


def blank_secrets(text: str):
    """把密钥类键的值清空，其余逐字保留（含注释与排版）。返回 (新文本, 清空计数)。"""
    out, n = [], 0
    for line in text.splitlines(keepends=True):
        m = _LINE_RE.match(line.rstrip("\r\n"))
        if m and vadar_env.is_secret(m.group("key")):
            out.append("%s%s=\n" % (m.group("indent"), m.group("key")))
            n += 1
        else:
            out.append(line)
    return "".join(out), n


def main() -> int:
    if not os.path.isfile(LIVE):
        print("工作副本不存在: %s" % LIVE)
        return 2

    live_text = io.open(LIVE, encoding="utf-8").read()
    tpl_text, n = blank_secrets(live_text)

    with io.open(TEMPLATE, "w", encoding="utf-8", newline="") as f:
        f.write(tpl_text)

    tpl_values = vadar_env.parse_env_text(tpl_text)
    live_values = vadar_env.parse_env_text(live_text)

    # 自检三条 —— 生成器自己出错的话，后面所有依赖模板的测试都会失去意义
    leaked = [k for k in tpl_values if vadar_env.is_secret(k) and tpl_values[k]]
    assert not leaked, "模板里仍有非空密钥: %s" % leaked

    # 只扫**非注释行**。文件里的文档注释本来就写着「形如 sk-xxxxxxxx...」，
    # 拿裸子串 `sk-` 去断言会把文档本身判成泄露 —— 检查本身也得有精度。
    for raw in tpl_text.splitlines():
        if raw.lstrip().startswith("#"):
            continue
        assert not _KEYLIKE_RE.search(raw), "模板的生效行里出现疑似真 key: %s" % raw.strip()[:40]

    tpl_keys, live_keys = set(tpl_values), set(live_values)
    only_live, only_tpl = sorted(live_keys - tpl_keys), sorted(tpl_keys - live_keys)

    print("清空密钥键 %d 个: %s" % (n, ", ".join(sorted(k for k in tpl_keys
                                                      if vadar_env.is_secret(k)))))
    print("模板: %s (%d 字节)" % (TEMPLATE, os.path.getsize(TEMPLATE)))
    if only_live or only_tpl:
        print("⚠ 键集合不一致 —— 模板需要同步：")
        for k in only_live:
            print("    只在工作副本里: %s" % k)
        for k in only_tpl:
            print("    只在模板里:     %s" % k)
        return 1
    print("键集合一致（%d 个键），可以分发。" % len(tpl_keys))
    return 0


if __name__ == "__main__":
    sys.exit(main())
