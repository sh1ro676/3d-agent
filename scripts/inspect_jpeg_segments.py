#!/usr/bin/env python
r"""把一张 JPEG 的**段结构**打出来 —— 用来回答「EXIF 到底在不在文件里」。

## 为什么 `inspect_exif.py --raw` 不够

它在 EXIF 为空时只打印一行 `<EXIF 为空>`，而这句话对下面两种情况是**同一个输出**：

    ① 段里真的没有 APP1      —— 文件被重编码过（微信 / 社交软件转存、截图），没救
    ② 有 APP1 但 PIL 读不出  —— 段结构或标签类型不常见，可能还有救

两者的处置**完全不同**（① 只能换途径重传原图；② 值得再挖一层），所以必须分开。
本脚本绕过 PIL 直接扫 marker，就是为了把这两种情形区分开。

## 必须带阳性对照

一个「总是报告没有 APP1」的扫描器，对任何输入都会说「没有 EXIF」—— 只用它扫
目标文件是**无信息**的。同时给一个已知含 EXIF 的文件，确认它确实报出
`APP1 ← Exif`，那条「没扫到」才算证据。可用的对照：

    .cache/exif_fixture/rgb_exif.jpg      （项目自己写的夹具，小）
    .cache/cross_source/sony.jpg          （Phase 0 下的真实相机照片）
    .cache/cross_source/panasonic.jpg

2026-09-23 实测（详见 `reports/real_photo_exif_probe.md`）：两张 iPhone 照片经微信
送达后都是情形 ①（10 段、0 个 APP1、段序列逐项同构），三个对照全部报出 APP1
⟹ 「没扫到」是有信息的。

## 用法

    # 目标 + 阳性对照一起给（缺了对照，结论不成立）
    python scripts/inspect_jpeg_segments.py 待查的图.jpg .cache/cross_source/sony.jpg

## 顺带能看出来的东西

段序列本身就是指纹：

    APP0/JFIF + APP2(ICC_PROFILE) + DQT×2 + SOF0 + DHT(31,181,31,181)

`DHT` 长度 31/181 是 libjpeg 的**标准** Huffman 表；相机直出一般用按图优化的表，
长度不会是这两个数。所以「标准表 + 小 ICC + 无 APP1」基本可以判定这张图过了
某个默认参数的重编码器 —— 这解释了 EXIF 为什么会没。
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

MARKER_NAMES = {
    0xE0: "APP0/JFIF", 0xE1: "APP1", 0xE2: "APP2", 0xE3: "APP3",
    0xE4: "APP4", 0xE5: "APP5", 0xE6: "APP6", 0xE7: "APP7",
    0xE8: "APP8", 0xE9: "APP9", 0xEA: "APP10", 0xEB: "APP11",
    0xEC: "APP12", 0xED: "APP13", 0xEE: "APP14", 0xEF: "APP15",
    0xFE: "COM",
}


def scan(path: Path) -> list[tuple[str, str]]:
    b = path.read_bytes()
    if b[:2] != b"\xff\xd8":
        return [("NOT_JPEG", b[:4].hex())]
    rows: list[tuple[str, str]] = []
    i, n = 2, len(b)
    while i < n - 3:
        if b[i] != 0xFF:
            rows.append(("DESYNC@%d" % i, b[i:i + 8].hex()))
            break
        m = b[i + 1]
        if m in (0xD8, 0x01) or 0xD0 <= m <= 0xD7:
            i += 2
            continue
        if m == 0xDA:                       # SOS：之后是熵编码数据，不再有元数据段
            rows.append(("SOS", "扫瞄数据起于 @%d（共 %d B）" % (i + 2, n - i - 2)))
            break
        ln = struct.unpack(">H", b[i + 2:i + 4])[0]
        payload = b[i + 4:i + 2 + ln]
        name = MARKER_NAMES.get(m, "MARKER_%02X" % m)
        # 段内标识：这几个是「谁占了这个段」，也是判断「有没有被重编码」的线索
        ident = ""
        if m == 0xE1:
            if payload[:6] == b"Exif\x00\x00":
                ident = "← Exif"
            elif payload[:5] == b"http:" or payload[:5] == b"http":
                ident = "← XMP"
            elif payload[:4] == b"Exif":
                ident = "← Exif(变体, 无\\x00\\x00)"
        elif m == 0xE2 and payload[:2] == b"MM":
            ident = "← ICC"
        elif m == 0xE2 and payload[:11] == b"ICC_PROFILE":
            ident = "← ICC_PROFILE"
        elif m == 0xED:
            ident = "← IPTC/Photoshop"
        elif m == 0xEE:
            ident = "← Adobe"
        elif m == 0xE0 and payload[:5] == b"JFIF\x00":
            ident = "← JFIF"
        elif m == 0xFE:
            ident = "← 注释: " + payload[:40].decode("latin-1", "replace")
        rows.append((name, "len=%-6d %s" % (ln, ident)))
        i += 2 + ln
    return rows


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    n_pos = 0
    for a in sys.argv[1:]:
        p = Path(a)
        if not p.is_file():
            print("=== %s  [不存在]" % a)
            continue
        rows = scan(p)
        n_exif = sum(1 for _k, v in rows if "Exif" in v)
        n_pos += int(n_exif > 0)
        print("=== %s  %d B  (%d 段, Exif 段 %d)" %
              (a, p.stat().st_size, len(rows), n_exif))
        for k, v in rows:
            print("     %-12s %s" % (k, v))
        print()
    if len(sys.argv) > 2:
        print("小结：%d/%d 个文件含 Exif 段。"
              % (n_pos, len(sys.argv) - 1))
        if n_pos == 0:
            print("⚠ 一个都没扫到 —— **先怀疑扫描器，再下结论**：换一个 100% 含 EXIF 的"
                  "文件（如 .cache/cross_source/sony.jpg）当对照重跑，它必须报出 APP1。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
