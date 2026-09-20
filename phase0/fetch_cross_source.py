"""跨来源验证素材抓取（§22.9 证据缺口 ②）。

两个素材臂，来源、成像链路、相机都与现有的 UniDepth demo 完全不同：

  A 臂「真实相机 EXIF」(jsDelivr 代理 hMatoba/Piexif 的测试集)
      —— 7 台不同厂商/型号的真实相机照片，EXIF 是**真实的**（不是我们合成的）。
      用途：验证 `vision/exif.py` 在真实 EXIF 上是否可靠（合成夹具只证明了「格式正确」）。

  B 臂「真实场景、无 EXIF」(Lorem Picsum / Pexels / Unsplash CDN)
      —— 真实照片但 CDN 已重编码，EXIF 被剥掉。
      用途：验证「相机头预测的视场不可信」这一结论是否跨来源成立。

只用标准库 + Pillow（都不需要 GPU）。所有结果写盘，不依赖控制台。
"""

from __future__ import annotations

import json
import ssl
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / ".cache" / "cross_source"
REPORT = Path(__file__).with_name("fetch_cross_source_report.txt")
MANIFEST = Path(__file__).with_name("cross_source_manifest.json")

ssl._create_default_https_context = ssl._create_unverified_context
UA = {"User-Agent": "Mozilla/5.0 (3D-Spatial-Agent; course project; contact: local)"}

JSD = "https://cdn.jsdelivr.net/gh/hMatoba/Piexif@master/tests/images"

# A 臂：真实相机（文件名即相机厂商，Piexif 测试集的 r_ 前缀文件）
ARM_A = [
    ("canon", f"{JSD}/r_canon.jpg"),
    ("casio", f"{JSD}/r_casio.jpg"),
    ("olympus", f"{JSD}/r_olympus.jpg"),
    ("panasonic", f"{JSD}/r_pana.jpg"),
    ("pentax", f"{JSD}/r_pen.jpg"),
    ("ricoh", f"{JSD}/r_ricoh.jpg"),
    ("sigma", f"{JSD}/r_sigma.jpg"),
    ("sony", f"{JSD}/r_sony.jpg"),
]

# B 臂：真实场景、横竖混合（顺带压测长边约定与 Orientation）
ARM_B = [
    ("picsum_land_a", "https://picsum.photos/id/1018/1600/1200"),
    ("picsum_land_b", "https://picsum.photos/id/1036/1600/1067"),
    ("picsum_port_a", "https://picsum.photos/id/1005/1200/1600"),
    ("picsum_port_b", "https://picsum.photos/id/1027/1200/1600"),
    ("picsum_square", "https://picsum.photos/id/1062/1200/1200"),
    ("pexels_photo", "https://images.pexels.com/photos/414612/pexels-photo-414612.jpeg?w=1600"),
    ("unsplash_photo", "https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=1600"),
    ("pixabay_tree", "https://cdn.pixabay.com/photo/2015/04/23/22/00/tree-736885_1280.jpg"),
]


def fetch(name: str, url: str) -> dict:
    out = DEST / f"{name}.jpg"
    rec = {"name": name, "url": url, "path": None, "bytes": 0, "ok": False, "note": ""}
    if out.exists() and out.stat().st_size > 1024:
        rec.update(path=str(out), bytes=out.stat().st_size, ok=True, note="cached")
        return rec
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=30) as r:
            blob = r.read()
        if len(blob) < 1024:
            rec["note"] = f"too small ({len(blob)} B)"
            return rec
        out.write_bytes(blob)
        rec.update(path=str(out), bytes=len(blob), ok=True)
    except Exception as e:  # noqa: BLE001 - 抓取失败要记录而不是中断
        rec["note"] = f"{type(e).__name__}: {str(e)[:80]}"
    return rec


def probe_image(rec: dict) -> dict:
    """读尺寸 + EXIF 摘要，供臂归类与后续探针使用。"""
    if not rec.get("ok"):
        return rec
    try:
        from PIL import Image, ExifTags

        with Image.open(rec["path"]) as im:
            rec["size"] = list(im.size)
            ex = im.getexif()
            rec["exif_ifd0_tags"] = len(ex)
            sub = {}
            try:
                sub = ex.get_ifd(0x8769)
            except Exception:  # noqa: BLE001
                sub = {}
            rec["exif_subifd_tags"] = len(sub)
            tags = {}
            for k, v in list(ex.items()) + list(sub.items()):
                tags[ExifTags.TAGS.get(k, str(k))] = v if not isinstance(v, bytes) else "<bytes>"
            rec["exif"] = {k: str(v)[:60] for k, v in tags.items()}
            rec["make"] = tags.get("Make")
            rec["model"] = tags.get("Model")
            rec["has_focal"] = any(k in tags for k in ("FocalLength", "FocalLengthIn35mmFilm"))
            rec["orientation"] = tags.get("Orientation")
    except Exception as e:  # noqa: BLE001
        rec["ok"] = False
        rec["note"] = f"PIL: {type(e).__name__}: {str(e)[:80]}"
    return rec


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    lines = ["跨来源素材抓取 — A 臂（真实相机 EXIF）+ B 臂（真实场景、无 EXIF）", ""]
    records = []

    for arm, items in (("A_camera_exif", ARM_A), ("B_scene_noexif", ARM_B)):
        lines.append(f"=========== {arm} ===========")
        for name, url in items:
            rec = probe_image(fetch(name, url))
            rec["arm"] = arm
            records.append(rec)
            if rec["ok"]:
                sz = rec.get("size")
                lines.append(
                    f"  OK   {name:16} {rec['bytes']:>9} B  {sz!s:>14}  "
                    f"IFD0={rec.get('exif_ifd0_tags')} Sub={rec.get('exif_subifd_tags')}  "
                    f"make={rec.get('make')} model={rec.get('model')}"
                )
                lines.append(f"       → {rec.get('path')}")
            else:
                lines.append(f"  FAIL {name:16} {rec['note']}")
        lines.append("")

    lines.append("=========== A 臂 EXIF 明细（只要有 FocalLength 的）===========")
    for rec in records:
        if rec.get("arm") != "A_camera_exif" or not rec.get("ok"):
            continue
        lines.append(f"--- {rec['name']} {rec.get('size')} make={rec.get('make')} model={rec.get('model')}")
        for k, v in sorted((rec.get("exif") or {}).items()):
            if k in (
                "FocalLength",
                "FocalLengthIn35mmFilm",
                "ExifImageWidth",
                "ExifImageHeight",
                "PixelXDimension",
                "PixelYDimension",
                "Orientation",
                "Make",
                "Model",
                "ExifOffset",
                "ExifVersion",
            ):
                lines.append(f"      {k} = {v}")

    ok_a = sum(1 for r in records if r["arm"].startswith("A") and r.get("ok"))
    ok_b = sum(1 for r in records if r["arm"].startswith("B") and r.get("ok"))
    fx_a = sum(1 for r in records if r["arm"].startswith("A") and r.get("has_focal"))
    lines += [
        "",
        "=========== SUMMARY ===========",
        f"A 臂成功 {ok_a}/{len(ARM_A)}，其中带焦距标签 {fx_a}",
        f"B 臂成功 {ok_b}/{len(ARM_B)}",
        f"素材目录 {DEST}",
    ]

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    MANIFEST.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"report -> {REPORT}")
    print(f"manifest -> {MANIFEST}")
    print("A ok", ok_a, "of", len(ARM_A), "| B ok", ok_b, "of", len(ARM_B))
    return 0


if __name__ == "__main__":
    sys.exit(main())
