r"""端到端验证：上传一张图 → 异步建图 → 并入演示台。

为什么写成文件而不是一行命令：本机单条前台命令约 121 s 会被杀，
而建图要加载三个模型，远超这个上限。所以脚本自己把进度**每轮都落盘** ——
即使脚本中途被杀，已经发生的事也不会丢。
"""
from __future__ import annotations

import base64
import json
import pathlib
import time
import urllib.request

ROOT = pathlib.Path(r"D:\3D_Spatial_Agent")
BASE = "http://127.0.0.1:8770"
LOG = ROOT / "logs" / "_build_test.txt"

out: list[str] = []


def flush() -> None:
    LOG.write_text("\n".join(out), encoding="utf-8")


# 用仓库里已有的那张客厅图当"用户上传"的素材：它没有 EXIF（PNG），
# 所以正好能验证「无标定内参 → 场景被标为尺度未标定」这条路。
src = ROOT / "vendor" / "UniDepth" / "assets" / "demo" / "rgb.png"
raw = src.read_bytes()
out.append("上传源: %s  %.1f KB  (PNG，无 EXIF)" % (src.name, len(raw) / 1024))
flush()

body = {
    "data": base64.b64encode(raw).decode(),
    "filename": "rgb.png",
    "scene_id": "upload_probe",
    "prompt": "sofa. chair. table. picture. mirror.",
    "intrinsics": "auto",
}
req = urllib.request.Request(
    BASE + "/api/build", data=json.dumps(body).encode("utf-8"),
    headers={"Content-Type": "application/json"}, method="POST")

try:
    with urllib.request.urlopen(req, timeout=60) as r:
        sub = json.loads(r.read())
except Exception as exc:                                  # noqa: BLE001
    out.append("提交失败: %r" % (exc,))
    flush()
    raise SystemExit(1)

out.append("提交返回: " + json.dumps(sub, ensure_ascii=False))
flush()
job = sub.get("job_id")
if not job:
    raise SystemExit(1)

t0 = time.time()
last = ""
while time.time() - t0 < 1200:
    time.sleep(3)
    try:
        with urllib.request.urlopen(
                BASE + "/api/build/status?id=" + job, timeout=30) as r:
            s = json.loads(r.read())
    except Exception as exc:                              # noqa: BLE001
        out.append("[%.0fs] 轮询失败 %r" % (time.time() - t0, exc))
        flush()
        continue
    line = "[%6.1fs] %-8s %-8s | %s" % (
        s.get("elapsed_s") or 0, s.get("state"), s.get("stage"),
        (s.get("stage_text") or "")[:110])
    if line != last:
        out.append(line)
        last = line
        flush()
    if s.get("state") in ("done", "failed"):
        out.append("")
        out.append("--- 子进程日志尾部 ---")
        out.extend((s.get("tail") or [])[-28:])
        out.append("")
        out.append("--- 结果摘要 ---")
        out.append(json.dumps(s.get("result"), ensure_ascii=False, indent=1)[:3000])
        out.append("error = %s" % s.get("error"))
        break
else:
    out.append("!! 超时未结束")

flush()

# 建完后确认它真的进了演示台的场景清单
try:
    with urllib.request.urlopen(BASE + "/api/health", timeout=30) as r:
        h = json.loads(r.read())
    out.append("")
    out.append("health.scenes = %s" % (h.get("scenes"),))
except Exception as exc:                                  # noqa: BLE001
    out.append("health 探测失败 %r" % (exc,))

idx = ROOT / "demo" / "data" / "index.json"
if idx.is_file():
    d = json.loads(idx.read_text(encoding="utf-8"))
    hit = [x for x in (d.get("scenes") or []) if x.get("scene_id") == "upload_probe"]
    out.append("index.json 里 upload_probe = %s" % json.dumps(hit, ensure_ascii=False))
out.append("DONE")
flush()
