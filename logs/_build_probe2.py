r"""验证两件事：

1. 修复后的结果摘要里必须能看到**内参来源**与 FOV —— 它们是"尺度未标定"
   这条提示的唯一具体证据。
2. **排队语义**：连着提交两个任务时，第二个应当如实停在 `queued`，
   而不是两个建图一起把 8188 MiB 显存挤爆。
   （锁被刻意放在工作线程里而不是提交处，所以这个行为必须实测，不能靠读代码确认。）
"""
from __future__ import annotations

import base64
import json
import pathlib
import time
import urllib.request

ROOT = pathlib.Path(r"D:\3D_Spatial_Agent")
BASE = "http://127.0.0.1:8770"
LOG = ROOT / "logs" / "_build_test2.txt"

out: list[str] = []


def flush() -> None:
    LOG.write_text("\n".join(out), encoding="utf-8")


def get(path: str, timeout: int = 30):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.loads(r.read())


def submit(scene_id: str) -> str:
    raw = (ROOT / "vendor" / "UniDepth" / "assets" / "demo" / "rgb.png").read_bytes()
    body = {"data": base64.b64encode(raw).decode(), "filename": "rgb.png",
            "scene_id": scene_id, "prompt": "sofa. chair. table. picture. mirror.",
            "intrinsics": "auto"}
    req = urllib.request.Request(BASE + "/api/build", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read()).get("job_id")


out.append("提交 A ...")
a = submit("upload_probe2")
out.append("  A job = %s" % a)
flush()

out.append("立刻提交 B ...")
b = submit("upload_probe3")
out.append("  B job = %s" % b)
flush()

t0 = time.time()
seen: dict[str, set] = {a: set(), b: set()}
last_line = ""
finished: set[str] = set()
snaps: dict[str, dict] = {}

while time.time() - t0 < 1500 and len(finished) < 2:
    time.sleep(3)
    for tag, job in (("A", a), ("B", b)):
        if job in finished:
            continue
        try:
            s = get("/api/build/status?id=" + job)
        except Exception as exc:                          # noqa: BLE001
            out.append("[%5.0fs] %s 轮询失败 %r" % (time.time() - t0, tag, exc))
            flush()
            continue
        snaps[job] = s
        key = "%s|%s" % (s.get("state"), s.get("stage"))
        if key not in seen[job]:
            seen[job].add(key)
            out.append("[%6.1fs] %s job=%s  state=%-8s stage=%-8s  %s" % (
                s.get("elapsed_s") or 0, tag, job, s.get("state"), s.get("stage"),
                (s.get("stage_text") or "")[:80]))
            flush()
        if s.get("state") in ("done", "failed"):
            finished.add(job)
            out.append("    ↑ %s 结束：%s" % (tag, s.get("state")))
            flush()

out.append("")
out.append("=== 排队行为判定 ===")
b_saw_queued = any(k.startswith("queued") for k in seen[b])
a_saw_queued = any(k.startswith("queued") for k in seen[a])
out.append("A 见过 queued = %s（它先提交，通常没来得及被观察到）" % a_saw_queued)
out.append("B 见过 queued = %s  ← 这是排队的证据" % b_saw_queued)
out.append("B 观察到过的 (state|stage) 序列：%s" % sorted(seen[b]))

out.append("")
out.append("=== A 的结果摘要（看内参来源）===")
ra = (snaps.get(a) or {}).get("result") or {}
for k in ("scene_id", "image_url", "image_hw", "n_nodes", "n_edges",
          "intrinsics_source", "intrinsics", "fov",
          "scale_calibrated", "up_axis_tilt_deg", "up_axis_reliable", "up_axis_reason",
          "mask_box_coverage_min", "prompt"):
    out.append("  %-24s %s" % (k, json.dumps(ra.get(k), ensure_ascii=False)))
flush()

out.append("")
out.append("=== B 的结果摘要 ===")
rb = (snaps.get(b) or {}).get("result") or {}
for k in ("scene_id", "image_url", "image_hw", "intrinsics_source", "scale_calibrated"):
    out.append("  %-24s %s" % (k, json.dumps(rb.get(k), ensure_ascii=False)))

out.append("")
out.append("=== 上传目录 ===")
up = ROOT / "uploads"
if up.is_dir():
    for p in sorted(up.rglob("*")):
        if p.is_file():
            out.append("  %s  %d bytes" % (p.relative_to(ROOT), p.stat().st_size))

out.append("")
out.append("=== 导出后的资源 ===")
for sid in ("upload_probe2", "upload_probe3"):
    d = ROOT / "demo" / "assets" / sid
    if d.is_dir():
        out.append("  assets/%s/ → %s" % (sid, sorted(x.name for x in d.iterdir())))

out.append("")
try:
    h = get("/api/health")
    out.append("health.scenes = %s" % (h.get("scenes"),))
except Exception as exc:                                  # noqa: BLE001
    out.append("health 失败 %r" % (exc,))
out.append("DONE")
flush()
