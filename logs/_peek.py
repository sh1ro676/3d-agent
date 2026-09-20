import json, urllib.request, pathlib
base = "http://127.0.0.1:8770"
out = []
def ask(scene, q, atype):
    req = urllib.request.Request(base + "/api/ask",
        data=json.dumps({"scene_id": scene, "question": q, "answer_type": atype,
                         "planner": "off", "vlm": True}).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=240) as r:
        return json.loads(r.read())

for q, atype in (("场景里一共有多少个物体？", "int"),
                 ("沙发和桌子之间的三维距离是多少米？", "float")):
    out.append("=== [upload_probe] %s ===" % q)
    try:
        d = ask("upload_probe", q, atype)
        if not d.get("ok"):
            out.append("  失败: " + str(d.get("error"))); continue
        run = d["run"]
        out.append("  status=%s  answer=%r  attempts=%s  elapsed=%.2fs" % (
            run.get("status"), run.get("answer"), run.get("attempts"), run.get("elapsed_s") or 0))
        v = run.get("verdict") or {}
        out.append("  verdict=%s" % v.get("level"))
        out.append("  evidence=" + json.dumps(run.get("evidence"), ensure_ascii=False)[:260])
        out.append("  tools=%s" % [t.get("tool") for t in (run.get("trace") or [])])
        out.append("  cost=%s CNY  saved=%s" % ((d.get("usage") or {}).get("cost_cny"), d.get("saved_as")))
    except Exception as exc:
        out.append("  FAIL %r" % (exc,))

# 反事实：对新场景也走一次
out.append("")
out.append("=== [upload_probe] 反事实：移走 table_1 ===")
try:
    req = urllib.request.Request(base + "/api/counterfactual",
        data=json.dumps({"scene_id": "upload_probe", "remove": ["table_1"]}).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read())
    res = d.get("result") or {}
    v = res.get("value") or {}
    diff = v.get("diff") or {}
    out.append("  ok=%s  关系 %s → %s" % (res.get("ok"), diff.get("n_relations_before"), diff.get("n_relations_after")))
    out.append("  evidence=" + json.dumps(res.get("evidence"), ensure_ascii=False)[:300])
except Exception as exc:
    out.append("  FAIL %r" % (exc,))

pathlib.Path(r"D:\3D_Spatial_Agent\logs\_upload_ask.txt").write_text("\n".join(out), encoding="utf-8")
