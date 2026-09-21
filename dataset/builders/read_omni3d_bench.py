#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""read_omni3d_bench.py —— 把 Omni3D-Bench 的 parquet 展开成本项目数据集契约的目录布局。

为什么需要这一层
----------------
HF 仓库里只有一个 parquet（106 MB），而本项目的数据集契约期望的是：

    data/omni3d-bench/annotations.json      # {"questions": [ {...}, ... ]}
    data/omni3d-bench/images/<image_filename>

而且字段名是硬约定 —— 逐条从上游实现的源码里读出来的，不是猜的：
    evaluate.py:29          questions = questions_data["questions"]
    evaluate.py:42          random.sample(questions, num_api_questions)
    agents.py:157,806       question_data["image_filename"]
    agents.py:132           question_data["image_index"] / ["question_index"]
    engine.py:110           os.path.join(images_folder_path, question["image_filename"])
    engine.py:434-443       question["answer_type"], question["answer"]

    ⚠ 上面几处行号指向的是**已从本仓库移除**的早期基线检出。之所以保留，
      是因为它们解释了「字段名为什么恰好是这几个」；要核对时按 README 的
      参考文献找回上游即可。

所以本脚本做三件事：
    ① 读 parquet，把每行的图像字节写成 `images/<image_id>.png`
    ② 生成 `annotations.json`（约定字段名 + 原样保留其余列到 `_extra`）
    ③ 把 schema、answer_type 分布、去重后的图片数等写进 `read_report.json`

为什么要单独写 `_extra`
------------------------
Omni3D-Bench 的每行可能还带 3D 标注（Phase 12 的 GT 相机就指望它）。
丢掉就再也拿不回来了（parquet 有，但没人会想再解析一遍），
所以**原样留着**，即使当前臂不用。

用法
----
    python dataset/builders/read_omni3d_bench.py --inspect          # 只看 schema
    python dataset/builders/read_omni3d_bench.py                   # 展开 + 写报告
    python dataset/builders/read_omni3d_bench.py --limit 40         # 只导出前 40 题
"""

import argparse
import hashlib
import io
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
RAW = os.path.join(REPO_ROOT, "dataset", "raw", "omni3d-bench")
PARQUET = os.path.join(RAW, "train-00000-of-00001.parquet")
OUT_DIR = os.path.join(RAW, "unpacked")
IMAGES_DIR = os.path.join(OUT_DIR, "images")
ANNOTATIONS = os.path.join(OUT_DIR, "annotations.json")
REPORT = os.path.join(OUT_DIR, "read_report.json")

#: 候选列名 → 约定字段。按优先级找第一个存在的。
#: 实测（2026-09-17，parquet schema）真实列名是：
#:     image_index / image / q_index / question / answer / answer_type
#: 注意 `q_index` —— 不是 `question_index`。下游契约要的是
#: `question["question_index"]`，所以这里必须显式重命名，
#: 而不是靠模糊匹配碰运气。
FIELD_CANDIDATES = {
    "image_index": ["image_index", "image_id", "image_name", "id"],
    "question_index": ["question_index", "q_index", "question_id", "qid"],
    "question": ["question", "query", "prompt"],
    "answer": ["answer", "gt_answer", "label"],
    "answer_type": ["answer_type", "type", "ans_type"],
}


def _pick(colnames, key):
    """在列名集合里按候选顺序找。返回 (真实列名, 是否严格命中)。"""
    cands = FIELD_CANDIDATES[key]
    for c in cands:
        if c in colnames:
            return c, (c == cands[0])
    for c in cands:
        for real in colnames:
            if c in real.lower():
                return real, False
    return None, False


def load_table(limit=None):
    import pyarrow.parquet as pq

    t = pq.read_table(PARQUET)
    if limit:
        t = t.slice(0, int(limit))
    return t


def inspect(limit=3):
    import pyarrow as pa  # noqa: F401

    t = load_table(limit=None)
    info = {
        "parquet": PARQUET,
        "parquet_bytes": os.path.getsize(PARQUET) if os.path.isfile(PARQUET) else None,
        "n_rows": int(t.num_rows),
        "n_cols": int(t.num_columns),
        "schema": [],
    }
    for f in t.schema:
        info["schema"].append({"name": f.name, "type": str(f.type)})

    cols = [f.name for f in t.schema]
    info["field_mapping"] = {}
    for k in FIELD_CANDIDATES:
        real, strict = _pick(cols, k)
        info["field_mapping"][k] = {"column": real, "exact_name": strict}

    # 打印前几行的结构（不打印图像字节）
    head = t.slice(0, limit).to_pylist()
    info["sample_rows"] = [_shrink(r) for r in head]

    # answer_type 分布 —— 决定 §16.3 指标怎么聚合的第一手依据
    at_col, _ = _pick(cols, "answer_type")
    if at_col:
        from collections import Counter

        info["answer_type_distribution"] = dict(
            Counter(str(r) for r in t.column(at_col).to_pylist()))
    return info


def _shrink(d, max_str=180):
    out = {}
    for k, v in d.items():
        if isinstance(v, (bytes, bytearray)):
            out[k] = "<bytes %d>" % len(v)
        elif isinstance(v, dict):
            out[k] = {kk: ("<bytes %d>" % len(vv) if isinstance(vv, (bytes, bytearray)) else _shrink_scalar(vv))
                      for kk, vv in v.items()}
        elif isinstance(v, (list, tuple)) and len(v) > 6:
            out[k] = "<list len %d>" % len(v)
        else:
            out[k] = _shrink_scalar(v)
    return out


def _shrink_scalar(v, max_str=180):
    s = v if isinstance(v, str) else repr(v)
    return s if len(s) <= max_str else s[:max_str] + "…"


def _image_bytes_and_suffix(cell):
    """HF Image 特征在 parquet 里是 struct{bytes, path}；也容忍裸 bytes。"""
    if cell is None:
        return None, None
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell), None
    if isinstance(cell, dict):
        b = cell.get("bytes")
        p = cell.get("path")
        suf = os.path.splitext(p)[1] if isinstance(p, str) else None
        return (bytes(b) if b is not None else None), suf
    return None, None


def export(limit=None, overwrite=False):
    t0 = time.time()
    t = load_table(limit=limit)
    cols = [f.name for f in t.schema]
    rows = t.to_pylist()

    mapping = {}
    for k in FIELD_CANDIDATES:
        real, strict = _pick(cols, k)
        mapping[k] = real

    img_col, _ = _pick(cols, "image_index")
    # 图像列：优先 'image'，其次含 image 的列
    image_col = "image" if "image" in cols else next(
        (c for c in cols if "image" in c.lower() and isinstance(rows[0].get(c), (dict, bytes))), None)

    missing = [k for k, v in mapping.items() if v is None]
    if image_col is None:
        missing.append("image(列)")
    if missing:
        raise SystemExit("无法定位以下字段，请先 --inspect 看 schema：%s" % missing)

    os.makedirs(IMAGES_DIR, exist_ok=True)
    questions = []
    seen_files = {}
    n_dup_bytes = 0
    used_cols = set(mapping.values()) | {image_col}

    for i, r in enumerate(rows):
        raw, suf = _image_bytes_and_suffix(r.get(image_col))
        if raw is None:
            # 极少数行可能只给了 path 没给 bytes；跳过而不是写一个坏文件
            continue
        h = hashlib.sha1(raw).hexdigest()
        if h in seen_files:
            # 同一张图被多道题共用 —— 复用文件名，避免重复落盘
            fname = seen_files[h]
            n_dup_bytes += 1
        else:
            suf = (suf or ".png").lower()
            if suf not in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
                suf = ".png"
            img_index = str(r.get(mapping["image_index"]))
            # `image_index` 实测形如 "91339.246_00000463.jpg" —— 它**自带扩展名**，
            # 而 HF Image 特征的 `path` 也带同样的扩展名。直接拼会得到
            # `xxx.jpg.jpg`（第一次跑就踩了），所以先判一次再决定加不加。
            if img_index.lower().endswith(suf) or img_index.lower().endswith(".jpeg"):
                fname = img_index
            else:
                fname = "%s%s" % (img_index, suf)
            path = os.path.join(IMAGES_DIR, fname)
            if overwrite or not os.path.isfile(path):
                with open(path, "wb") as f:
                    f.write(raw)
            seen_files[h] = fname

        q = {
            "image_index": str(r.get(mapping["image_index"])),
            "question_index": str(r.get(mapping["question_index"])),
            "question": str(r.get(mapping["question"])),
            "answer_type": str(r.get(mapping["answer_type"])),
            "answer": r.get(mapping["answer"]),
            "image_filename": fname,
        }
        extra = {k: r[k] for k in cols if k not in used_cols}
        if extra:
            q["_extra"] = json.loads(json.dumps(extra, default=str))
        questions.append(q)

    with open(ANNOTATIONS, "w", encoding="utf-8") as f:
        json.dump({"questions": questions}, f, ensure_ascii=False, indent=1)

    from collections import Counter

    # str 类里混着 yes/no 和 multi-choice，必须按**答案取值**再切一刀 ——
    # 只看 answer_type 会以为 str 是一类，而本项目的指标把 str 拆成两类
    # （engine.py:392-406）。这个拆分是「Total 聚合口径能否被反推」的关键输入。
    n_yn = sum(1 for q in questions
               if q["answer_type"] == "str" and str(q["answer"]) in ("yes", "no"))
    n_multi = sum(1 for q in questions
                  if q["answer_type"] == "str" and str(q["answer"]) not in ("yes", "no"))
    n_ct = sum(1 for q in questions if q["answer_type"] == "int")
    n_other = sum(1 for q in questions if q["answer_type"] == "float")

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_parquet": PARQUET,
        "source_rows": int(t.num_rows),
        "questions_written": len(questions),
        "unique_images": len(seen_files),
        "duplicate_image_rows": n_dup_bytes,
        "images_dir": IMAGES_DIR,
        "annotations": ANNOTATIONS,
        "field_mapping": mapping,
        # ★ 2026-09-17 实测结论，与项目原计划相冲突，故显式落盘而不是藏在注释里：
        # parquet 只有 image_index / image / q_index / question / answer / answer_type
        # 六列；HF README 的 annotations 格式也没有任何场景/相机/深度字段。
        # ⟹ **Omni3D-Bench 不提供 GT 内参**，主实验臂的横向米制尺度仍然是
        #    「模型自己猜相机」的那条不确定路径，不能宣称已被基准兜住。
        "has_3d_annotations": False,
        "note_3d": ("六列全为二维与问答字段，无相机内参/深度/物体三维框。"
                    "原「主实验臂自带 GT 相机」的说法不成立，见 read_report.json 本字段。"),
        "answer_type_distribution": dict(Counter(q["answer_type"] for q in questions)),
        "answer_type_source_distribution": dict(
            Counter(type(q["answer"]).__name__ for q in questions)),
        "metric_class_counts": {
            "numeric_count": n_ct, "numeric_other": n_other,
            "yes_no": n_yn, "multi_choice": n_multi, "total": len(questions),
        },
        "extra_columns_kept": sorted(
            {k for q in questions for k in (q.get("_extra") or {})}),
        "elapsed_s": round(time.time() - t0, 2),
    }

    # 用真实题数直接验算「Total = 按题数加权的 micro 平均」这个口径。
    # 论文给的是各方法 4 个子指标 + Total，题数已知就能逐行复算 ——
    # 这比最小二乘反推硬得多（见 evaluation/metrics.verify_total_aggregation
    # 里「为什么拟合不行」的实测记录）。
    import sys
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    try:
        from evaluation.metrics import verify_total_aggregation

        report["total_aggregation_check"] = verify_total_aggregation(counts={
            "numeric_count": n_ct, "numeric_other": n_other,
            "yes_no": n_yn, "multi_choice": n_multi,
        })
    except Exception as e:  # 允许单独跑本脚本而不依赖 evaluation/
        report["total_aggregation_check"] = {"error": "%s: %s" % (type(e).__name__, e)}

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def main():
    ap = argparse.ArgumentParser(description="Omni3D-Bench parquet → 本项目数据集目录布局")
    ap.add_argument("--inspect", action="store_true", help="只看 schema / 字段映射")
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 行")
    ap.add_argument("--overwrite", action="store_true", help="重写已存在的图片")
    args = ap.parse_args()

    if not os.path.isfile(PARQUET):
        raise SystemExit("缺少 parquet：%s\n请先跑 fetch_omni3d_bench.py" % PARQUET)

    out = inspect() if args.inspect else export(limit=args.limit, overwrite=args.overwrite)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
