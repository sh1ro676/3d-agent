#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_omni3d_bench.py -- 获取 Omni3D-Bench 原始 parquet（可断点续传）

为什么是「parquet 而不是 load_dataset」
--------------------------------------
HF 仓库 dmarsili/Omni3D-Bench 的 data/ 下只有一个文件：
    data/train-00000-of-00001.parquet   约 106.5 MB
`datasets.load_dataset` 在本机没装（且会再拉一遍同样的字节），
所以直接按 HF 的 resolve 端点把 parquet 拉下来，交给 read_omni3d_bench.py 解析。
好处：不引入 datasets 依赖；下载可续传；原始字节可在离线环境反复重放。

为什么要独立成脚本、且写报告文件
--------------------------------
本机（Windows + Agent 宿主）PowerShell 的 stdout 不回传给 Agent，
所以脚本一律把结论写进 UTF-8 报告文件，由 Agent 读取文件而不是读 stdout。

用法
----
    python dataset/builders/fetch_omni3d_bench.py                 # 下载（已存在则跳过）
    python dataset/builders/fetch_omni3d_bench.py --force         # 强制重下
    python dataset/builders/fetch_omni3d_bench.py --check         # 只探测远端元信息

环境变量
--------
    HF_ENDPOINT    默认 https://hf-mirror.com（本机实测可达；hf.co 直连不稳）
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

REPO = "dmarsili/Omni3D-Bench"
REVISION = "main"
REMOTE_PATH = "data/train-00000-of-00001.parquet"

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
DEST_DIR = os.path.join(REPO_ROOT, "dataset", "raw", "omni3d-bench")
DEST_PARQUET = os.path.join(DEST_DIR, "train-00000-of-00001.parquet")
REPORT = os.path.join(DEST_DIR, "fetch_report.json")

EXPECTED_SIZE = 106490728  # 来自 HF tree API，用作完整性下限校验
UA = "Mozilla/5.0 (omni3d-bench-fetch)"
CHUNK = 1 << 20


def endpoint():
    return (os.environ.get("HF_ENDPOINT") or "https://hf-mirror.com").rstrip("/")


def resolve_url():
    return "%s/datasets/%s/resolve/%s/%s" % (endpoint(), REPO, REVISION, REMOTE_PATH)


def remote_meta():
    """走 HF tree API 拿远端文件大小/oid，用来和本地比对。"""
    url = "%s/api/datasets/%s/tree/%s/data" % (endpoint(), REPO, REVISION)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=40) as r:
        items = json.loads(r.read().decode("utf-8"))
    for it in items:
        if it.get("path") == REMOTE_PATH:
            return it
    # 某些镜像会返回不带 path 前缀的条目
    return items[0] if items else {}


def download(force=False):
    os.makedirs(DEST_DIR, exist_ok=True)
    info = {"repo": REPO, "endpoint": endpoint(), "url": resolve_url(),
            "dest": DEST_PARQUET, "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    try:
        meta = remote_meta()
        info["remote_size"] = meta.get("size")
        info["remote_oid"] = meta.get("oid")
    except Exception as e:
        info["remote_meta_error"] = "%s: %s" % (type(e).__name__, e)
        meta = {}

    have = os.path.getsize(DEST_PARQUET) if os.path.isfile(DEST_PARQUET) else 0
    want = meta.get("size") or EXPECTED_SIZE

    if have and not force and have >= want:
        info["status"] = "skipped_already_complete"
        info["local_size"] = have
        write_report(info)
        return info

    if have and not force:
        # 续传：HF 支持 Range
        info["resume_from"] = have
        headers = {"User-Agent": UA, "Range": "bytes=%d-" % have}
        mode = "ab"
    else:
        have = 0
        headers = {"User-Agent": UA}
        mode = "wb"

    req = urllib.request.Request(resolve_url(), headers=headers)
    t0 = time.time()
    got = 0
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            info["http_status"] = getattr(r, "status", None)
            total = r.headers.get("Content-Length")
            info["content_length"] = int(total) if total and total.isdigit() else None
            with open(DEST_PARQUET, mode) as f:
                while True:
                    buf = r.read(CHUNK)
                    if not buf:
                        break
                    f.write(buf)
                    got += len(buf)
    except Exception as e:
        info["status"] = "failed"
        info["error"] = "%s: %s" % (type(e).__name__, e)
        info["downloaded_bytes"] = got
        info["elapsed_s"] = round(time.time() - t0, 2)
        write_report(info)
        return info

    size = os.path.getsize(DEST_PARQUET)
    info["local_size"] = size
    info["downloaded_bytes"] = got
    info["elapsed_s"] = round(time.time() - t0, 2)
    info["mb_per_s"] = round(size / 1e6 / max(time.time() - t0, 1e-6), 2)
    info["status"] = "ok" if size >= want else "incomplete"
    if info["status"] != "ok":
        info["hint"] = "重新运行本脚本会从断点续传"
    write_report(info)
    return info


def write_report(info):
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


def check():
    info = {"repo": REPO, "endpoint": endpoint(), "mode": "check"}
    try:
        t = urllib.request.Request(
            "%s/api/datasets/%s/tree/%s?recursive=1" % (endpoint(), REPO, REVISION),
            headers={"User-Agent": UA})
        with urllib.request.urlopen(t, timeout=30) as r:
            items = json.loads(r.read().decode("utf-8"))
        files = [x for x in items if x.get("type") == "file"]
        info["n_files"] = len(files)
        info["total_mb"] = round(sum(x.get("size", 0) for x in files) / 1e6, 1)
        info["files"] = [{"path": x["path"], "size": x.get("size")} for x in files]
        info["reachable"] = True
    except Exception as e:
        info["reachable"] = False
        info["error"] = "%s: %s" % (type(e).__name__, e)
    write_report(info)
    return info


def main():
    ap = argparse.ArgumentParser(description="获取 Omni3D-Bench parquet")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--check", action="store_true", help="只探测远端，不下载")
    args = ap.parse_args()
    info = check() if args.check else download(force=args.force)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0 if info.get("status") != "failed" and info.get("reachable", True) else 1


if __name__ == "__main__":
    sys.exit(main())
