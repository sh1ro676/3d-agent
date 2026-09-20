#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""runner.py —— 跑一个实验臂，产出 results/<arm>/*。

这是 §17 里预留的 `evaluation/runner.py`：把 VADAR 的四段流水线
（Signature → API → Program → Engine）串起来，加上**VADAR 缺的三样东西**：

    ① 可复现的题集选择   —— 原版 `random.sample(questions, num_api_questions)`
                            没有种子，同一份数据两次跑出不同题 → 数字无法归因（bug ③）
    ② 归一化结果文件     —— 原版只留 execution.json / execution.csv / results.txt，
                            没有模型指纹、没有环境指纹、没有成本/延迟
    ③ 失败不致命         —— 单题炸掉不该让整轮实验作废

关于「保真」与「修 bug」的边界（这是本文件最重要的设计决定）
------------------------------------------------------------
臂 A 的定义是「VADAR 原始流水线」。所以默认行为**照抄原版**，包括 bug：

    --fix-signature-prompt   默认关 → SignatureAgent.dataset 恒为 "clevr"（bug ①），
                                     omni3d 用了 CLEVR 的 signature prompt

**唯一无法照抄的是题集选择**：原版的随机性让「两次跑的结果不一样」，
那不是保真，那是没有基线。所以这里替换成种子化的 `random.Random(seed).sample`，
并把**被选中的题目清单单独落盘**（`subset.json`），后续任何臂都可以用
`--subset-file` 复现同一批题。这是对原版的**显式偏离**，报告里必须写。

三种题集选择方式（`--select`）：

    vadarspec      questions[:N]     —— 原版 evaluate.py:29 的行为，按文件顺序取前 N
    seeded-sample  种子化均匀抽样     —— 需要跨图片覆盖时用
    stratify       按四类指标分层配比 —— 子集太小（<40 题）时避免某类为 0 导致指标为 None

无论哪种，都会把选中题目与四类计数写进 subset.json —— 小样本跑出来的
分项指标如果某一类为 0，报告里必须能看出来，否则会被误读成「该类得分为 0」。

用法
----
    # 只做前检查（选子集、验契约、装适配层），不调 LLM、不加载模型
    python evaluation/runner.py --plan

    # 真跑：2 题 smoke（需要 VADAR_API_KEY）
    python evaluation/runner.py --arm A --num-questions 2 --num-api-questions 2

    # 复用某次跑过的题集
    python evaluation/runner.py --arm A --subset-file results/A/subset.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from datetime import datetime

# ---------------------------------------------------------------------
# 环境变量必须在 **import torch 之前** 设置好 —— 否则 huggingface_hub
# 已经在导入期读走了默认值，再设就晚了（Phase 0 踩过的坑）。
# ---------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)

# ---------------------------------------------------------------------
# 后端配置：先读 configs/llm_backend.env，再让下面的 DEFAULTS 补缺。
#
# 为什么需要一个文件：`phase0/06_deepseek_setup.ps1` 设的是**会话级**变量，
# 不落盘、也传不进任何别的进程 —— 2026-09-17 实跑 arm A 就是卡在「缺 key」。
# 详见仓库根 `vadar_env.py` 的模块注释（含全部解析语义）。
#
# 与下面那段环境变量同一个理由：**必须在 import torch 之前完成**。
# ---------------------------------------------------------------------
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import vadar_env  # noqa: E402

_ABSENT = object()          # 「这个键原来根本不存在」的哨兵
_APPLIED_BY_FILE = set()    # 由配置文件写进 os.environ 的键
_ENV_ORIG = {}              # 它们被覆盖前的原值，用于换文件时撤销


def use_env_file(path: str) -> dict:
    """加载配置文件（可中途换路径），返回**脱敏**报告。

    换路径时会把上一份文件写进去、而新文件里没有的键**撤销**回原值 ——
    否则「换了文件却还在用旧配置」会静默发生，这类残留比配置写错更难查。

    实现上先在 `os.environ` 的**副本**里加载，再按 diff 落到真实环境：
    这样「哪些键是文件写的」是算出来的，不是猜的。
    """
    global ENV_FILE_PATH, ENV_FILE_REPORT, ENV_FILE_ERROR
    ENV_FILE_PATH = path
    ENV_FILE_ERROR = None
    for k in _APPLIED_BY_FILE:                      # 撤销上一份文件
        orig = _ENV_ORIG.pop(k, _ABSENT)
        if orig is _ABSENT:
            os.environ.pop(k, None)
        else:
            os.environ[k] = orig
    _APPLIED_BY_FILE.clear()

    scratch = dict(os.environ)
    try:
        rep = vadar_env.load_env_file(path, environ=scratch)
    except vadar_env.EnvFileError as e:
        # 写坏的配置文件**不能**被容错跳过：拿默认值跑完会产出一份
        # 「看起来正常、配置不对」的结果。留个空壳报告交给调用方转 fatal。
        ENV_FILE_ERROR = str(e)
        rep = {"path": str(path or ""), "exists": bool(path and os.path.isfile(path)),
               "error": ENV_FILE_ERROR, "keys": [], "applied": [], "kept_from_env": [],
               "conflicts": [], "unknown_keys": [], "empty_values": [], "secrets": {},
               "config_sha256": None}

    for k in rep.get("applied") or []:
        _ENV_ORIG.setdefault(k, os.environ.get(k, _ABSENT))
        os.environ[k] = scratch[k]
        _APPLIED_BY_FILE.add(k)
    ENV_FILE_REPORT = rep
    return rep


# 导入期就加载一次：这样 `--plan`、以及任何在 main() 里就用到配置的分支
# 都已经有值。`ENV_FILE_ERROR` 只在 use_env_file() 开头被清空，
# 所以导入期的错误会一直留到 main() 里被转成 fatal，不会被静默吞掉。
ENV_FILE_PATH = os.environ.get("VADAR_ENV_FILE") or vadar_env.default_path(PROJECT_ROOT)
ENV_FILE_REPORT = {}
ENV_FILE_ERROR = None
use_env_file(ENV_FILE_PATH)


def _api_key_source() -> str:
    """key 是从哪来的 —— 「配置文件 / 进程环境 / 根本没有」。

    这三个值必须能在报告里区分：上一轮的教训是「配置看起来对」与
    「配置真的生效了」是两件事，而只有后者能解释实验数字。
    """
    if not os.environ.get("VADAR_API_KEY"):
        return "missing"
    if "VADAR_API_KEY" in (ENV_FILE_REPORT.get("applied") or []):
        return "env_file"
    return "process_env"

DEFAULTS = {
    "HF_HOME": os.path.join(PROJECT_ROOT, ".cache", "huggingface"),
    "HF_ENDPOINT": "https://hf-mirror.com",
    "HF_HUB_DISABLE_XET": "1",
    "HF_HUB_OFFLINE": "1",       # 权重已在本地；离线可避免每次跑都去探网
    "VADAR_GDINO_DIR": os.path.join(PROJECT_ROOT, ".cache", "models", "grounding-dino-tiny"),
    "VADAR_CALL_LOG": os.path.join(PROJECT_ROOT, "logs", "vadar_llm_calls.jsonl"),
    "VADAR_TEMPERATURE": "0.2",
    "VADAR_MAX_TOKENS": "8192",
    # 关掉思考模式，让 temperature 真的生效（见 §7.4：思考开着时
    # temperature 静默失效，不报错也没效果）。
    "VADAR_EXTRA_BODY": '{"thinking": {"type": "disabled"}}',
}


def _apply_env_defaults() -> dict:
    applied = {}
    for k, v in DEFAULTS.items():
        if not os.environ.get(k):
            os.environ[k] = v
            applied[k] = v
    return applied


_ENV_APPLIED = _apply_env_defaults()

DEFAULT_ANNOTATIONS = os.path.join(
    PROJECT_ROOT, "dataset", "raw", "omni3d-bench", "unpacked", "annotations.json")
DEFAULT_IMAGES = os.path.join(
    PROJECT_ROOT, "dataset", "raw", "omni3d-bench", "unpacked", "images")
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")


# =====================================================================
# 1. 题集选择
# =====================================================================
def metric_class(q: dict) -> str:
    at = str(q.get("answer_type") or "").strip().lower()
    if at == "int":
        return "numeric_count"
    if at == "float":
        return "numeric_other"
    if at == "str":
        return "yes_no" if str(q.get("answer")) in ("yes", "no") else "multi_choice"
    return "unknown"


def select_subset(questions: list, n: int, mode: str, seed: int) -> list:
    """选 N 道题。三种模式都必须是**确定性的**（这是相对原版的核心修复）。"""
    # 先校验 mode：放在「n 覆盖全集」的提前返回之前。
    # 否则 `select_subset(pool, len(pool), "拼错了", 0)` 会安静地走原版路径，
    # 拼错的参数变成「看起来正常」的结果。
    if mode not in ("vadarspec", "seeded-sample", "stratify"):
        raise ValueError("未知的 --select 模式: %r" % (mode,))
    if n is None or n < 0 or n >= len(questions):
        return list(questions)
    if mode == "vadarspec":
        # 原版 evaluate.py:29 `questions = questions_data["questions"][:num_questions]`
        return list(questions[:n])
    if mode == "seeded-sample":
        rng = random.Random(seed)
        return rng.sample(questions, n)
    if mode == "stratify":
        # 按四类指标的实测占比做配额，再在类内按固定顺序取 ——
        # 固定顺序而非再抽一次，是为了让「同样的 seed + 同样的 N」结果稳定到题。
        by_cls = {}
        for q in questions:
            by_cls.setdefault(metric_class(q), []).append(q)
        total = len(questions)
        rng = random.Random(seed)
        quota = {}
        assigned = 0
        for cls, items in by_cls.items():
            k = int(round(n * len(items) / total))
            quota[cls] = max(0, min(k, len(items)))
            assigned += quota[cls]
        # 配额取整会差几道，用最大余数补
        order = sorted(by_cls, key=lambda c: -(n * len(by_cls[c]) / total - quota[c]))
        i = 0
        while assigned < n and order:
            c = order[i % len(order)]
            if quota[c] < len(by_cls[c]):
                quota[c] += 1
                assigned += 1
            i += 1
            if i > 10 * len(order):
                break
        out = []
        for cls in sorted(by_cls):
            pool = by_cls[cls]
            out.extend(rng.sample(pool, quota[cls]) if quota[cls] else [])
        return out
    raise AssertionError("unreachable: mode 已在函数入口校验")


def subset_record(questions: list, annotated: list, mode: str, seed: int,
                  num_api_questions: int, api_pick: list) -> dict:
    from collections import Counter

    return {
        "mode": mode,
        "seed": seed,
        "n_requested": len(questions),
        "n_selected": len(annotated),
        "metric_class_counts": dict(Counter(metric_class(q) for q in annotated)),
        "answer_type_counts": dict(Counter(str(q.get("answer_type")) for q in annotated)),
        "n_unique_images": len({q["image_index"] for q in annotated}),
        "annotation_pool_size": None,   # 调用方填
        "num_api_questions": num_api_questions,
        "selected": [
            {"image_index": q["image_index"], "question_index": q["question_index"],
             "metric_class": metric_class(q), "answer_type": q.get("answer_type"),
             "question": q.get("question")}
            for q in annotated
        ],
        "api_subset": [
            {"image_index": q["image_index"], "question_index": q["question_index"]}
            for q in api_pick
        ],
    }


# =====================================================================
# 2. 前检查
# =====================================================================
def precheck(annotations_path: str, images_path: str) -> dict:
    """不加载模型、不联网，把能验的都验掉。"""
    info = {"annotations": annotations_path, "images": images_path, "problems": []}
    if not os.path.isfile(annotations_path):
        info["problems"].append("annotations.json 不存在：%s" % annotations_path)
        return info
    with open(annotations_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    qs = data.get("questions") or []
    info["n_questions"] = len(qs)
    if not qs:
        info["problems"].append("questions 为空")
        return info

    need = ("image_index", "question_index", "question", "answer_type", "answer",
            "image_filename")
    missing_fields = [k for k in need if k not in qs[0]]
    if missing_fields:
        info["problems"].append("题目缺字段（VADAR 硬约定）：%s" % missing_fields)
    info["question_fields"] = sorted(qs[0].keys())

    fnames = {q.get("image_filename") for q in qs if q.get("image_filename")}
    info["n_unique_image_filenames"] = len(fnames)
    absent = [f for f in fnames if not os.path.isfile(os.path.join(images_path, f))]
    info["n_missing_image_files"] = len(absent)
    if absent:
        info["problems"].append("有 %d 张图片文件缺失，例如 %s" % (len(absent), absent[:3]))
    info["ok"] = not info["problems"]
    return info


# =====================================================================
# 3. 主流程
# =====================================================================
def run_stages(args, questions, api_pick, out_dir, report):
    """四段流水线。每段单独 try —— 一段失败不等于整轮实验作废。"""
    from agents.agents import APIAgent, ProgramAgent, SignatureAgent
    from engine.engine import Engine
    from prompts.modules import MODULES_SIGNATURES, MODULES_SIGNATURES_CLEVR

    os.makedirs(out_dir, exist_ok=True)
    stages = report["stages"] = {}
    # 注意：三个 Agent 的构造签名里**没有** api_key_path
    # （SignatureAgent.__init__ 只吃 predef_signatures/model_name/write_results/headers，
    #  APIAgent 与 ProgramAgent 同理），api_key_path 是 Engine 的构造参数。
    # 三个 Agent 内部走 Agent.__init__ 的默认 "./api.key"，而那个默认值会被
    # 桥接的 PatchedGenerator 忽略 —— 所以这里不需要、也不能传。
    common = dict(model_name="gpt-4o", write_results=True)

    t0 = time.time()
    try:
        if args.dataset == "clevr":
            sig_agent = SignatureAgent(MODULES_SIGNATURES_CLEVR, **common)
        else:
            sig_agent = SignatureAgent(MODULES_SIGNATURES, **common)

        # bug ①：SignatureAgent.__init__ 走的是 Agent.__init__(model, write)
        # 两参数版本，dataset 拿到默认值 "clevr"，于是 get_signatures 永远用
        # SIGNATURE_PROMPT_CLEVR（agents.py:123-126）。默认保持原样；
        # --fix-signature-prompt 时改成真实数据集名。
        report["signature_agent_dataset"] = sig_agent.dataset
        if args.fix_signature_prompt:
            sig_agent.dataset = args.dataset
            report["signature_agent_dataset"] = args.dataset
            report["signature_prompt_fix_applied"] = True

        sig_agent.get_signatures(api_pick, args.images, out_dir)
        stages["signatures"] = {"ok": True, "s": round(time.time() - t0, 2),
                                "n_api_questions": len(api_pick),
                                "n_methods": len(sig_agent.method_names),
                                "methods": list(sig_agent.method_names)}
    except Exception as e:
        stages["signatures"] = {"ok": False, "s": round(time.time() - t0, 2),
                                "error": "%s: %s" % (type(e).__name__, e),
                                "traceback": traceback.format_exc()[-2000:]}
        return None

    t0 = time.time()
    try:
        api_agent = APIAgent(sig_agent, args.dataset, **common)
        api_agent.get_api_implementations(out_dir)
        stages["api"] = {"ok": True, "s": round(time.time() - t0, 2),
                         "n_api": len(api_agent.api)}
    except Exception as e:
        stages["api"] = {"ok": False, "s": round(time.time() - t0, 2),
                         "error": "%s: %s" % (type(e).__name__, e),
                         "traceback": traceback.format_exc()[-2000:]}
        return None

    t0 = time.time()
    try:
        prog_agent = ProgramAgent(api_agent, dataset=args.dataset, **common)
        prog_agent.get_programs(questions, args.images, out_dir)
        n_prog = sum(1 for p in prog_agent.programs if p.get("program"))
        stages["programs"] = {"ok": True, "s": round(time.time() - t0, 2),
                              "n_programs": len(prog_agent.programs),
                              "n_nonempty": n_prog}
    except Exception as e:
        stages["programs"] = {"ok": False, "s": round(time.time() - t0, 2),
                              "error": "%s: %s" % (type(e).__name__, e),
                              "traceback": traceback.format_exc()[-2000:]}
        return None

    t0 = time.time()
    try:
        engine = Engine(api_agent.api, api_key_path="./api.key",
                        results_folder_path=out_dir, models_path=args.models_path,
                        dataset=args.dataset)
        engine.execute_programs(prog_agent.programs, questions, args.images,
                                oracle=False, scenes_json_path="")
        stages["execute"] = {"ok": True, "s": round(time.time() - t0, 2)}
        report["execution_json"] = os.path.join(
            out_dir, "program_execution", "execution.json")
    except Exception as e:
        stages["execute"] = {"ok": False, "s": round(time.time() - t0, 2),
                             "error": "%s: %s" % (type(e).__name__, e),
                             "traceback": traceback.format_exc()[-2000:]}
    return stages.get("execute", {}).get("ok") and report.get("execution_json")


def collect_records(execution_json_path: str) -> list:
    """把 VADAR 的 execution.json 摊平成 metrics.compute_metrics 要的记录。

    结构（engine.py:296-308）： [{ "execution": {question:{...}, program, answer} }, ...]
    """
    with open(execution_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    recs = []
    for item in data:
        blk = item.get("execution") or item.get("oracle_execution")
        if not blk:
            continue
        q = blk.get("question") or {}
        recs.append({
            "answer_type": q.get("answer_type"),
            "ground_truth": q.get("answer"),
            "prediction": blk.get("answer"),
            "image_index": q.get("image_index"),
            "question_index": q.get("question_index"),
            "question": q.get("question"),
            "has_program": bool(blk.get("program")),
        })
    return recs


def summarize_call_log(path: str) -> dict:
    """复用 bridge 的汇总，但失败时不要让整轮实验挂掉。"""
    try:
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "phase0"))
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "vadar_llm_bridge", os.path.join(PROJECT_ROOT, "phase0", "03_vadar_llm_bridge.py"))
        bridge = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bridge)
        return bridge.summarize_log(path)
    except Exception as e:
        return {"error": "%s: %s" % (type(e).__name__, e), "path": path}


def main():
    ap = argparse.ArgumentParser(description="跑一个实验臂")
    ap.add_argument("--arm", default="A")
    ap.add_argument("--dataset", default="omni3d", choices=["omni3d", "clevr", "gqa"])
    ap.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    ap.add_argument("--images", default=DEFAULT_IMAGES)
    ap.add_argument("--models-path", default=os.path.join(PROJECT_ROOT, ".cache", "models"))
    ap.add_argument("--num-questions", type=int, default=20)
    ap.add_argument("--num-api-questions", type=int, default=3,
                    help="原版默认 10；小样本时 3 更省。注意 API 实现是**全量生成**的，"
                         "这个数只决定签名阶段的样例数。")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--select", default="vadarspec",
                    choices=["vadarspec", "seeded-sample", "stratify"])
    ap.add_argument("--subset-file", default=None, help="复用已落盘的题集")
    ap.add_argument("--results-root", default=RESULTS_ROOT)
    ap.add_argument("--fix-signature-prompt", action="store_true",
                    help="修 VADAR bug ①（SignatureAgent.dataset 恒为 clevr）")
    ap.add_argument("--plan", action="store_true",
                    help="只做前检查 + 选子集 + 装适配层，不调 LLM、不加载模型")
    ap.add_argument("--env-file", default=None,
                    help="后端配置文件，默认 configs/llm_backend.env"
                         "（也可用环境变量 VADAR_ENV_FILE 指到别处）")
    ap.add_argument("--no-env-file", action="store_true",
                    help="完全忽略配置文件，只用进程环境 + 代码默认值")
    args = ap.parse_args()

    # 换配置文件（或 --no-env-file）必须在这里落地：torch 还没被导入，
    # 所以「必须在 import torch 之前生效」这个约束仍然满足。
    wanted = None if args.no_env_file else (args.env_file or ENV_FILE_PATH)
    if wanted != ENV_FILE_PATH:
        use_env_file(wanted)

    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    report = {"arm": args.arm, "started": started, "argv": sys.argv[1:],
              "env_applied_by_runner": _ENV_APPLIED, "dataset": args.dataset,
              # 脱敏的后端配置快照（密钥只有 mask/fingerprint）——没有它，
              # 「这次到底发了什么配置」只能靠回忆，实验无法独立追溯。
              "llm_env": ENV_FILE_REPORT,
              "api_key_source": _api_key_source()}

    # 配置文件写坏了 → 整轮停住。**不退回默认值继续跑**：
    # 那会产出一份看起来正常、配置其实不对的结果，比直接失败昂贵得多。
    if ENV_FILE_ERROR:
        report["fatal"] = "配置文件内容有错，已停住（不退回默认值）：%s" % ENV_FILE_ERROR
        _write_report(args, report, kind="fatal")
        print(json.dumps({"fatal": report["fatal"], "llm_env": ENV_FILE_REPORT},
                         ensure_ascii=False, indent=2))
        return 2

    # ---- 1. 前检查 ----
    pc = precheck(args.annotations, args.images)
    report["precheck"] = pc
    if not pc.get("ok"):
        report["fatal"] = "前检查未通过"
        _write_report(args, report, kind="fatal")
        print(json.dumps(pc, ensure_ascii=False, indent=2))
        return 2

    with open(args.annotations, "r", encoding="utf-8") as f:
        pool = (json.load(f).get("questions") or [])

    # ---- 2. 选子集 ----
    if args.subset_file:
        with open(args.subset_file, "r", encoding="utf-8") as f:
            sub = json.load(f)
        wanted = {(d["image_index"], d["question_index"]) for d in sub["selected"]}
        chosen = [q for q in pool
                  if (q["image_index"], q["question_index"]) in wanted]
        api_pick = [q for q in pool if (q["image_index"], q["question_index"])
                    in {(d["image_index"], d["question_index"]) for d in sub.get("api_subset", [])}]
        sub["replayed_from"] = args.subset_file
        report["subset"] = sub
    else:
        chosen = select_subset(pool, args.num_questions, args.select, args.seed)
        rng = random.Random(args.seed)
        k = max(0, min(args.num_api_questions, len(chosen)))
        # 原版是 random.sample(questions, num_api_questions)（无种子）。
        # 这里同一个调用，但用**有种子**的 Random 实例 —— 这是唯一的偏离。
        api_pick = rng.sample(chosen, k) if k else []
        sub = subset_record(chosen, chosen, args.select, args.seed,
                            args.num_api_questions, api_pick)
        sub["annotation_pool_size"] = len(pool)
        sub["deviation_from_original"] = (
            "原版用无种子的 random.sample 抽 api_questions（bug ③，不可复现）；"
            "本运行器改用 random.Random(seed).sample，并把选中清单落盘。")
        report["subset"] = sub

    if not chosen:
        report["fatal"] = "选不出题目（检查 --num-questions / --subset-file）"
        _write_report(args, report, kind="fatal")
        return 2

    # ---- 3. 装适配层 ----
    sys.path.insert(0, PROJECT_ROOT)
    from evaluation import vadar_compat

    try:
        inst = vadar_compat.install_vadar(verbose=False)
        # shim 对象不能序列化，换成描述字典
        inst["steps"]["alarm"]["shim"] = inst["steps"]["alarm"]["shim"].describe()
        inst["steps"]["stubs"] = {"count": inst["steps"]["stubs"]["count"]}
        report["adapter"] = inst
        report["adapter_ok"] = True
    except Exception as e:
        report["adapter_ok"] = False
        report["adapter_error"] = "%s: %s" % (type(e).__name__, e)
        report["adapter_traceback"] = traceback.format_exc()[-3000:]

    if args.plan:
        report["mode"] = "plan"
        report["plan_summary"] = {
            "n_questions": len(chosen),
            "n_api_questions": len(api_pick),
            "metric_class_counts": sub["metric_class_counts"],
            "n_unique_images": sub["n_unique_images"],
            "api_key_present": bool(os.environ.get("VADAR_API_KEY")),
            "api_key_source": _api_key_source(),
            "env_file": {
                "path": ENV_FILE_REPORT.get("path"),
                "exists": ENV_FILE_REPORT.get("exists"),
                "n_keys": len(ENV_FILE_REPORT.get("keys") or []),
                "config_sha256": ENV_FILE_REPORT.get("config_sha256"),
            },
            "next_command": (sys.argv[0] + " --arm %s --subset-file <本次落盘的 subset.json>"
                             % args.arm),
        }
        _write_report(args, report, kind="plan")
        out = dict(report)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0 if report.get("adapter_ok") else 3

    if not os.environ.get("VADAR_API_KEY"):
        p = ENV_FILE_REPORT.get("path")
        line = vadar_env.locate(p, "VADAR_API_KEY") if p else None
        if line:
            where = ("把 key 填进 %s 的第 %d 行（打开：notepad \"%s\"）"
                     % (p, line, p))
        elif p:
            where = "在 %s 里补一行 VADAR_API_KEY=sk-xxx" % p
        else:
            where = "设置环境变量 VADAR_API_KEY"
        report["fatal"] = ("缺少 VADAR_API_KEY：既不在配置文件里，也不在进程环境里。%s。"
                           "配置文件状态：%s"
                           % (where, vadar_env.describe(ENV_FILE_REPORT)))
        report["note"] = ("本次失败**没有**覆盖 latest_run.json；它写到了 "
                          "latest_failed.json。上一次成功的实验记录仍是完整的。")
        _write_report(args, report, kind="fatal")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2

    # ---- 4. 跑四段 ----
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(args.results_root, args.arm, ts)
    os.makedirs(run_dir, exist_ok=True)
    report["run_dir"] = run_dir
    exec_path = run_stages(args, chosen, api_pick, run_dir, report)
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    # 生成程序的路径消毒计数。「修了几个文件、几处」是**实验证据**不是日志：
    # 它直接决定这批数字是在「修好的」环境里跑出来的，还是踩了 VADAR 的
    # Windows 路径转义 bug（见 vadar_compat §5）。
    try:
        from evaluation import vadar_compat as _vc
    except ImportError:
        import vadar_compat as _vc
    report["program_sanitizer"] = dict(_vc.SANITIZER_STATS)

    # 删除策略也要进记录。「怎么删的」是环境指纹的一部分：VADAR 在
    # `agents.py:383` 用 `shutil.rmtree` 清自己的工作目录，而宿主的安全删除
    # 守卫会在目录里文件数超阈值时要求人工确认 —— 子进程无法确认，直接 exit=1。
    # 不记下来的话，没人能回答「这批数字是在什么删除策略下跑出来的」。
    report["deletion_policy"] = {
        "host_safe_delete_enabled":
            os.environ.get("CODEBUDDY_SAFE_DELETE_ENABLED", "1") != "0",
        "why": "VADAR 会 rmtree 自己的 exec 目录；实测正常方法 4 个文件，"
               "出错重试的那个累积到 51（> 宿主阈值 50）",
    }

    # ---- 5. 计分 ----
    if exec_path and os.path.isfile(exec_path):
        from evaluation.metrics import compute_metrics, verify_total_aggregation

        recs = collect_records(exec_path)
        report["metrics"] = compute_metrics(recs)
        report["metrics"]["records"] = recs
        report["aggregation_rule"] = verify_total_aggregation()
    else:
        report["metrics"] = {"error": "没有可用的 execution.json"}

    report["llm_calls"] = summarize_call_log(
        os.environ.get("VADAR_CALL_LOG", DEFAULTS["VADAR_CALL_LOG"]))
    _write_report(args, report, run_dir, kind="run")
    print(json.dumps({k: v for k, v in report.items() if k != "adapter"},
                     ensure_ascii=False, indent=2)[:6000])
    return 0 if report.get("metrics", {}).get("submetrics") else 1


def _write_report(args, report, run_dir=None, kind="run"):
    """同时写两份：一份在 results/<arm>/ 下当稳定入口，一份随 run 目录存档。

    入口文件名按 `kind` 分流 —— 这条分流是**踩出来的**，不是设计洁癖：

        2026-09-17 实测：不带 --plan 直接跑（那时 env 里没有 key），
        运行器在「缺 key」这一步 return 2，但它**照旧写了 latest_run.json**，
        于是上一次 --plan 的记录（里面有 plan_summary / next_command / 模型指纹）
        被一条 fatal 记录覆盖掉了。下一次想照着 next_command 跑，
        打开 latest_run.json 看到的是自己刚刚的失败。

    ⟹ 语义定死：`latest_run.json` **只属于真正进过流水线的 run**；
       计划态与致命态各有自己的稳定入口，互不覆盖。
    """
    os.makedirs(os.path.join(args.results_root, args.arm), exist_ok=True)
    # 让产物**自述**它是哪一类 —— 打开 latest_failed.json 时不看文件名也知道。
    # 用 setdefault：plan 分支已经自己把 mode 设成 "plan"，不覆盖它。
    report.setdefault("mode", kind)
    blob = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    fname = {"run": "latest_run.json",
             "plan": "latest_plan.json",
             "fatal": "latest_failed.json"}[kind]
    stable = os.path.join(args.results_root, args.arm, fname)
    with open(stable, "w", encoding="utf-8") as f:
        f.write(blob)
    sub_path = os.path.join(args.results_root, args.arm, "subset.json")
    # fatal 态**不写** subset.json：那份题集根本没跑过，覆盖掉上一次真跑过的
    # 题集会让「复现同一批题」这条链路指向一个从未执行的选择。
    if kind != "fatal" and report.get("subset") and not args.subset_file:
        with open(sub_path, "w", encoding="utf-8") as f:
            json.dump(report["subset"], f, ensure_ascii=False, indent=2)
    if run_dir:
        with open(os.path.join(run_dir, "arm_report.json"), "w", encoding="utf-8") as f:
            f.write(blob)


if __name__ == "__main__":
    sys.exit(main())
