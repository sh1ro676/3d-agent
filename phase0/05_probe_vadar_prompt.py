#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_probe_vadar_prompt.py -- 4B 级本地模型能不能驱动 VADAR 的程序合成？

为什么需要这个脚本
------------------
这是项目**最大的未知**，而它现在可以被一次性测掉：

    VADAR 用 gpt-4o 做三段式程序合成，在 Omni3D 上才拿到 40.4。
    换成 4B 的开源模型，风险不是"分数下降"，而是**流程压根跑不起来**。
    因为 VADAR 全程用正则硬解析 LLM 输出，没有任何容错：

        agents/agents.py:82   re.findall(r"<docstring>(.*?)</docstring>", ...)
        agents/agents.py:83   re.findall(r"<signature>(.*?)</signature>", ...)
        agents/agents.py:89   re.compile(r"def (\\w+)\\s*\\(.*\\):").search(sig).group(1)
        agents/agents.py:757  re.findall(r"<program>(.*?)</program>", ...)
        engine/engine.py:344  同上

    标签少一个 -> IndexError 或 method_names 为空 -> 整道题作废，而且不报可读的错。

本脚本做两件事，全部用 VADAR 源码里的**原版 prompt 和原版正则**：

  Stage A  SignatureAgent 协议测试
           喂 SIGNATURE_PROMPT，看模型是否吐出成对的
           <docstring></docstring><signature></signature>，且方法名以 _ 开头，
           且**每个签名都能被 :89 的正则提取出方法名**（见下面的"注解陷阱"）。

  Stage B  ProgramAgent 协议测试
           喂 PROGRAM_PROMPT，看模型是否吐出 <program> 代码块，
           代码能否 compile()，以及是否按 prompt 要求把答案存进 final_result。

不需要 API key
--------------
默认打本机 Ollama 的 OpenAI 兼容端点。也支持任何 OpenAI 兼容的云端端点，
两种模式跑的是同一套测试，因此结果可以直接进消融实验表。

用法
----
    # 本机 Ollama（推荐先跑这个，零成本）
    python 05_probe_vadar_prompt.py

    # 指定其它本地模型
    python 05_probe_vadar_prompt.py --model qwen3.5:9b

    # 对比多个模型（会输出一张对比表）
    python 05_probe_vadar_prompt.py --models qwen3.5:4b,qwen3.5:9b

    # 云端端点（需要 key）
    python 05_probe_vadar_prompt.py --base-url https://api.deepseek.com \\
        --model deepseek-v4-flash --api-key sk-xxxx

    # 跑全部 4 道样例题，而不是默认的 1 道
    python 05_probe_vadar_prompt.py --all-questions

输出
----
    控制台可读报告 + JSON（--report 指定，默认 vadar_prompt_probe.json）
    退出码：0 = 所有 Stage 全部通过；1 = 有硬失败（链路会崩）

只用标准库
----------
    urllib / json / re / importlib，Windows 原生 Python 直接跑，
    不需要 WSL、不需要 GPU、不需要 torch / transformers。
"""

import argparse
import importlib.util
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
VADAR_PROMPTS = os.path.join(PROJECT_ROOT, "vendor", "VADAR", "prompts")


# =====================================================================
# 0. VADAR 的真实正则契约 —— 从源码逐字复制，不要改
# =====================================================================
RE_DOCSTRING = re.compile(r"<docstring>(.*?)</docstring>", re.DOTALL)      # agents.py:82
RE_SIGNATURE = re.compile(r"<signature>(.*?)</signature>", re.DOTALL)      # agents.py:83
RE_PROGRAM = re.compile(r"<program>(.*?)</program>", re.DOTALL)            # agents.py:757
RE_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)               # predefined_modules.py:354
# agents.py:89 —— 注意结尾的 "):" 是字面量，这就是"注解陷阱"的来源
RE_DEF_NAME = re.compile(r"def (\w+)\s*\(.*\):")


def load_vadar_prompts():
    """按文件路径加载 VADAR 的原版 prompt 常量。

    signature_prompt.py 和 modules.py 里只有字符串字面量、没有任何 import，
    所以可以直接 import，不会牵扯 torch / sam2 / openai。
    """
    out = {}
    for fname, names in (
        ("signature_prompt.py", ["SIGNATURE_PROMPT", "SIGNATURE_PROMPT_CLEVR"]),
        ("program_prompt.py", ["PROGRAM_PROMPT"]),
        ("modules.py", ["MODULES_SIGNATURES"]),
    ):
        path = os.path.join(VADAR_PROMPTS, fname)
        if not os.path.isfile(path):
            raise SystemExit(
                "找不到 %s\n"
                "  项目根目录探测到的是 %s\n"
                "  请确认 vendor/VADAR/prompts/ 存在（源码目录必须叫 VADAR）"
                % (path, PROJECT_ROOT)
            )
        spec = importlib.util.spec_from_file_location("vadar_" + fname[:-3], path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for n in names:
            out[n] = getattr(mod, n)
    return out


# =====================================================================
# 1. Stage A —— SignatureAgent 协议测试
# =====================================================================
def test_signature_stage(reply):
    """按 agents.py:82-98 的真实流程解析并打分。"""
    checks = []

    docstrings = RE_DOCSTRING.findall(reply)
    signatures = RE_SIGNATURE.findall(reply)
    checks.append({
        "name": "吐出了 <docstring> 标签",
        "pass": len(docstrings) > 0,
        "detail": "找到 %d 个" % len(docstrings),
        "hard": True,
    })
    checks.append({
        "name": "吐出了 <signature> 标签",
        "pass": len(signatures) > 0,
        "detail": "找到 %d 个" % len(signatures),
        "hard": True,
    })
    checks.append({
        "name": "docstring 与 signature 数量配对",
        "pass": len(docstrings) == len(signatures) and len(signatures) > 0,
        "detail": "%d 对 %d" % (len(docstrings), len(signatures)),
        "hard": False,
    })

    # 提示词明确要求"DO NOT INCLUDE ``` tags"，泄漏不致命但会污染签名文本
    leakage = ("```" in reply)
    checks.append({
        "name": "没有泄漏 ``` 代码围栏",
        "pass": not leakage,
        "detail": "提示词明确禁止（signature_prompt.py:32）",
        "hard": False,
    })

    # ---- 注解陷阱：这是换模型最容易被击穿的一处 ----
    failed_defs = []
    for sig in signatures:
        if RE_DEF_NAME.search(sig) is None:
            failed_defs.append(sig.strip().split("\n")[-1][:90])
    if signatures:
        checks.append({
            "name": "每个签名都能被取到方法名（注解陷阱）",
            "pass": len(failed_defs) == 0,
            "detail": ("全部通过" if not failed_defs else
                       "%d/%d 个失败 -> AttributeError: 'NoneType' has no attribute 'group'"
                       % (len(failed_defs), len(signatures))),
            "hard": True,
        })
    else:
        checks.append({
            "name": "每个签名都能被取到方法名（注解陷阱）",
            "pass": False,
            "detail": "没有签名可测",
            "hard": True,
        })

    # 方法名必须以 _ 开头（signature_prompt.py:28 硬要求）
    method_names = []
    for sig in signatures:
        m = RE_DEF_NAME.search(sig)
        if m:
            method_names.append(m.group(1))
    bad_prefix = [n for n in method_names if not n.startswith("_")]
    if method_names:
        checks.append({
            "name": "方法名以 _ 开头",
            "pass": len(bad_prefix) == 0,
            "detail": ("全部合规" if not bad_prefix else
                       "违规: " + ", ".join(bad_prefix[:5])),
            "hard": False,
        })
    else:
        checks.append({
            "name": "方法名以 _ 开头",
            "pass": False,
            "detail": "一个方法名都没提取到",
            "hard": False,
        })

    # 返回注解会同时命中"注解陷阱"，这里单独检出以便诊断
    annotated = [s for s in signatures if re.search(r"\)\s*->", s)]
    checks.append({
        "name": "签名里没有返回值类型注解",
        "pass": len(annotated) == 0,
        "detail": ("无" if not annotated else
                   "%d 个带 -> 注解，会击穿 agents.py:89" % len(annotated)),
        "hard": False,
    })

    return {
        "checks": checks,
        "n_docstrings": len(docstrings),
        "n_signatures": len(signatures),
        "method_names": method_names,
        "annotated_signatures": [a.strip()[:120] for a in annotated[:5]],
        "failed_defs": failed_defs[:5],
    }


# =====================================================================
# 2. Stage B —— ProgramAgent 协议测试
# =====================================================================
def test_program_stage(reply):
    """按 agents.py:755-757 与 engine 的真实要求解析并打分。"""
    checks = []

    programs = RE_PROGRAM.findall(reply)
    checks.append({
        "name": "吐出了 <program> 标签",
        "pass": len(programs) > 0,
        "detail": "找到 %d 个" % len(programs),
        "hard": True,
    })

    body = programs[0] if programs else ""
    parses = False
    err = ""
    if body.strip():
        try:
            compile(_dedent_block(body), "<vadar_program>", "exec")
            parses = True
        except SyntaxError as e:
            err = "line %s: %s" % (e.lineno, e.msg)
    checks.append({
        "name": "程序体能通过语法编译",
        "pass": parses,
        "detail": ("compile() OK" if parses else ("SyntaxError " + err if err else "无程序体")),
        "hard": True,
    })

    has_final = bool(re.search(r"^\s*final_result\s*=", body, re.M))
    checks.append({
        "name": "按提示词要求赋值了 final_result",
        "pass": has_final,
        "detail": "program_prompt.py:55/75/79 三次强调",
        "hard": True,
    })

    leakage = ("```" in reply)
    checks.append({
        "name": "没有泄漏 ``` 代码围栏",
        "pass": not leakage,
        "detail": "VADAR 会剥掉，但会污染程序文本",
        "hard": False,
    })

    # 用到的 API 是否在预定义列表里（幻觉方法会 NameError）
    known = {"loc", "vqa", "depth", "same_object", "get_2D_object_size"}
    used = set(re.findall(r"\b(\w+)\s*\(", body))
    builtins = {
        "print", "len", "range", "int", "float", "str", "list", "dict", "set",
        "tuple", "enumerate", "zip", "sum", "min", "max", "abs", "round",
        "sorted", "any", "all", "map", "filter", "isinstance", "bool", "type",
        "getattr", "setattr", "hasattr", "format", "join", "append", "items",
        "keys", "values", "strip", "split", "lower", "upper", "replace",
        "startswith", "endswith", "find", "index", "count", "copy", "pop",
    }
    unknown = sorted(used - known - builtins)
    checks.append({
        "name": "没有调用预定义 API 之外的方法",
        "pass": len(unknown) == 0,
        "detail": ("调用的都是预定义 API" if not unknown
                   else "可疑调用: " + ", ".join(unknown[:8])),
        "hard": False,
    })

    return {
        "checks": checks,
        "n_programs": len(programs),
        "program_body": body,
        "program_chars": len(body),
        "suspicious_calls": unknown,
    }


def _dedent_block(block):
    """VADAR 的 engine 会对程序做缩进整理；这里做最小化的等价处理，
    避免因为模型多缩进了 4 个空格就误判成语法错误。"""
    import textwrap
    lines = block.split("\n")
    # 去掉首尾纯空行
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return textwrap.dedent("\n".join(lines))


# =====================================================================
# 3. HTTP（OpenAI 兼容）
# =====================================================================
def chat(base_url, model, prompt, api_key=None, timeout=600, max_tokens=2048,
         temperature=0.0, extra_body=None):
    url = base_url.rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    url += "/chat/completions"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if extra_body:
        payload.update(extra_body)

    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:600]
        raise RuntimeError("HTTP %s from %s\n%s" % (e.code, url, body))
    except urllib.error.URLError as e:
        raise RuntimeError(
            "连不上 %s\n  %s\n"
            "  如果用的是本机 Ollama：先跑 phase0/04_install_ollama.ps1" % (url, e.reason)
        )
    dt = time.time() - t0

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        raise RuntimeError("返回的不是 JSON：\n" + raw[:600])

    msg = obj["choices"][0]["message"]
    content = msg.get("content") or ""
    # 推理型模型会把思考过程放在这里；content 为空是最常见的假失败
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    usage = obj.get("usage", {}) or {}
    return {
        "content": content,
        "reasoning": reasoning,
        "latency_s": round(dt, 2),
        "usage": usage,
    }


# =====================================================================
# 4. 样例题（取自 VADAR 自己 prompt 里的示例，保证风格一致）
# =====================================================================
QUESTIONS = [
    "How many mugs are there in the dishwasher?",
    "How many plates are on the table?",
    "How many objects have the same color as the metal bowl?",
    "How many objects of the same height as the mug would you have to stack "
    "to achieve an object the same height as the cabinet?",
]


def run_one(base_url, model, api_key, prompts, question, max_tokens, timeout,
            extra_body, verbose):
    res = {"model": model, "question": question}

    # ---------- Stage A ----------
    p = prompts["SIGNATURE_PROMPT"].format(
        signatures=prompts["MODULES_SIGNATURES"], question=question
    )
    res["signature_prompt_chars"] = len(p)
    try:
        out_a = chat(base_url, model, p, api_key, timeout, max_tokens, extra_body=extra_body)
    except RuntimeError as e:
        res["error"] = str(e)
        return res

    res["signature_latency_s"] = out_a["latency_s"]
    res["signature_usage"] = out_a["usage"]
    res["signature_reply_chars"] = len(out_a["content"])
    res["signature_reasoning_chars"] = len(out_a["reasoning"])
    res["signature"] = test_signature_stage(out_a["content"])
    if verbose:
        res["signature_raw"] = out_a["content"]

    # ---------- Stage B ----------
    # 冷启动时 api_agent.api 是空的（agents.py:744-746），api_text = ""
    p2 = prompts["PROGRAM_PROMPT"].format(
        predef_signatures=prompts["MODULES_SIGNATURES"], api="", question=question
    )
    res["program_prompt_chars"] = len(p2)
    try:
        out_b = chat(base_url, model, p2, api_key, timeout, max_tokens, extra_body=extra_body)
    except RuntimeError as e:
        res["program_error"] = str(e)
        return res

    res["program_latency_s"] = out_b["latency_s"]
    res["program_usage"] = out_b["usage"]
    res["program_reply_chars"] = len(out_b["content"])
    res["program_reasoning_chars"] = len(out_b["reasoning"])
    res["program"] = test_program_stage(out_b["content"])
    if verbose:
        res["program_raw"] = out_b["content"]

    return res


def all_hard_pass(stage):
    return all(c["pass"] for c in stage["checks"] if c["hard"])


# =====================================================================
# 5. 报告
# =====================================================================
def print_stage(title, stage):
    print()
    print("  " + title)
    for c in stage["checks"]:
        mark = "PASS" if c["pass"] else ("FAIL" if c["hard"] else "warn")
        color = {"PASS": "\033[32m", "FAIL": "\033[31m", "warn": "\033[33m"}[mark]
        print("    %s%-4s\033[0m  %-42s %s" % (color, mark, c["name"], c["detail"]))


def print_report(results, questions):
    for r in results:
        print()
        print("=" * 72)
        print("  model    : " + r["model"])
        print("  question : " + r["question"])
        print("=" * 72)

        if "error" in r:
            print("  [FAIL] 请求失败：" + r["error"])
            continue

        print("  Stage A  SignatureAgent")
        print("    prompt %d chars | 回复 %d chars | 推理 %d chars | %.1f s | tokens %s"
              % (r["signature_prompt_chars"], r["signature_reply_chars"],
                 r["signature_reasoning_chars"], r["signature_latency_s"],
                 r["signature_usage"].get("total_tokens", "?")))
        print_stage("检查项", r["signature"])
        if r["signature"]["method_names"]:
            print("    提取到的方法名: " + ", ".join(r["signature"]["method_names"]))
        if r["signature"]["failed_defs"]:
            print("    \033[31m被注解陷阱击穿的签名:\033[0m")
            for s in r["signature"]["failed_defs"]:
                print("      " + s)

        if "program" not in r:
            print()
            print("  Stage B  ProgramAgent —— 未执行（" + r.get("program_error", "?") + "）")
            continue

        print()
        print("  Stage B  ProgramAgent")
        print("    prompt %d chars | 回复 %d chars | 推理 %d chars | %.1f s | tokens %s"
              % (r["program_prompt_chars"], r["program_reply_chars"],
                 r["program_reasoning_chars"], r["program_latency_s"],
                 r["program_usage"].get("total_tokens", "?")))
        print_stage("检查项", r["program"])
        body = r["program"]["program_body"]
        if body:
            print()
            print("    --- 生成的程序 ---")
            for line in body.strip().split("\n")[:25]:
                print("    | " + line)
            print("    -------------------")


def print_summary_table(results):
    print()
    print("=" * 72)
    print("  汇总")
    print("=" * 72)
    hdr = "  %-42s %-22s %s"
    print(hdr % ("model", "Stage A", "Stage B"))
    print("  " + "-" * 68)
    for r in results:
        if "error" in r:
            print(hdr % (r["model"][:42], "请求失败", "-"))
            continue
        a = "PASS" if all_hard_pass(r["signature"]) else "FAIL"
        b = "-"
        if "program" in r:
            b = "PASS" if all_hard_pass(r["program"]) else "FAIL"
        print(hdr % (r["model"][:42], a, b))
    print()
    print("  PASS 判定只看 hard 项（少一个标签就会让 VADAR 直接崩的那些）。")


def main():
    ap = argparse.ArgumentParser(
        description="用 VADAR 原版 prompt 与正则测试模型是否遵守其标签契约",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--base-url", default="http://127.0.0.1:11434",
                    help="OpenAI 兼容端点（默认本机 Ollama）")
    ap.add_argument("--model", default="qwen3.5:4b")
    ap.add_argument("--models", default=None,
                    help="逗号分隔的多个模型，覆盖 --model，用同一端点做对比")
    ap.add_argument("--api-key", default=os.environ.get("VADAR_API_KEY"))
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--all-questions", action="store_true",
                    help="跑全部 4 道样例题（默认只跑第 1 道）")
    ap.add_argument("--verbose", action="store_true", help="把模型原始回复也写进 JSON")
    ap.add_argument("--report", default="vadar_prompt_probe.json")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",")] if args.models else [args.model]
    questions = QUESTIONS if args.all_questions else QUESTIONS[:1]

    print()
    print("=" * 72)
    print(" VADAR 程序合成协议探针 —— 4B 级模型能不能扛住？")
    print("=" * 72)
    print("  endpoint : " + args.base_url)
    print("  models   : " + ", ".join(models))
    print("  questions: %d" % len(questions))
    print("  prompts  : %s（VADAR 原版，未经改写）" % VADAR_PROMPTS)

    prompts = load_vadar_prompts()
    print("  已加载   : SIGNATURE_PROMPT(%d chars)  MODULES_SIGNATURES(%d chars)  "
          "PROGRAM_PROMPT(%d chars)"
          % (len(prompts["SIGNATURE_PROMPT"]),
             len(prompts["MODULES_SIGNATURES"]),
             len(prompts["PROGRAM_PROMPT"])))

    results = []
    for model in models:
        for q in questions:
            print()
            print("  >> %s  |  %s" % (model, q))
            r = run_one(args.base_url, model, args.api_key, prompts, q,
                        args.max_tokens, args.timeout, None, args.verbose)
            results.append(r)
            if "error" in r:
                print("     \033[31m请求失败\033[0m: " + r["error"].split("\n")[0])
            else:
                a = "PASS" if all_hard_pass(r["signature"]) else "FAIL"
                b = "-"
                if "program" in r:
                    b = "PASS" if all_hard_pass(r["program"]) else "FAIL"
                print("     Stage A=%s  Stage B=%s  (%.1fs / %s)"
                      % (a, b, r["signature_latency_s"],
                         ("%.1fs" % r["program_latency_s"]) if "program" in r else "-"))

    print_report(results, questions)
    print_summary_table(results)

    # ---------- 结论 ----------
    ok = 0
    total = 0
    for r in results:
        if "error" in r:
            total += 2
            continue
        total += 2
        if all_hard_pass(r["signature"]):
            ok += 1
        if "program" in r and all_hard_pass(r["program"]):
            ok += 1

    print()
    print("=" * 72)
    if ok == total:
        print(" \033[32m结论：链路协议全部通过。\033[0m")
        print(" 该模型可以进入 Pipeline —— VADAR 的标签契约守住了。")
        print(" 下面要测的就不再是「能不能跑」，而是「答得对不对」（准确率）。")
    else:
        print(" \033[33m结论：%d/%d 项硬检查未通过。\033[0m" % (total - ok, total))
        print(" 链路会在对应位置直接崩（AttributeError / IndexError / 空程序）。")
        print(" 处理顺序建议：")
        print("   1. 先看是不是 max_tokens 太小 -> 提高 --max-tokens 重试")
        print("   2. 再看显存是不是装不下 -> 换更小的量化或更小的模型")
        print("   3. 若模型本身不守格式 -> 这正是 QLoRA 要解决的问题，记录为训练目标")
        print("   4. 若是「注解陷阱」单项失败 -> 用 03_vadar_llm_bridge.py 的容错层兜住")
    print("=" * 72)

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump({
            "endpoint": args.base_url,
            "models": models,
            "questions": questions,
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print()
    print("  报告已写入 " + os.path.abspath(args.report))

    return 0 if ok == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
