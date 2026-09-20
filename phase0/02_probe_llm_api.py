#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_probe_llm_api.py -- VADAR 链路接入自检探针（零第三方依赖）

目的
----
在动 WSL / GPU / VADAR 之前，先用纯 HTTP 验证一件事：

    "把 VADAR 的 LLM 后端换成国产模型，这条链路到底走不走得通？"

脚本按 VADAR 源码里**真实的**正则契约去测，任何一条不通过，VADAR 都会在
运行时 AttributeError / IndexError，而不是给你一个可读的报错。

VADAR 输出的四类标签（全部是硬解析，无容错）
--------------------------------------------
  <docstring>...</docstring>       agents/agents.py:82
  <signature>...</signature>       agents/agents.py:83
  <implementation>...</implementation>  agents/agents.py:298
  <program>...</program>           agents/agents.py:757 / engine/engine.py:344
  <answer>...</answer>             engine/predefined_modules.py:354   <-- [0] 直接索引！

两个特别容易踩的点，本脚本会专门测：
  A) engine/predefined_modules.py:354 是
         re.findall(r"<answer>(.*?)</answer>", output, re.DOTALL)[0].lower()
     列表取 [0]，模型若不吐 <answer> 标签 -> IndexError，整个问题作废。
  B) agents/agents.py:89 / :352 / :483 / :668 用
         re.compile(r"def (\\w+)\\s*\\(.*\\):").search(sig).group(1)
     提取方法名。注意正则里要求出现字面量 "):"，
     所以 `def object_height(image, bbox) -> float:` 这种**带返回值类型注解**的
     签名会匹配失败 -> None.group(1) -> AttributeError。
     gpt-4o 通常不写返回值注解，换模型后是否仍守规矩，必须实测。

用法
----
    python 02_probe_llm_api.py --base-url https://api.deepseek.com \
                               --model deepseek-flash \
                               --api-key sk-xxxx

    # key 也可以走环境变量，避免进 shell history
    export VADAR_API_KEY=sk-xxxx
    python 02_probe_llm_api.py --base-url https://api.deepseek.com \
                               --model deepseek-flash

    # 双端点：文本用 DeepSeek，视觉用便宜很多的 Qwen3-VL-Flash
    python 02_probe_llm_api.py \
        --base-url https://api.deepseek.com --model deepseek-v4-flash \
        --vision-base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
        --vision-model qwen3-vl-flash --vision-api-key sk-xxxx

    # 关思考（与 03_vadar_llm_bridge.py 读同一个环境变量，保证「测的=跑的」）
    export VADAR_EXTRA_BODY='{"thinking": {"type": "disabled"}}'
    python 02_probe_llm_api.py

为什么 --extra-body 必须存在（2026-09-17 修）
--------------------------------------------
本脚本的 chat_completion() 一直支持 extra_body，但 argparse 既没有 --extra-body，
也不读 VADAR_EXTRA_BODY，5 个调用点也全都没传 —— 于是「显式关思考」在探针里
**从未被送达**。后果是：一次报告里出现 reasoning_tokens，事后无法判断是
「关闭参数无效」还是「探针根本没发这个参数」。报告 cfg 里也没记 extra_body，
连追溯都做不到。

现在：① 参数透传；② 报告 cfg 记录 extra_body 原文；
      ③ 新增 T1b 思考模式 A/B，直接实测关闭参数是否生效；
      ④ 每个测试项都记 usage，用于外推全量实验的 token 与费用。

输出
----
  控制台可读报告 + JSON 报告（--report 指定，默认 llm_probe_report.json）

注意
----
  本脚本只用标准库（urllib / zlib / struct），可在 Windows 原生 Python 直接跑，
  不需要 WSL、不需要 GPU、不需要 openai / torch / transformers。
"""

import argparse
import base64
import json
import os
import re
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib

# Windows 控制台默认 cp936，中文输出可能炸；失败就退回 ascii 替换
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# =====================================================================
# 0. VADAR 的真实正则契约（从源码复制，不要改）
# =====================================================================
RE_DOCSTRING = re.compile(r"<docstring>(.*?)</docstring>", re.DOTALL)
RE_SIGNATURE = re.compile(r"<signature>(.*?)</signature>", re.DOTALL)
RE_IMPLEMENTATION = re.compile(r"<implementation>(.*?)</implementation>", re.DOTALL)
RE_PROGRAM = re.compile(r"<program>(.*?)</program>", re.DOTALL)
RE_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
# agents/agents.py:89 —— 注意结尾的 "):" 字面量
RE_DEF_NAME = re.compile(r"def (\w+)\s*\(.*\):")

# VADAR prompts/vqa_prompt.py:26
VQA_PROMPT = (
    "You will be shown an image with a red bounding box and asked to answer a "
    "question based on the object inside the bounding box. Please only answer "
    "with regards to the object IN the bounding box. Answer this question with "
    "one word based on the object in the bounding box and put your answer in "
    "between <answer></answer> tags: {question}"
)

# 思考模式显式关闭所用的请求体片段（DeepSeek 官方文档确认的字段）
THINKING_DISABLED = {"thinking": {"type": "disabled"}}


# =====================================================================
# 1. 纯标准库生成测试图（模仿 VADAR 的 box_image：红框 + 框内目标 + 框外干扰物）
# =====================================================================
def _png_chunk(tag, data):
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))


def _png_bytes(width, height, pixels):
    """pixels: list[list[(r,g,b)]]，返回 PNG 字节流（truecolor 8bit）。"""
    raw = b"".join(
        b"\x00" + bytes(v for px in row for v in px) for row in pixels
    )
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n"
            + _png_chunk(b"IHDR", ihdr)
            + _png_chunk(b"IDAT", zlib.compress(raw, 6))
            + _png_chunk(b"IEND", b""))


COLORS = {
    "blue": (0, 80, 230),
    "green": (0, 170, 60),
    "yellow": (250, 220, 40),
    "purple": (150, 60, 210),
    "orange": (245, 130, 30),
}


def make_test_image(target_color, distractor_color, size=256):
    """
    返回 (png_bytes, ground_truth_word)。
    布局：
      - 白底
      - 中央 = 目标色方块（在红框**内**）
      - 红框 = 模仿 VADAR box_image 画的框
      - 右下角 = 干扰色方块（在红框**外**），用来检验模型是否真的只看框内
    红色不参与颜色测试（会和框颜色混淆）。
    """
    white = (255, 255, 255)
    red = (255, 0, 0)
    img = [[white] * size for _ in range(size)]

    m = 44          # 目标方块距边距
    pad = 10        # 红框外扩
    lw = 3          # 红框线宽

    obj = COLORS[target_color]
    for y in range(m, size - m):
        for x in range(m, size - m):
            img[y][x] = obj

    x0, y0, x1, y1 = m - pad, m - pad, size - m + pad, size - m + pad
    for y in range(y0, y1):
        for x in range(x0, x1):
            if y < y0 + lw or y >= y1 - lw or x < x0 + lw or x >= x1 - lw:
                img[y][x] = red

    dis = COLORS[distractor_color]
    d0, d1 = size - 26, size - 6          # 位于红框外
    for y in range(d0, d1):
        for x in range(d0, d1):
            img[y][x] = dis

    return _png_bytes(size, size, img), target_color


# =====================================================================
# 2. OpenAI 兼容 HTTP 调用（不依赖 openai SDK）
# =====================================================================
def chat_completion(base_url, api_key, model, messages, temperature=0.0,
                    max_tokens=4096, timeout=180, extra_body=None):
    """
    返回 dict:
      ok(bool) status(int|None) elapsed(float) text(str)
      finish_reason(str) usage(dict) has_reasoning(bool) error(str)
    """
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if extra_body:
        payload.update(extra_body)

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + api_key,
        },
        method="POST",
    )

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        dt = time.time() - t0
        msg = body["choices"][0]["message"]
        text = msg.get("content") or ""
        return {
            "ok": True,
            "status": 200,
            "elapsed": dt,
            "text": text,
            "finish_reason": body["choices"][0].get("finish_reason"),
            "usage": body.get("usage") or {},
            "has_reasoning": bool(msg.get("reasoning_content")),
            "model_returned": body.get("model"),
            "error": "",
        }
    except urllib.error.HTTPError as e:
        dt = time.time() - t0
        raw = ""
        try:
            raw = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        return {
            "ok": False, "status": e.code, "elapsed": dt, "text": "",
            "finish_reason": None, "usage": {}, "has_reasoning": False,
            "model_returned": None,
            "error": ("HTTP %s: %s" % (e.code, raw))[:900],
        }
    except Exception as e:
        dt = time.time() - t0
        return {
            "ok": False, "status": None, "elapsed": dt, "text": "",
            "finish_reason": None, "usage": {}, "has_reasoning": False,
            "model_returned": None,
            "error": ("%s: %s" % (type(e).__name__, e))[:900],
        }


def _text_of(r):
    return r.get("text", "") if r else ""


def _eb(cfg):
    """取本次要透传的 extra_body（None 表示不附加任何字段）。"""
    return cfg.get("extra_body")


# =====================================================================
# 3. 各项测试
# =====================================================================
def t1_connectivity(cfg):
    """最小连通性 + 模型名有效性 + 基础延迟 + 思考模式探测。"""
    r = chat_completion(
        cfg["base_url"], cfg["api_key"], cfg["model"],
        [{"role": "user", "content": "Reply with the single word: pong"}],
        temperature=0.0, max_tokens=64, timeout=60,
        extra_body=_eb(cfg),
    )
    return r


def t1b_thinking_ab(cfg, runs=2):
    """
    思考模式 A/B —— 本探针存在的意义之一。

    同一个极短提示跑两臂：
      disabled : 显式带 {"thinking": {"type": "disabled"}}
      default  : 什么都不带（用端点默认）

    要回答的问题不是「文档怎么说」，而是「本机实测到底生不生效」：
      - disabled 臂若仍出现 reasoning_content -> 关闭参数被忽略，
        那就只能选「留思考 + n>1 报方差」，不能再声称低温可复现；
      - disabled 臂无思考、default 臂有 -> 参数有效，可以关思考换真低温。
    另外记录两臂耗时，关思考省多少延迟也要有数。
    """
    msgs = [{"role": "user", "content": "Reply with the single word: pong"}]
    arms = [("disabled", THINKING_DISABLED), ("default", None)]
    out = []
    for name, eb in arms:
        for i in range(max(1, runs)):
            r = chat_completion(cfg["base_url"], cfg["api_key"], cfg["model"],
                                msgs, temperature=cfg["temperature"],
                                max_tokens=64, timeout=120, extra_body=eb)
            u = r.get("usage") or {}
            cdet = u.get("completion_tokens_details") or {}
            out.append({
                "arm": name,
                "run": i + 1,
                "ok": r["ok"],
                "elapsed": round(r["elapsed"], 2),
                "has_reasoning": r.get("has_reasoning"),
                # 必须保留**整份** usage：token_totals() 只认 r["usage"]。
                # 老版本这里只挑出两个字段，于是 T1b 在记账里恒为 0 ——
                # 而 T1b 恰恰是「思考模式多花多少 token」的那个实验，最不该漏。
                "usage": u,
                "completion_tokens": u.get("completion_tokens"),
                "reasoning_tokens": cdet.get("reasoning_tokens"),
                "finish_reason": r.get("finish_reason"),
                "error": r["error"][:200],
            })
    return out


def t2_program_tags(cfg, runs=3):
    """
    模拟 ProgramAgent：要求把程序放在 <program></program> 里，
    且答案写入 final_result。用 VADAR 的正则解析 + ast.parse 校验语法。
    """
    prompt = (
        "You are a spatial reasoning agent. Available API:\n"
        "def locate(image, object_name):\n"
        "    \"\"\"Return a list of bounding boxes [x1, y1, x2, y2] for the named object.\"\"\"\n"
        "def depth(image, x, y):\n"
        "    \"\"\"Return the metric depth in meters at pixel (x, y).\"\"\"\n"
        "def same_object(image, bbox1, bbox2):\n"
        "    \"\"\"Return True if the two boxes refer to the same physical object.\"\"\"\n\n"
        "Question: Is the leftmost cube closer to the camera than the rightmost cube?\n"
        "Using the provided API, output a program inside the tags <program></program> "
        "to answer the question. The program must store its answer in a variable "
        "called \"final_result\".\n"
    )
    results = []
    for i in range(runs):
        r = chat_completion(cfg["base_url"], cfg["api_key"], cfg["model"],
                            [{"role": "user", "content": prompt}],
                            temperature=cfg["temperature"], max_tokens=cfg["max_tokens"],
                            extra_body=_eb(cfg))
        rec = {"run": i + 1, "ok": r["ok"], "elapsed": round(r["elapsed"], 2),
               "finish_reason": r["finish_reason"], "error": r["error"][:200]}
        if r["ok"]:
            # 记账：漏了 usage 就无法外推全量实验的 token 与费用。
            rec["usage"] = r["usage"]
            rec["has_reasoning"] = r["has_reasoning"]
            text = _text_of(r)
            found = RE_PROGRAM.findall(text)
            rec["tag_found"] = bool(found)
            rec["n_tags"] = len(found)
            if found:
                code = found[0]
                rec["code_chars"] = len(code)
                try:
                    compile(code, "<vadar_program>", "exec")
                    rec["parses"] = True
                except SyntaxError as se:
                    rec["parses"] = False
                    rec["syntax_error"] = str(se)[:200]
                rec["has_final_result"] = "final_result" in code
            else:
                rec["parses"] = False
                rec["has_final_result"] = False
                rec["preview"] = text[:200].replace("\n", " ")
            rec["truncated"] = (r["finish_reason"] == "length")
        results.append(rec)
    return results


def t3_signature_tags(cfg, runs=3):
    """
    模拟 SignatureAgent：要求 <docstring> 紧跟 <signature>。
    额外用 VADAR 的真实正则验证能否提取出方法名（返回值注解会破坏它）。
    """
    prompt = (
        "You are an API designer for a spatial reasoning agent.\n"
        "Existing API:\n"
        "def locate(image, object_name):\n"
        "    \"\"\"Return bounding boxes for the named object.\"\"\"\n"
        "def depth(image, x, y):\n"
        "    \"\"\"Return metric depth at a pixel.\"\"\"\n\n"
        "Question that the API cannot currently answer: how tall is the object "
        "at a given bounding box, in meters?\n\n"
        "Propose exactly one new method to answer it.\n"
        "For each proposed method, output the docstring inside "
        "<docstring></docstring> immediately followed by the method signature for "
        "the docstring inside <signature></signature>. "
        "Do not propose methods that are already in the API.\n"
    )
    results = []
    for i in range(runs):
        r = chat_completion(cfg["base_url"], cfg["api_key"], cfg["model"],
                            [{"role": "user", "content": prompt}],
                            temperature=cfg["temperature"], max_tokens=cfg["max_tokens"],
                            extra_body=_eb(cfg))
        rec = {"run": i + 1, "ok": r["ok"], "elapsed": round(r["elapsed"], 2),
               "finish_reason": r["finish_reason"], "error": r["error"][:200]}
        if r["ok"]:
            rec["usage"] = r["usage"]
            rec["has_reasoning"] = r["has_reasoning"]
            text = _text_of(r)
            docs = RE_DOCSTRING.findall(text)
            sigs = RE_SIGNATURE.findall(text)
            rec["docstring_found"] = bool(docs)
            rec["signature_found"] = bool(sigs)
            rec["n_signatures"] = len(sigs)
            if sigs:
                sig = sigs[0].strip()
                rec["signature_preview"] = sig[:120]
                # --- VADAR 的真实解析路径（agents.py:89）---
                m = RE_DEF_NAME.search(sig)
                rec["vadar_can_parse_name"] = bool(m)
                if m:
                    rec["method_name"] = m.group(1)
                else:
                    rec["vadar_failure"] = (
                        "VADAR 的 RE_DEF_NAME 匹配失败 -> .group(1) 会抛 AttributeError。"
                        " 常见原因：签名带返回值类型注解（-> float），正则要求字面量 '):'。"
                    )
                # 顺带记录是否带返回值注解
                rec["has_return_annotation"] = "->" in sig
            rec["pair_ok"] = bool(docs) and bool(sigs)
        results.append(rec)
    return results


def t4_vision_answer_tags(cfg, runs=None):
    """
    模拟 VQAModule.predict：base64 内联图片 + VADAR 的 VQA_PROMPT，
    用 VADAR 的正则解析 <answer>，并与 ground truth 颜色比对。
    runs 默认跑遍 4 种颜色。
    """
    pairs = [("blue", "orange"), ("green", "purple"),
             ("yellow", "blue"), ("purple", "green")]
    if runs:
        pairs = pairs[:runs]

    use_vision = cfg["vision_enabled"]
    base_url = cfg["vision_base_url"] if use_vision else cfg["base_url"]
    api_key = cfg["vision_api_key"] if use_vision else cfg["api_key"]
    model = cfg["vision_model"] if use_vision else cfg["model"]

    results = []
    for i, (target, distractor) in enumerate(pairs):
        png, truth = make_test_image(target, distractor)
        b64 = base64.b64encode(png).decode("ascii")
        prompt = VQA_PROMPT.format(question="what color is this?")
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": "data:image/png;base64," + b64}},
            ],
        }]
        r = chat_completion(base_url, api_key, model, messages,
                            temperature=cfg["temperature"], max_tokens=cfg["max_tokens"],
                            extra_body=_eb(cfg))
        rec = {"run": i + 1, "truth": truth, "distractor": distractor,
               "ok": r["ok"], "elapsed": round(r["elapsed"], 2),
               "finish_reason": r["finish_reason"], "error": r["error"][:300]}
        if r["ok"]:
            text = _text_of(r)
            found = RE_ANSWER.findall(text)
            rec["tag_found"] = bool(found)
            if found:
                ans = found[0].strip().lower()
                rec["answer"] = ans
                rec["correct"] = (truth in ans)
            else:
                rec["answer"] = None
                rec["correct"] = False
                rec["preview"] = text[:200].replace("\n", " ")
                rec["vadar_failure"] = ("无 <answer> 标签 -> predefined_modules.py:354 "
                                        "取 [0] 会抛 IndexError，该问题整体作废")
            rec["has_reasoning"] = r["has_reasoning"]
            rec["usage"] = r["usage"]
        results.append(rec)
    return results


def t5_error_path(cfg):
    """
    确认「把图片发给纯文本模型」时返回的是可辨识的 4xx。
    这条很重要：Generator.generate 的 except 分支是
        time.sleep(60); return self.generate(...)   # 无限递归
    若错误不可辨识，VADAR 不会报错，而是**静默卡死**。
    """
    if cfg["vision_model"] == cfg["model"] and \
            cfg["vision_base_url"] == cfg["base_url"]:
        return {"skipped": True,
                "reason": "文本与视觉用同一模型，无需测错误路径"}

    png, _ = make_test_image("blue", "orange")
    b64 = base64.b64encode(png).decode("ascii")
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "what color is this?"},
            {"type": "image_url",
             "image_url": {"url": "data:image/png;base64," + b64}},
        ],
    }]
    r = chat_completion(cfg["base_url"], cfg["api_key"], cfg["model"],
                        messages, temperature=0.0, max_tokens=64, timeout=60,
                        extra_body=_eb(cfg))
    return {
        "skipped": False,
        "text_model_rejected_image": (not r["ok"]),
        "status": r["status"],
        "error": r["error"][:400],
        "verdict": ("可辨识：模型名写错时会立刻 4xx 报错，不会静默卡死"
                    if not r["ok"] else
                    "注意：纯文本模型竟然接受了图片请求，需人工确认返回内容是否合理"),
    }


# =====================================================================
# 4. 汇总
# =====================================================================
def _median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def ab_thinking_verdict(t1b):
    """把 T1b 两臂结果压成一句可引用的结论。"""
    if not t1b:
        return {"ran": False, "reason": "--skip-thinking-ab，或 T1 未通过而未执行"}
    arms = {}
    for r in t1b:
        if r.get("ok"):
            arms.setdefault(r["arm"], []).append(r)
    off = arms.get("disabled", [])
    on = arms.get("default", [])
    if not off or not on:
        return {"ran": True, "conclusive": False,
                "reason": "至少一臂没有成功的调用，无法比较",
                "disabled_ok": len(off), "default_ok": len(on)}
    off_r = any(x.get("has_reasoning") for x in off)
    on_r = any(x.get("has_reasoning") for x in on)
    off_ms = _median([x["elapsed"] for x in off])
    on_ms = _median([x["elapsed"] for x in on])
    if on_r and not off_r:
        note = ("关闭参数【有效】：disabled 臂无 reasoning_content，default 臂有。"
                " -> 可以关思考换真低温。")
    elif off_r and on_r:
        note = ("关闭参数【无效/被忽略】：两臂都出现 reasoning_content。"
                " -> 只能选「留思考 + n>1 报方差」，不得声称低温可复现。")
    elif not off_r and not on_r:
        note = ("两臂都没有 reasoning_content：该端点本就不回吐思考内容，"
                " 不能据此判断参数是否生效，请看 usage.reasoning_tokens。")
    else:
        note = "异常：disabled 臂有思考而 default 臂没有，需人工核对。"
    return {
        "ran": True,
        "conclusive": True,
        "disable_param_works": (not off_r),
        "disabled_arm_has_reasoning": off_r,
        "default_arm_has_reasoning": on_r,
        "disabled_median_elapsed": off_ms,
        "default_median_elapsed": on_ms,
        "note": note,
    }


def observed_thinking(cfg, t1, t1b, t2, t3, t4):
    """
    汇总「主路径这一轮实际跑的是开还是关」。

    [!] T1b 必须**按臂拆开**再并入。T1b 的 default 臂是刻意构造的对照组，
        按定义就带思考；若把它算进「整轮跑的是开还是关」，
        本函数会恒判「思考开启」，并输出「不得声称低温可复现」的法令式结论 ——
        与同一份报告里的 thinking_ab（disable 参数有效）**直接矛盾**。
        两臂对比是 thinking_ab 的职责，本函数只回答主路径。
    """
    main_arm = None
    if t1b:
        cfg_eb = cfg.get("extra_body") or None
        if cfg_eb is None:
            main_arm = "default"
        elif cfg_eb == THINKING_DISABLED:
            main_arm = "disabled"
        # 自定义 extra_body 时无法确认它属于哪一臂 -> 不并入主路径统计
    t1b_main = [r for r in (t1b or []) if main_arm and r.get("arm") == main_arm]
    t1b_ctrl = [r for r in (t1b or []) if not (main_arm and r.get("arm") == main_arm)]

    calls = []
    for r in [t1] + t1b_main + list(t2 or []) + list(t3 or []) + list(t4 or []):
        if r and r.get("ok"):
            calls.append(r)
    n = sum(1 for c in calls if c.get("has_reasoning"))
    ctrl = [r for r in t1b_ctrl if r and r.get("ok")]
    out = {
        "extra_body_sent": cfg.get("extra_body"),
        "temperature_configured": cfg.get("temperature"),
        "main_path_arm": main_arm,
        "calls_ok": len(calls),
        "calls_with_reasoning": n,
        "has_reasoning": (n > 0) if calls else None,
        "excluded_control_arm_calls_ok": len(ctrl),
        "excluded_control_arm_with_reasoning": sum(1 for r in ctrl if r.get("has_reasoning")),
        "excluded_control_arm_note": (
            "T1b 对照组按构造就会带思考，已排除在主路径统计之外，"
            "不是反例；两臂对比见 thinking_ab。"),
    }
    if not calls:
        out["note"] = "无成功调用，无法判断"
    elif n:
        out["note"] = ("思考模式生效 -> temperature=%s 被静默忽略；"
                       "报告里不得声称「低温可复现」。该覆盖说明必须写进实验章节。"
                       % cfg.get("temperature"))
    else:
        out["note"] = ("未出现思考痕迹 -> temperature=%s 真正生效。"
                       % cfg.get("temperature"))
    return out


def token_totals(cfg, t1, t1b, t2, t3, t4):
    """
    把每次调用的 usage 汇总，用于外推全量实验的 token 与费用。
    只报 token，不猜价格 —— 单价必须由调用方用 --price-in/--price-out 给出。
    """
    groups = [("T1", [t1]), ("T1b", t1b or []), ("T2", t2 or []),
              ("T3", t3 or []), ("T4", t4 or [])]
    rows = []
    tp = tc = 0
    for name, recs in groups:
        if not recs:
            continue
        p = c = nok = 0
        for r in recs:
            # 只统计**成功**调用：失败响应没有 usage，
            # 若把它们计入分母，per_call 均值会被系统性算低。
            if not (r or {}).get("ok"):
                continue
            u = r.get("usage") or {}
            p += u.get("prompt_tokens") or 0
            c += u.get("completion_tokens") or 0
            nok += 1
        tp += p
        tc += c
        denom = nok or 1
        rows.append({"test": name, "calls": len(recs), "calls_ok": nok,
                     "prompt_tokens": p, "completion_tokens": c,
                     "per_call_prompt": round(p / denom, 1),
                     "per_call_completion": round(c / denom, 1)})
    out = {"by_test": rows, "total_prompt": tp, "total_completion": tc,
           "total_tokens": tp + tc}
    if cfg.get("price_in") or cfg.get("price_out"):
        out["cost_cny_at_given_prices"] = round(
            tp / 1e6 * (cfg.get("price_in") or 0)
            + tc / 1e6 * (cfg.get("price_out") or 0), 4)
    return out


def summarize(t1b, t2, t3, t4, t5):
    v = []

    ab = ab_thinking_verdict(t1b)
    if ab.get("ran"):
        v.append({
            "check": "思考模式关闭参数是否生效（A/B 实测）",
            "pass_rate": ("works" if ab.get("disable_param_works") else "ignored")
                         if ab.get("conclusive") else "n/a",
            "impact": "决定「低温可复现」能不能写进报告；直接决定实验章节的复现性口径",
            "blocking": False,
        })

    prog_ok = sum(1 for x in t2 if x.get("tag_found") and x.get("parses"))
    v.append({
        "check": "<program> 标签 + 语法可编译",
        "pass_rate": "%d/%d" % (prog_ok, len(t2)),
        "impact": "ProgramAgent / Engine 的程序合成主链路（agents.py:757, engine.py:344）",
        "blocking": prog_ok < len(t2),
    })

    sig_ok = sum(1 for x in t3 if x.get("vadar_can_parse_name"))
    v.append({
        "check": "签名可被 VADAR 正则提取出方法名",
        "pass_rate": "%d/%d" % (sig_ok, len(t3)),
        "impact": "SignatureAgent / APIAgent（agents.py:89/352/483/668）",
        "blocking": sig_ok < len(t3),
    })

    ans_ok = sum(1 for x in t4 if x.get("tag_found"))
    ans_correct = sum(1 for x in t4 if x.get("correct"))
    v.append({
        "check": "<answer> 标签出现率",
        "pass_rate": "%d/%d" % (ans_ok, len(t4)),
        "impact": "VQAModule.predict（predefined_modules.py:354，列表取 [0]）",
        "blocking": ans_ok == 0,
    })
    v.append({
        "check": "视觉问答答对率（红框内目标颜色）",
        "pass_rate": "%d/%d" % (ans_correct, len(t4)),
        "impact": "vqa() 的结果质量，直接影响最终答案准确率",
        "blocking": False,
    })

    if not t5.get("skipped"):
        v.append({
            "check": "错误路径可辨识（图片->纯文本模型）",
            "pass_rate": "yes" if t5.get("text_model_rejected_image") else "no",
            "impact": "engine_utils.py:94-96 的无限重试陷阱能否被及时发现",
            "blocking": False,
        })
    return v


def fmt(x, nd=2):
    try:
        return ("%%.%df" % nd) % x
    except Exception:
        return str(x)


def print_report(cfg, t1, t1b, t2, t3, t4, t5, verdicts, ab, obs, tok):
    print("")
    print("=" * 74)
    print("VADAR LLM 后端自检报告")
    print("=" * 74)
    print("  time            : %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    print("  文本端点        : %s" % cfg["base_url"])
    print("  文本模型        : %s" % cfg["model"])
    if cfg["vision_enabled"]:
        print("  视觉端点        : %s  (独立端点)" % cfg["vision_base_url"])
        print("  视觉模型        : %s" % cfg["vision_model"])
    else:
        print("  视觉            : 复用文本端点（要求该模型本身多模态）")
    print("  temperature     : %s   max_tokens: %s"
          % (cfg["temperature"], cfg["max_tokens"]))
    # extra_body 必须出现在报告抬头：不写就事后无法判断这轮跑的是开还是关。
    if cfg.get("extra_body"):
        print("  extra_body      : %s" % json.dumps(cfg["extra_body"], ensure_ascii=False))
    else:
        print("  extra_body      : (无 —— 用的是端点默认值)")
    if obs.get("has_reasoning") is not None:
        print("  思考模式实测    : %d/%d 次成功调用出现 reasoning_content -> %s"
              % (obs["calls_with_reasoning"], obs["calls_ok"],
                 "开着" if obs["has_reasoning"] else "未出现"))

    print("-" * 74)
    print("[T1] 连通性 / 模型名 / 基础延迟")
    if t1["ok"]:
        print("  OK   status=%s  latency=%ss  returned_model=%s"
              % (t1["status"], fmt(t1["elapsed"]), t1["model_returned"]))
        print("       finish_reason=%s  usage=%s"
              % (t1["finish_reason"], json.dumps(t1["usage"], ensure_ascii=False)))
        print("       思考模式痕迹(reasoning_content)=%s" % t1["has_reasoning"])
        print("       回复内容: %r" % _text_of(t1)[:80])
        if t1["model_returned"] and t1["model_returned"] != cfg["model"]:
            print("       [!] 端点回传的模型名(%s) != 请求的模型名(%s)，"
                  "任何按名字断言模型的地方都会误判"
                  % (t1["model_returned"], cfg["model"]))
    else:
        print("  FAIL status=%s  elapsed=%ss" % (t1["status"], fmt(t1["elapsed"])))
        print("       %s" % t1["error"])
        print("       -> 端点或 key 有问题，后面几项不必看，先修这个。")

    print("-" * 74)
    print("[T1b] 思考模式 A/B（决定「低温可复现」能不能写进报告）")
    if not ab.get("ran"):
        print("  skipped: %s" % ab.get("reason"))
    else:
        for r in (t1b or []):
            if not r["ok"]:
                print("  %-9s run%d FAIL %s" % (r["arm"], r["run"], r.get("error", "")[:130]))
                continue
            print("  %-9s run%d reasoning=%-5s reasoning_tok=%-5s %ss"
                  % (r["arm"], r["run"], r.get("has_reasoning"),
                     r.get("reasoning_tokens"), fmt(r["elapsed"])))
        print("  -> %s" % ab.get("note", ab.get("reason", "")))
        if ab.get("conclusive"):
            print("     disabled 中位延迟=%ss   default 中位延迟=%ss"
                  % (fmt(ab["disabled_median_elapsed"]), fmt(ab["default_median_elapsed"])))

    print("-" * 74)
    print("[T2] <program> 标签合规 + 语法可编译   (ProgramAgent / Engine)")
    for r in t2:
        if not r["ok"]:
            print("  run%d FAIL %s" % (r["run"], r.get("error", "")[:150]))
            continue
        print("  run%d tag=%-5s parses=%-5s final_result=%-5s chars=%-5s %ss%s"
              % (r["run"], r.get("tag_found"), r.get("parses"),
                 r.get("has_final_result"), r.get("code_chars", "-"),
                 fmt(r["elapsed"]),
                 "  [TRUNCATED]" if r.get("truncated") else ""))
        if r.get("syntax_error"):
            print("        syntax: %s" % r["syntax_error"])
        if not r.get("tag_found"):
            print("        preview: %s" % r.get("preview", "")[:150])

    print("-" * 74)
    print("[T3] <docstring>/<signature> 标签合规 + VADAR 正则可解析   (Signature/APIAgent)")
    for r in t3:
        if not r["ok"]:
            print("  run%d FAIL %s" % (r["run"], r.get("error", "")[:150]))
            continue
        print("  run%d doc=%-5s sig=%-5s vadar_name=%-5s ret_annot=%-5s %ss"
              % (r["run"], r.get("docstring_found"), r.get("signature_found"),
                 r.get("vadar_can_parse_name"), r.get("has_return_annotation"),
                 fmt(r["elapsed"])))
        if r.get("signature_preview"):
            print("        sig: %s" % r["signature_preview"])
        if r.get("vadar_failure"):
            print("        !! %s" % r["vadar_failure"])

    print("-" * 74)
    print("[T4] 视觉 + <answer> 标签合规        (VQAModule.predict / vqa())")
    for r in t4:
        if not r["ok"]:
            print("  run%d FAIL truth=%-7s status=%s %s"
                  % (r["run"], r["truth"], r.get("status"), r.get("error", "")[:160]))
            continue
        print("  run%d truth=%-7s tag=%-5s answer=%-9s correct=%-5s %ss"
              % (r["run"], r["truth"], r.get("tag_found"),
                 str(r.get("answer")), r.get("correct"), fmt(r["elapsed"])))
        if r.get("vadar_failure"):
            print("        !! %s" % r["vadar_failure"])
        if r.get("preview"):
            print("        preview: %s" % r["preview"][:150])

    print("-" * 74)
    print("[T5] 错误路径可辨识性   (关系到 engine_utils.py:94-96 的无限重试陷阱)")
    if t5.get("skipped"):
        print("  skipped: %s" % t5["reason"])
    else:
        print("  图片发纯文本模型 -> status=%s rejected=%s"
              % (t5["status"], t5["text_model_rejected_image"]))
        print("  %s" % t5["verdict"])
        if t5.get("error"):
            print("  raw: %s" % t5["error"][:250])

    print("-" * 74)
    print("[T6] token 记账（用于外推全量实验成本）")
    if not tok.get("by_test"):
        print("  无 usage 数据（调用全部失败？）")
    else:
        print("  %-6s %6s %5s %15s %18s"
              % ("test", "calls", "ok", "prompt/call", "completion/call"))
        for row in tok["by_test"]:
            print("  %-6s %6d %5d %15s %18s"
                  % (row["test"], row["calls"], row.get("calls_ok", 0),
                     fmt(row["per_call_prompt"], 1), fmt(row["per_call_completion"], 1)))
        print("  合计: prompt=%d  completion=%d  total=%d"
              % (tok["total_prompt"], tok["total_completion"], tok["total_tokens"]))
        if "cost_cny_at_given_prices" in tok:
            print("  本轮费用(按给定单价): %.4f 元  —— 高峰价，空闲时段约减半"
                  % tok["cost_cny_at_given_prices"])
        else:
            print("  未给 --price-in/--price-out -> 不猜价格，只报 token。")
    print("  注意：一个 VADAR 问题 ≈ 1 次程序合成 + k 次 vqa 调用，")
    print("        外推时用各项 per_call 乘以真实调用次数组合，不要拿单次总延迟乘题数。")

    print("=" * 74)
    print("结论映射（每条不通，VADAR 都会在运行时硬崩，而不是给你可读报错）")
    print("=" * 74)
    for v in verdicts:
        mark = "BLOCK" if v["blocking"] else ("  ok " if "/" in v["pass_rate"]
                                              and not v["pass_rate"].startswith("0/")
                                              else " WARN")
        print("  [%s] %-42s %s" % (mark, v["check"], v["pass_rate"]))
        print("         %s" % v["impact"])

    blocking = [v for v in verdicts if v["blocking"]]
    print("-" * 74)
    if blocking:
        print("  判定：链路【不可直接跑通】，先修上面的 BLOCK 项。")
    elif t1["ok"]:
        print("  判定：LLM 后端【可用】。可以进入 Phase 1 的 Omni3D 冒烟测试。")
        print("        下一步：用 llm_backend.env 里的同一组参数去做 03_vadar_llm_bridge.py。")
    else:
        print("  判定：端点本身不通，先解决连通性。")
    print("")


# =====================================================================
# 5. main
# =====================================================================
def main():
    ap = argparse.ArgumentParser(
        description="VADAR LLM 后端自检（零依赖，可在 Windows 原生 Python 运行）")
    ap.add_argument("--base-url", default=os.environ.get("VADAR_BASE_URL",
                                                         "https://api.deepseek.com"))
    # 官方 2026-09 口径：deepseek-flash 为规范名；deepseek-v4-flash 与
    # deepseek-v4-flash-vision-exp 是**遗留别名**，对应模型已退役，
    # 请求统一由 DeepSeek-V4.1-Flash 承接（支持 Vision），按 Flash 价计费。
    # 别名当前仍可用，但规范名更耐久 —— 别「好心」改回带 vision-exp 的旧名。
    ap.add_argument("--model", default=os.environ.get("VADAR_MODEL",
                                                      "deepseek-flash"))
    ap.add_argument("--api-key", default=os.environ.get("VADAR_API_KEY", ""))
    ap.add_argument("--vision-base-url", default=os.environ.get("VADAR_VISION_BASE_URL", ""))
    ap.add_argument("--vision-model", default=os.environ.get("VADAR_VISION_MODEL", ""))
    ap.add_argument("--vision-api-key", default=os.environ.get("VADAR_VISION_API_KEY", ""))
    ap.add_argument("--temperature", type=float,
                    default=float(os.environ.get("VADAR_TEMPERATURE", "0.2")))
    ap.add_argument("--max-tokens", type=int,
                    default=int(os.environ.get("VADAR_MAX_TOKENS", "4096")))
    ap.add_argument("--extra-body", default=os.environ.get("VADAR_EXTRA_BODY", ""),
                    help="JSON 对象字符串，原样并入请求体，例如 "
                         "'{\"thinking\":{\"type\":\"disabled\"}}'。"
                         "默认读同一个环境变量 VADAR_EXTRA_BODY —— "
                         "与 03_vadar_llm_bridge.py 保持一致，否则「探针测的参数」"
                         "与「实验跑的参数」不是同一套。")
    ap.add_argument("--skip-thinking-ab", action="store_true",
                    help="跳过 T1b 思考模式 A/B（默认执行，用于实测关闭参数是否生效）")
    ap.add_argument("--ab-runs", type=int, default=2,
                    help="T1b 每臂重复次数，默认 2")
    ap.add_argument("--price-in", type=float, default=0.0,
                    help="输入价（元/百万 token）；给了才打印费用，不给就不猜")
    ap.add_argument("--price-out", type=float, default=0.0,
                    help="输出价（元/百万 token）")
    ap.add_argument("--runs", type=int, default=3,
                    help="T2/T3 重复次数，默认 3")
    ap.add_argument("--vision-runs", type=int, default=4,
                    help="T4 图片数，最多 4，默认 4（4 种颜色）")
    ap.add_argument("--skip-vision", action="store_true",
                    help="跳过视觉测试（只验证文本链路）")
    ap.add_argument("--report", default="llm_probe_report.json")
    args = ap.parse_args()

    if not args.api_key:
        print("缺少 API key。用 --api-key 或设置环境变量 VADAR_API_KEY。")
        return 2

    # ---- extra_body：解析失败即中止，绝不静默吞掉 ----
    # 静默吞掉正是上一次「报告里有 reasoning 但查不出为什么」的成因。
    extra_body = None
    if args.extra_body.strip():
        try:
            extra_body = json.loads(args.extra_body)
            if not isinstance(extra_body, dict):
                raise ValueError("顶层必须是 JSON 对象")
        except Exception as e:
            print("--extra-body / VADAR_EXTRA_BODY 不是合法 JSON 对象: %s" % e)
            print("  收到: %r" % args.extra_body)
            return 2

    vision_base = args.vision_base_url or args.base_url
    vision_model = args.vision_model or args.model
    vision_key = args.vision_api_key or args.api_key

    cfg = {
        "base_url": args.base_url,
        "model": args.model,
        "api_key": args.api_key,
        "vision_base_url": vision_base,
        "vision_model": vision_model,
        "vision_api_key": vision_key,
        # 语义澄清：本字段说的是「视觉是否走独立端点」，**不是**「视觉是否可用」。
        # 未配独立端点时 T4 回落到主模型，实测能答对，所以另设 vision_source 说明实际走法。
        "vision_enabled": bool(args.vision_base_url or args.vision_model),
        "vision_source": ("dedicated" if (args.vision_base_url or args.vision_model)
                          else "main_model_fallback"),
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "extra_body": extra_body,
        "extra_body_raw": args.extra_body.strip(),
        "price_in": args.price_in,
        "price_out": args.price_out,
    }

    print("探针启动：先测连通性…")
    t1 = t1_connectivity(cfg)
    if not t1["ok"]:
        obs = observed_thinking(cfg, t1, None, None, None, None)
        tok = token_totals(cfg, t1, None, None, None, None)
        print_report(cfg, t1, [], [], [], [], {"skipped": True, "reason": "未执行"},
                     [], {"ran": False, "reason": "T1 未通过"}, obs, tok)
        _dump(cfg, t1, [], [], [], {}, args.report, obs=obs, tok=tok)
        return 1

    t1b = []
    if not args.skip_thinking_ab:
        print("T1b 思考模式 A/B（关 vs 默认）…")
        t1b = t1b_thinking_ab(cfg, runs=max(1, args.ab_runs))

    print("T2 程序合成标签…")
    t2 = t2_program_tags(cfg, runs=max(1, args.runs))
    print("T3 签名标签…")
    t3 = t3_signature_tags(cfg, runs=max(1, args.runs))
    if args.skip_vision:
        t4, t5 = [], {"skipped": True, "reason": "--skip-vision"}
    else:
        print("T4 视觉 + <answer>…")
        t4 = t4_vision_answer_tags(cfg, runs=args.vision_runs)
        print("T5 错误路径…")
        t5 = t5_error_path(cfg)

    verdicts = summarize(t1b, t2, t3, t4, t5)
    ab = ab_thinking_verdict(t1b)
    obs = observed_thinking(cfg, t1, t1b, t2, t3, t4)
    tok = token_totals(cfg, t1, t1b, t2, t3, t4)
    print_report(cfg, t1, t1b, t2, t3, t4, t5, verdicts, ab, obs, tok)
    _dump(cfg, t1, t2, t3, t4, t5, args.report, verdicts=verdicts,
          t1b=t1b, ab=ab, obs=obs, tok=tok)
    print("JSON 报告已写入: %s" % os.path.abspath(args.report))
    return 0


def _dump(cfg, t1, t2, t3, t4, t5, path, verdicts=None,
          t1b=None, ab=None, obs=None, tok=None):
    safe = dict(cfg)
    safe["api_key"] = "***"
    safe["vision_api_key"] = "***"
    out = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": safe,
        "vadar_tag_contract": {
            "docstring": RE_DOCSTRING.pattern,
            "signature": RE_SIGNATURE.pattern,
            "implementation": RE_IMPLEMENTATION.pattern,
            "program": RE_PROGRAM.pattern,
            "answer": RE_ANSWER.pattern,
            "def_name": RE_DEF_NAME.pattern,
        },
        "thinking_observed": obs,
        "thinking_ab": ab,
        "t1_connectivity": t1,
        "t1b_thinking_ab": t1b if t1b is not None else [],
        "t2_program": t2,
        "t3_signature": t3,
        "t4_vision": t4,
        "t5_error_path": t5,
        "token_totals": tok,
        "verdicts": verdicts or [],
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("报告写入失败: %s" % e)


if __name__ == "__main__":
    sys.exit(main())
