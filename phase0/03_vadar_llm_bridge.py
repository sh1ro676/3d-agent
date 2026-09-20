#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_vadar_llm_bridge.py -- VADAR LLM 后端适配器（运行时注入，不改 vendor 源码）

为什么用适配器而不是改源码
--------------------------
项目需要同时保留"原始 VADAR 基线臂 A"。如果直接编辑 vendor/VADAR/ 里的文件，
基线臂就没了。VADAR 每个模块都是 `from engine.engine_utils import Generator`
这种**模块级绑定**，所以只要在导入 agents/engine 之前把这些命名空间里的
`Generator` 符号换掉，就能做到零侵入替换。

需要替换的四个命名空间（缺一个就有一部分代码还在走 OpenAI）
--------------------------------------------------------
    engine.engine_utils        <- 定义处
    engine.predefined_modules  <- predefined_modules.py:23 `from .engine_utils import *`
    engine.engine              <- engine.py:20-25 显式 from-import
    agents.agents              <- agents.py:19-26 显式 from-import

顺带修掉三个"换模型后才会暴露"的坑
----------------------------------
1. engine_utils.py:94-96  异常分支是 `time.sleep(60); return self.generate(...)`
                          无限递归。任何一次 4xx（例如把图发给纯文本模型）
                          都会变成**静默卡死**而不是报错。-> 改为有界重试 + 快速失败。
2. predefined_modules.py:354  `re.findall(r"<answer>...")[0]`
                          模型不吐标签就 IndexError，整道题作废。-> 加容错兜底 + 告警。
3. 调用处都没有传 max_tokens，靠服务端默认值。程序合成输出可能被**静默截断**，
                          生成一个语法不完整的程序。-> 显式设置并检测 finish_reason。

同时输出结构化调用日志（JSONL），直接喂给评估章节的
"Average Latency / Inference Cost" 指标。

配置（全部走环境变量，不必改文件）
----------------------------------
    VADAR_BASE_URL          https://api.deepseek.com
    VADAR_API_KEY           sk-xxxx
    VADAR_MODEL             deepseek-flash
    VADAR_TEMPERATURE       0.2
    VADAR_MAX_TOKENS        8192
    VADAR_MAX_RETRIES       2
    VADAR_CALL_LOG          logs/vadar_llm_calls.jsonl
    VADAR_LOG_PROMPTS       0
    VADAR_STRICT_TAGS       0
    VADAR_EXTRA_BODY        {"thinking": {"type": "disabled"}}   <- 原样透传，不用猜参数名
    VADAR_REPO              /home/<user>/projects/3d_spatial_agent/vendor/VADAR

    可选：视觉走独立端点（更省钱；文本/视觉不同服务商时用）
    VADAR_VISION_BASE_URL   https://dashscope.aliyuncs.com/compatible-mode/v1
    VADAR_VISION_API_KEY    sk-yyyy
    VADAR_VISION_MODEL      qwen3-vl-flash

    价格表（人民币/百万 token），JSON，可覆盖内置表；未配置的模型只记 token 不算钱
    VADAR_PRICE_TABLE       {"deepseek-flash": [3.0, 9.0]}

用法
----
    import vadar_llm_bridge as bridge
    bridge.install()                     # 必须在 import agents/engine 之前调用
    from agents.agents import SignatureAgent, APIAgent, ProgramAgent
    from engine.engine import Engine
    ...

    # 自检（不需要 VADAR、不需要 torch）
    python 03_vadar_llm_bridge.py --selftest
    python 03_vadar_llm_bridge.py --show-config
    python 03_vadar_llm_bridge.py --summary logs/vadar_llm_calls.jsonl

依赖：仅标准库。不依赖 openai SDK，因此不受 VADAR 钉死的 openai==1.51.2 影响。
"""

import argparse
import base64
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# =====================================================================
# 1. 配置
# =====================================================================
# 内置价格表：仅收录有明确官方公开价的模型（人民币 / 百万 token，[输入, 输出]）
# 注意：DeepSeek 分高峰/空闲时段，这里取**高峰价**（保守估计）。
# 其他模型（如 qwen3.5 系列）价格口径在不同页面上不一致，故意不内置，
# 请用 VADAR_PRICE_TABLE 显式配置，避免报告里出现来源不明的数字。
DEFAULT_PRICE_TABLE = {
    "deepseek-flash": [3.0, 9.0],
    "deepseek-v4-flash": [3.0, 9.0],
    "deepseek-v4-pro": [9.0, 27.0],
}


def _env(name, default=None):
    v = os.environ.get(name)
    return v if (v is not None and v != "") else default


def _envf(name, default):
    try:
        return float(_env(name, default))
    except (TypeError, ValueError):
        return float(default)


def _envi(name, default):
    try:
        return int(_env(name, default))
    except (TypeError, ValueError):
        return int(default)


class LLMConfig(object):
    """一个端点（文本或视觉）的配置。"""

    def __init__(self, base_url, api_key, model, temperature, max_tokens,
                 max_retries, timeout, extra_body, price_table, label="text"):
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or ""
        self.model = model or ""
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.timeout = timeout
        self.extra_body = extra_body or None
        self.price_table = price_table or {}
        self.label = label

    def price(self):
        """返回 (输入价, 输出价) / 百万 token；未知返回 None。"""
        for k, v in self.price_table.items():
            if k and (k in self.model):
                return v[0], v[1]
        return None

    def cost_cny(self, usage):
        p = self.price()
        if not p:
            return None
        pin = usage.get("prompt_tokens") or 0
        pout = usage.get("completion_tokens") or 0
        return (pin * p[0] + pout * p[1]) / 1e6

    def describe(self):
        return {
            "label": self.label,
            "base_url": self.base_url,
            "model": self.model,
            "api_key": ("***" if self.api_key else "(missing)"),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_retries": self.max_retries,
            "extra_body": self.extra_body,
            "price_cny_per_mtok": self.price(),
        }


def _load_price_table():
    tbl = dict(DEFAULT_PRICE_TABLE)
    raw = _env("VADAR_PRICE_TABLE")
    if raw:
        try:
            user = json.loads(raw)
            for k, v in user.items():
                if isinstance(v, (list, tuple)) and len(v) == 2:
                    tbl[str(k)] = [float(v[0]), float(v[1])]
        except Exception as e:
            print("[bridge] VADAR_PRICE_TABLE 解析失败，忽略: %s" % e)
    return tbl


def _load_extra_body():
    raw = _env("VADAR_EXTRA_BODY")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception as e:
        print("[bridge] VADAR_EXTRA_BODY 解析失败，忽略: %s" % e)
        return None


def load_config():
    """从环境变量组装 (text_cfg, vision_cfg)。"""
    price = _load_price_table()
    extra = _load_extra_body()
    temp = _envf("VADAR_TEMPERATURE", 0.2)
    mt = _envi("VADAR_MAX_TOKENS", 8192)
    mr = _envi("VADAR_MAX_RETRIES", 2)
    to = _envf("VADAR_TIMEOUT", 180.0)

    text = LLMConfig(
        base_url=_env("VADAR_BASE_URL", "https://api.deepseek.com"),
        api_key=_env("VADAR_API_KEY", ""),
        model=_env("VADAR_MODEL", "deepseek-flash"),
        temperature=temp, max_tokens=mt, max_retries=mr, timeout=to,
        extra_body=extra, price_table=price, label="text",
    )

    vbase = _env("VADAR_VISION_BASE_URL")
    vmodel = _env("VADAR_VISION_MODEL")
    if vbase or vmodel:
        vision = LLMConfig(
            base_url=vbase or text.base_url,
            api_key=_env("VADAR_VISION_API_KEY", text.api_key),
            model=vmodel or text.model,
            temperature=temp, max_tokens=_envi("VADAR_VISION_MAX_TOKENS", mt),
            max_retries=mr, timeout=to, extra_body=extra,
            price_table=price, label="vision",
        )
    else:
        vision = text  # 同一端点，要求该模型本身多模态
    return text, vision


# =====================================================================
# 2. 调用日志
# =====================================================================
class CallLogger(object):
    def __init__(self, path=None, log_prompts=False):
        self.path = path or _env("VADAR_CALL_LOG", "logs/vadar_llm_calls.jsonl")
        self.log_prompts = log_prompts or (_env("VADAR_LOG_PROMPTS", "0") == "1")

    def write(self, rec):
        try:
            d = os.path.dirname(os.path.abspath(self.path))
            if d and not os.path.isdir(d):
                os.makedirs(d, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass


_LOGGER = None


def _logger():
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = CallLogger()
    return _LOGGER


# =====================================================================
# 3. 底层 HTTP 调用（urllib，无第三方依赖）
# =====================================================================
class LLMError(RuntimeError):
    """不可重试的调用失败（4xx 等）。评估脚本应逐题 catch 它。"""

    def __init__(self, msg, status=None, body=""):
        RuntimeError.__init__(self, msg)
        self.status = status
        self.body = body


def _messages_look_visual(messages):
    """自动识别视觉调用：content 是 list 且含 image_url。"""
    if not messages:
        return False
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
    return False


def _short_messages_digest(messages):
    """给日志留一个轻量指纹，便于排查而不必记录全文。"""
    if not messages:
        return {"n": 0}
    images = 0
    chars = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            chars += len(c)
        elif isinstance(c, list):
            for part in c:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "image_url":
                    images += 1
                elif part.get("type") == "text":
                    chars += len(part.get("text") or "")
    return {"n": len(messages), "images": images, "text_chars": chars}


def call_chat(cfg, messages, purpose="generate"):
    """
    带重试与日志的单次调用。返回 dict:
      text, finish_reason, usage, has_reasoning, elapsed, attempts, model_returned
    不可重试的错误直接抛 LLMError。
    """
    url = cfg.base_url + "/chat/completions"
    payload = {
        "model": cfg.model,
        "messages": messages,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
    }
    if cfg.extra_body:
        payload.update(cfg.extra_body)

    if not cfg.api_key:
        raise LLMError("缺少 API key（VADAR_API_KEY 未设置）")

    attempt = 0
    last_err = ""
    last_status = None
    while attempt <= cfg.max_retries:
        attempt += 1
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + cfg.api_key,
            },
            method="POST",
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=cfg.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            dt = time.time() - t0
            msg = body["choices"][0]["message"]
            text = msg.get("content") or ""
            usage = body.get("usage") or {}
            rec = {
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "role": cfg.label,
                "model": cfg.model,
                "purpose": purpose,
                "ok": True,
                "http_status": 200,
                "attempts": attempt,
                "elapsed_s": round(dt, 3),
                "finish_reason": body["choices"][0].get("finish_reason"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "truncated": body["choices"][0].get("finish_reason") == "length",
                "has_reasoning": bool(msg.get("reasoning_content")),
                "est_cost_cny": cfg.cost_cny(usage),
                "msg_digest": _short_messages_digest(messages),
            }
            if rec["truncated"]:
                rec["warning"] = ("finish_reason=length：输出被截断。"
                                  "VADAR 生成的程序可能是残缺的，请调大 VADAR_MAX_TOKENS。")
            lg = _logger()
            if lg.log_prompts:
                rec["messages"] = messages
            lg.write(rec)
            return {
                "text": text,
                "finish_reason": body["choices"][0].get("finish_reason"),
                "usage": usage,
                "has_reasoning": bool(msg.get("reasoning_content")),
                "elapsed": dt,
                "attempts": attempt,
                "model_returned": body.get("model"),
            }
        except urllib.error.HTTPError as e:
            dt = time.time() - t0
            try:
                raw = e.read().decode("utf-8", errors="replace")
            except Exception:
                raw = ""
            last_err = raw[:600]
            last_status = e.code
            # ---- 关键修复：4xx 一律快速失败，不再 sleep(60) 无限重试 ----
            retryable = (e.code == 429 or e.code >= 500)
            _logger().write({
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "role": cfg.label, "model": cfg.model, "purpose": purpose,
                "ok": False, "http_status": e.code, "attempts": attempt,
                "elapsed_s": round(dt, 3), "retryable": retryable,
                "error": last_err, "msg_digest": _short_messages_digest(messages),
            })
            if not retryable:
                hint = ""
                if e.code == 400 and "image" in last_err.lower():
                    hint = ("  <-- 很可能把图片发给了不支持视觉的模型。"
                            "检查 VADAR_MODEL / VADAR_VISION_MODEL。")
                raise LLMError("HTTP %s: %s%s" % (e.code, last_err, hint),
                               status=e.code, body=last_err)
        except Exception as e:
            dt = time.time() - t0
            last_err = "%s: %s" % (type(e).__name__, e)
            last_status = None
            _logger().write({
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "role": cfg.label, "model": cfg.model, "purpose": purpose,
                "ok": False, "http_status": None, "attempts": attempt,
                "elapsed_s": round(dt, 3), "retryable": True,
                "error": last_err, "msg_digest": _short_messages_digest(messages),
            })

        if attempt <= cfg.max_retries:
            backoff = min(2 ** attempt, 16)
            print("[bridge] %s 调用失败(第%d次)，%.1fs 后重试: %s"
                  % (cfg.label, attempt, backoff, last_err[:160]))
            time.sleep(backoff)

    raise LLMError("重试 %d 次仍失败 (status=%s): %s"
                   % (cfg.max_retries, last_status, last_err[:400]),
                   status=last_status, body=last_err)


# =====================================================================
# 4. 替换 VADAR 的 Generator
# =====================================================================
def _remove_substring(output, substring):
    return output.replace(substring, "") if substring in output else output


class PatchedGenerator(object):
    """
    签名与 engine/engine_utils.py:52 的 Generator 完全一致，
    但忽略 model_name / api_key_path，改由环境变量驱动端点。
    """

    def __init__(self, model_name="gpt-4o", temperature=0.7, api_key_path="./api.key"):
        self._text_cfg, self._vision_cfg = load_config()
        self._requested_model_name = model_name
        self._requested_api_key_path = api_key_path
        self._requested_temperature = temperature

        # VADAR 会在 engine.py:352 / agents.py:766 读 generator.model_name 并写进结果 JSON。
        # 这里必须返回**真实调用的模型名**，否则实验记录会自相矛盾。
        self.model_name = self._text_cfg.model
        self.temperature = self._text_cfg.temperature
        self.api_key_path = api_key_path

        if model_name and model_name not in ("gpt-4o", self._text_cfg.model):
            print("[bridge] 注意：调用方请求 model_name=%r，已被环境变量覆盖为 %r"
                  % (model_name, self._text_cfg.model))

    # 与原始实现同一签名
    def remove_substring(self, output, substring):
        return _remove_substring(output, substring)

    def generate(self, prompt, messages=None):
        if messages:
            msgs = messages
        else:
            msgs = [{"role": "user", "content": prompt}]

        cfg = self._vision_cfg if _messages_look_visual(msgs) else self._text_cfg
        purpose = "vqa" if cfg.label == "vision" else "generate"

        res = call_chat(cfg, msgs, purpose=purpose)
        result = res["text"].lstrip("\n").rstrip("\n")
        result = _remove_substring(result, "```python")
        result = _remove_substring(result, "```")

        # 保留原始语义：把 assistant 回复追加回 messages 并连同返回
        new_messages = msgs
        try:
            new_messages.append({"role": "assistant", "content": result})
        except Exception:
            pass
        return result, new_messages


# =====================================================================
# 5. <answer> 容错（predefined_modules.py:354 的 [0] 崩溃点）
# =====================================================================
RE_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_STRIP = " \t\r\n`\"'.,;:。，、！？!?"


def extract_answer_tolerant(output, strict=False, context=""):
    """
    先按 VADAR 原样取标签；失败则兜底并告警，而不是让整道题 IndexError。
    兜底顺序：去标签 -> 最后一个非空行 -> 去常见前缀/标点。
    """
    hits = RE_ANSWER.findall(output)
    if hits:
        return hits[0].strip().lower()

    if strict:
        raise ValueError("<answer> 标签缺失（VADAR_STRICT_TAGS=1）: %r"
                         % output[:200])

    text = re.sub(r"</?[a-zA-Z_]+>", " ", output or "")
    cand = ""
    for line in reversed([l.strip() for l in text.splitlines()]):
        if line:
            cand = line
            break
    cand = cand.strip(_STRIP)
    for pref in ("answer:", "answer is", "the answer is", "the color is"):
        if cand.lower().startswith(pref):
            cand = cand[len(pref):].strip(_STRIP)
    parts = cand.split()
    if len(parts) > 1:
        cand = parts[-1].strip(_STRIP)

    print("[bridge] !! <answer> 标签缺失，已兜底为 %r（原始输出前 160 字: %r）"
          % (cand, (output or "")[:160]))
    _logger().write({
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "event": "answer_tag_fallback",
        "context": context,
        "fallback": cand,
        "raw_preview": (output or "")[:600],
    })
    return cand.lower()


def _patch_vqa(pm):
    """替换 VQAModule.predict，保留原逻辑 + 加容错。"""
    from io import BytesIO as _BytesIO  # noqa

    def predict(self, img, question, holistic=False):
        prompt = self._get_prompt(question, holistic)
        buffered = _BytesIO()
        img.save(buffered, format="PNG")
        b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": "data:image/png;base64," + b64}},
            ],
        }]
        output, _ = self.generator.generate("", messages)
        output = _remove_substring(output, "```python")
        output = _remove_substring(output, "```")
        strict = _env("VADAR_STRICT_TAGS", "0") == "1"
        return extract_answer_tolerant(output, strict=strict,
                                       context="VQAModule.predict")

    pm.VQAModule.predict = predict
    return True


# =====================================================================
# 6. install()
# =====================================================================
def locate_repo(start=None):
    """从 cwd 向上找含 engine/predefined_modules.py 的目录。"""
    env = _env("VADAR_REPO")
    if env and os.path.isfile(os.path.join(env, "engine", "predefined_modules.py")):
        return os.path.abspath(env)
    cur = os.path.abspath(start or os.getcwd())
    for _ in range(8):
        if os.path.isfile(os.path.join(cur, "engine", "predefined_modules.py")):
            return cur
        nxt = os.path.dirname(cur)
        if nxt == cur:
            break
        cur = nxt
    return None


def install(repo_root=None, light=False, verbose=True):
    """
    在导入 agents/engine **之前**调用。
    light=True 只替换 engine.engine_utils（不需要 torch/sam2/groundingdino）。
    """
    info = {"patched": [], "failed": [], "repo_root": None, "warnings": []}

    root = locate_repo(repo_root)
    if root:
        info["repo_root"] = root
        # predefined_modules.py:17 是 `from VADAR.prompts... import ...`，
        # 所以仓库根目录自己必须在 sys.path 上（供 VADAR.* 解析），
        # 同时其父目录也必须在（供 agents.*/engine.* 解析）。
        for p in (root, os.path.dirname(root)):
            if p and p not in sys.path:
                sys.path.insert(0, p)
        if os.path.basename(root) != "VADAR":
            info["warnings"].append(
                "源码目录名是 %r，不是 'VADAR'。predefined_modules.py:17 硬编码了 "
                "`from VADAR.prompts...`，非 light 模式会 ImportError。"
                % os.path.basename(root))
    else:
        info["warnings"].append("未定位到 VADAR 仓库根目录，请设置 VADAR_REPO。")

    try:
        import engine.engine_utils as eu
        eu.Generator = PatchedGenerator
        info["patched"].append("engine.engine_utils")
    except Exception as e:
        info["failed"].append(("engine.engine_utils", repr(e)))
        if verbose:
            print("[bridge] 无法导入 engine.engine_utils: %r" % e)
        return info

    if not light:
        try:
            import engine.predefined_modules as pm
            pm.Generator = PatchedGenerator
            _patch_vqa(pm)
            info["patched"].append("engine.predefined_modules (+VQAModule.predict)")
        except Exception as e:
            info["failed"].append(("engine.predefined_modules", repr(e)))

        try:
            import engine.engine as eg
            eg.Generator = PatchedGenerator
            info["patched"].append("engine.engine")
        except Exception as e:
            info["failed"].append(("engine.engine", repr(e)))

        try:
            import agents.agents as ag
            ag.Generator = PatchedGenerator
            info["patched"].append("agents.agents")
        except Exception as e:
            info["failed"].append(("agents.agents", repr(e)))

    if verbose:
        t, v = load_config()
        print("[bridge] 已替换 Generator 的模块: %s" % ", ".join(info["patched"]))
        for name, err in info["failed"]:
            print("[bridge] !! 替换失败 %s: %s" % (name, err[:200]))
        for w in info["warnings"]:
            print("[bridge] 警告: %s" % w)
        print("[bridge] 文本端点 %s  model=%s" % (t.base_url, t.model))
        print("[bridge] 视觉端点 %s  model=%s" % (v.base_url, v.model))
        print("[bridge] 调用日志 -> %s"
              % _env("VADAR_CALL_LOG", "logs/vadar_llm_calls.jsonl"))
    return info


# =====================================================================
# 7. 日志汇总（喂给评估章节的延迟/成本指标）
# =====================================================================
def summarize_log(path):
    recs = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        recs.append(json.loads(line))
                    except Exception:
                        pass
    except Exception as e:
        return {"error": str(e), "path": path}

    calls = [r for r in recs if "elapsed_s" in r]
    ok = [r for r in calls if r.get("ok")]
    bad = [r for r in calls if not r.get("ok")]
    lat = sorted(r["elapsed_s"] for r in ok)

    def pct(p):
        if not lat:
            return None
        i = min(len(lat) - 1, int(len(lat) * p))
        return round(lat[i], 3)

    cost = [r.get("est_cost_cny") for r in ok if r.get("est_cost_cny") is not None]
    by_role = {}
    for r in ok:
        k = r.get("role", "?")
        d = by_role.setdefault(k, {"n": 0, "sum_s": 0.0, "ptok": 0, "ctok": 0})
        d["n"] += 1
        d["sum_s"] += r["elapsed_s"]
        d["ptok"] += r.get("prompt_tokens") or 0
        d["ctok"] += r.get("completion_tokens") or 0

    return {
        "path": path,
        "total_records": len(recs),
        "calls": len(calls),
        "ok": len(ok),
        "failed": len(bad),
        "retried": len([r for r in ok if (r.get("attempts") or 1) > 1]),
        "truncated": len([r for r in ok if r.get("truncated")]),
        "answer_tag_fallbacks": len([r for r in recs
                                     if r.get("event") == "answer_tag_fallback"]),
        "latency_s": {
            "avg": round(sum(lat) / len(lat), 3) if lat else None,
            "p50": pct(0.50), "p95": pct(0.95),
            "min": lat[0] if lat else None, "max": lat[-1] if lat else None,
        },
        "est_cost_cny_total": round(sum(cost), 4) if cost else None,
        "costrable_calls": len(cost),
        "by_role": by_role,
        "failure_samples": [r.get("error", "")[:200] for r in bad[:5]],
    }


# =====================================================================
# 8. CLI
# =====================================================================
def _selftest():
    """不需要 VADAR / torch：直接验证 bridge 自己的调用层。"""
    text, vision = load_config()
    print("=" * 70)
    print("bridge 自检")
    print("=" * 70)
    print("文本配置:", json.dumps(text.describe(), ensure_ascii=False))
    print("视觉配置:", json.dumps(vision.describe(), ensure_ascii=False))
    print("-" * 70)

    print("[1] 文本调用…")
    try:
        r = call_chat(text, [{"role": "user", "content": "Reply with the single word: pong"}],
                      purpose="selftest_text")
        print("    OK %.2fs  finish=%s  usage=%s  reasoning=%s"
              % (r["elapsed"], r["finish_reason"], r["usage"], r["has_reasoning"]))
        print("    text=%r" % r["text"][:80])
    except LLMError as e:
        print("    FAIL %s" % e)

    print("[2] 视觉调用（内联一张纯色 PNG，验证多模态通道）…")
    # 最小可用 PNG：1x1 蓝色
    png = _one_pixel_png((0, 80, 230))
    b64 = base64.b64encode(png).decode("ascii")
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "What color is this image? Answer in one word inside "
                                 "<answer></answer> tags."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
    ]}]
    try:
        r = call_chat(vision, msgs, purpose="selftest_vision")
        print("    OK %.2fs  finish=%s" % (r["elapsed"], r["finish_reason"]))
        print("    text=%r" % r["text"][:120])
        print("    VADAR 正则提取: %r" % extract_answer_tolerant(r["text"]))
    except LLMError as e:
        print("    FAIL %s" % e)
        if "does not support image" in str(e) or "400" in str(e):
            print("    -> 该模型不吃图片。文本链路可用，但 vqa() 会崩，"
                  "必须换多模态模型或配 VADAR_VISION_* 独立端点。")
    print("-" * 70)
    s = summarize_log(_env("VADAR_CALL_LOG", "logs/vadar_llm_calls.jsonl"))
    print("日志汇总:", json.dumps(s, ensure_ascii=False, indent=2))


def _one_pixel_png(rgb, size=64):
    import struct
    import zlib
    w = h = size
    raw = (b"\x00" + bytes(rgb) * w) * h

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6))
            + chunk(b"IEND", b""))


def main():
    ap = argparse.ArgumentParser(description="VADAR LLM 后端适配器")
    ap.add_argument("--selftest", action="store_true",
                    help="验证调用层，不需要 VADAR/torch")
    ap.add_argument("--show-config", action="store_true")
    ap.add_argument("--summary", metavar="JSONL",
                    help="汇总调用日志，输出延迟/成本/失败统计")
    args = ap.parse_args()

    if args.show_config:
        t, v = load_config()
        print(json.dumps({"text": t.describe(), "vision": v.describe()},
                         ensure_ascii=False, indent=2))
        return 0
    if args.summary:
        print(json.dumps(summarize_log(args.summary), ensure_ascii=False, indent=2))
        return 0
    if args.selftest:
        _selftest()
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
