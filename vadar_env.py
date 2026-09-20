#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""vadar_env.py —— 从 `configs/llm_backend.env` 读实验配置（零依赖，不 import torch）。

为什么需要这个文件
==================
VADAR 后端的全部入参都是环境变量（`phase0/03_vadar_llm_bridge.py::_env()`），
而 `phase0/06_deepseek_setup.ps1` 设的是**会话级**变量：只在那一个 PowerShell
窗口里有效、**不落盘**、也不会传进任何别的进程。2026-09-17 实跑 arm A 就卡在这里 ——
运行器完整、`--plan` 全绿，真跑时读不到 `VADAR_API_KEY`，直接 `return 2`。

解法不是「每次手输一次 key」，而是把配置**落到一个文件里**：

    configs/llm_backend.env

于是「这次实验用的是哪套后端配置」变成一行可检查、可归档、可复现的事实，
而不是「某个窗口现在还开着吗」。

语义（这几条是刻意选的；动之前先读 `tests/test_vadar_env.py`）
==============================================================
* **文件里的值优先于进程环境。** 文件是用户**刚刚编辑过**的东西；进程环境可能
  是几天前设的、早已忘掉的残留。两者都有且**不同**时记一条可见的 `conflict`，
  绝不静默覆盖 —— 「失败要响」是本项目的地基之一。
* **空值视同未设。** 模板里留 `VADAR_API_KEY=` 是常态，它不该把一个真实存在的
  环境变量顶掉（那会让「我只填了一行，结果另一行把我顶了」变成静默事故）。
  与 bridge 的 `_env()` 口径一致：空串按「没有」处理。
* **`#` 只在行首（可前置空白）才是注释，不支持行尾注释。** 值里带 `#` 时行尾
  注释会**静默截断**配置 —— 宁可少一种语法，也不要多一类静默失败。
* **重复键直接报错。** 同一个键出现两次多半是改配置时忘了注释旧行，
  静默取最后一个正是最难查的那类 bug。
* **未知键只告警不报错**，但会写进报告的 `unknown_keys` —— 否则拼错一个
  `VADAR_BASE_URLL` 是完全静默的（探针显示"默认值"，看起来一切正常）。
* **密钥永不进报告。** 对外只有两种呈现：`mask()`（`***abcd`）与
  `fingerprint()`（sha256 前 8 位，不可逆但能回答"还是同一把 key 吗"）。

本文件放在**仓库根**而不是某个子包里：它是所有入口（`evaluation/runner.py`、
`phase0/*.py`、单测）共用的配置层，放子包里会造成 `evaluation ↔ phase0` 的
包级循环依赖。
"""

from __future__ import annotations

import hashlib
import os
import re

__all__ = [
    "DEFAULT_RELATIVE_PATH", "KNOWN_KEYS", "SECRET_MARKERS", "NOT_SECRET",
    "EnvFileError", "default_path", "is_secret", "mask", "fingerprint",
    "parse_env_text", "load_env_file", "locate", "describe",
]

# 相对**仓库根**的默认位置。用 VADAR_ENV_FILE 或各入口的 --env-file 覆盖。
DEFAULT_RELATIVE_PATH = os.path.join("configs", "llm_backend.env")

# 名字里带这些词的键一律按密钥处理（掩码 + 不进指纹明文）。
SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")

# 但 `TOKEN` 太容易误伤：`VADAR_MAX_TOKENS` 是**数量上限**不是密钥。
# 2026-09-18 实测踩到过 —— 报告里把 max_tokens 显示成 `***8192`，
# 而 `secrets.max_tokens.fingerprint` 还大摇大摆地写了一串哈希。
# 白名单式例外比"更聪明的正则"可靠：误判成密钥只是难看，
# 漏判才是安全事故，所以默认从严 + 显式例外。
NOT_SECRET = ("MAX_TOKENS", "MAX_OUTPUT_TOKENS", "TOKENIZER", "TOKENS_PER")

# 已知的键 —— **仅用于「拼错告警」，不用于拒绝**。
# 来源：`evaluation/runner.py::DEFAULTS` + `phase0/03_vadar_llm_bridge.py::_env()`
# 调用点 + `evaluation/vadar_compat.py`。加了新的环境变量请同步这里，
# 否则它自己会变成一条「未知键」告警（这正是这套机制想要的提醒）。
KNOWN_KEYS = frozenset([
    # 后端（bridge 全量读这些）
    "VADAR_BASE_URL", "VADAR_API_KEY", "VADAR_MODEL",
    "VADAR_TEMPERATURE", "VADAR_MAX_TOKENS", "VADAR_MAX_RETRIES", "VADAR_TIMEOUT",
    "VADAR_EXTRA_BODY", "VADAR_PRICE_TABLE", "VADAR_STRICT_TAGS", "VADAR_LOG_PROMPTS",
    "VADAR_VISION_BASE_URL", "VADAR_VISION_MODEL", "VADAR_VISION_API_KEY",
    "VADAR_VISION_MAX_TOKENS",
    # 记账与路径
    "VADAR_CALL_LOG", "VADAR_REPO", "VADAR_REPO_ROOT", "VADAR_GDINO_DIR",
    "VADAR_GDINO_CAPTION",
    # 运行器自己的开关
    "VADAR_ENV_FILE",
    # HuggingFace（必须在 import torch 之前生效，故同样放进这个文件）
    "HF_HOME", "HF_ENDPOINT", "HF_HUB_OFFLINE", "HF_HUB_DISABLE_XET",
])

_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class EnvFileError(ValueError):
    """文件本身写错了（重复键 / 不是 KEY=VALUE / 键名非法）。

    这类错误必须让整轮**停住**：一个写坏的配置文件如果被"容错"跳过，
    实验就会拿默认值跑完，产出一份**看起来正常但配置不对**的结果 ——
    那比直接报错昂贵得多。
    """


# =====================================================================
# 1. 呈现（密钥的唯一出口）
# =====================================================================
def is_secret(name: str) -> bool:
    up = (name or "").upper()
    if any(x in up for x in NOT_SECRET):
        return False
    return any(m in up for m in SECRET_MARKERS)


def mask(value) -> str:
    """密钥的对外呈现。短到没得遮时只给 `***`，不泄露长度分布。"""
    if not value:
        return "(empty)"
    v = str(value)
    return "***" + v[-4:] if len(v) > 4 else "***"


def fingerprint(value) -> str:
    """不可逆指纹 —— 用于回答「这次和上次是同一把 key 吗」。

    报告里存指纹而不存明文，是为了让「两次 run 用了不同凭据」这种
    归因问题**仍然可查**，同时不让密钥落到任何产物里。
    """
    if not value:
        return None
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:8]


# =====================================================================
# 2. 解析
# =====================================================================
def _parse(text: str):
    """→ (values, linenos)。两个都返回，因为报错与提示都要行号。"""
    values, linenos, dups, bad = {}, {}, [], []
    for i, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            bad.append((i, raw.strip()[:60]))
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if key.lower().startswith("export "):
            key = key[7:].strip()
        if not _KEY_RE.match(key):
            bad.append((i, raw.strip()[:60]))
            continue
        val = val.strip()
        # 只剥「整个值被同一对引号包住」的那一层。
        # 不能无脑 strip 引号：`VADAR_EXTRA_BODY={"thinking": {"type": "disabled"}}`
        # 是合法 JSON，首字符是 `{` 不是引号，必须原样保留。
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        if key in values:
            dups.append(key)
        values[key] = val
        linenos[key] = i
    if dups:
        raise EnvFileError("重复的键 %s —— 多半是改配置时忘了注释掉旧的那行；"
                           "请删掉其中一个（重复键会静默覆盖，所以这里直接报错）"
                           % ", ".join(sorted(set(dups))))
    if bad:
        raise EnvFileError("这些行不是合法的 KEY=VALUE：%s"
                           % "; ".join("第 %d 行 %r" % b for b in bad))
    return values, linenos


def parse_env_text(text: str) -> dict:
    """纯函数版：文本 → {键: 值}。写文档/测试时用这个。"""
    return _parse(text)[0]


def locate(path: str, key: str):
    """key 在文件里的行号 —— 用来给「把 key 填在第 12 行」这种提示。

    行号必须现算：模板改一行，写死在告警里的行号就变成误导。
    """
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
            _, linenos = _parse(f.read())
    except (OSError, EnvFileError):
        return None
    return linenos.get(key)


# =====================================================================
# 3. 加载
# =====================================================================
def default_path(root: str = None) -> str:
    root = root or os.path.dirname(os.path.abspath(__file__))
    return os.path.join(root, DEFAULT_RELATIVE_PATH)


def _digest(values: dict) -> str:
    """配置指纹。密钥被替换成 `<secret>`：能看出「有没有设」，看不出「是什么」。

    刻意**不**把密钥本身纳入摘要 —— 否则一个可以容忍泄露的 8 位摘要
    就变成了密钥的暴力破解目标，而它换不来任何多出来的归因能力
    （「是不是同一把 key」由 `secrets[*].fingerprint` 回答）。
    """
    lines = []
    for k in sorted(values):
        v = values[k]
        if is_secret(k) and v:
            shown = "<secret>"
        elif v == "":
            shown = "<empty>"
        else:
            shown = v
        lines.append("%s=%s" % (k, shown))
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def load_env_file(path: str, environ=None, override: bool = True) -> dict:
    """把 `path` 里的键写进 `environ`（默认 `os.environ`），返回一份**脱敏**报告。

    返回的 dict 是给实验产物用的（`results/<arm>/latest_run.json` 的 `llm_env`），
    所以它里面**不允许**出现密钥明文 —— 有任何字段会泄密钥，都是本项目的严重缺陷。

    文件不存在不报错（`exists=False` 就够了：可能是用户选了用户级环境变量那条路）。
    文件存在但**写坏了**则抛 `EnvFileError`，由调用方转成 fatal。
    """
    env = os.environ if environ is None else environ
    rep = {
        "path": str(path or ""), "exists": False, "keys": [], "applied": [],
        "kept_from_env": [], "conflicts": [], "unknown_keys": [],
        "empty_values": [], "error": None, "config_sha256": None, "secrets": {},
    }
    if not path:
        return rep
    if not os.path.isfile(path):
        return rep
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        values, _linenos = _parse(f.read())
    rep["exists"] = True

    for k, v in values.items():
        if k not in KNOWN_KEYS:
            rep["unknown_keys"].append(k)
        if is_secret(k):
            rep["secrets"][k] = {"present": bool(v), "fingerprint": fingerprint(v)}

    for k in sorted(values):
        v = values[k]
        if v == "":
            # 空值 = 没填。留住环境里可能存在的旧值，并记一笔让人看得见。
            rep["empty_values"].append(k)
            continue
        cur = env.get(k)
        if cur:
            if cur == v:
                rep["kept_from_env"].append(k)
            else:
                rep["conflicts"].append({
                    "key": k, "env_file": mask(v), "process_env": mask(cur),
                    "took": "env_file",
                    "hint": "进程环境里已有一个同名的不同值；按本模块的语义**文件优先**。"
                            "如果这不是你想要的，检查是不是设了用户级/会话级的环境变量。",
                })
        if override or not cur:
            env[k] = v
            rep["applied"].append(k)

    rep["keys"] = sorted(values)
    rep["config_sha256"] = _digest(values)
    return rep


def describe(rep: dict) -> str:
    """把报告压成一行中文，给前检查/human 读。"""
    if not rep.get("path"):
        return "未使用配置文件"
    if rep.get("error"):
        return "配置文件有错：%s" % rep["error"]
    if not rep.get("exists"):
        return "配置文件不存在：%s" % rep["path"]
    bits = ["%d 个键" % len(rep.get("keys") or [])]
    if rep.get("applied"):
        bits.append("新写入 %d" % len(rep["applied"]))
    if rep.get("empty_values"):
        bits.append("留空 %s" % ", ".join(rep["empty_values"]))
    if rep.get("unknown_keys"):
        bits.append("未知键 %s" % ", ".join(rep["unknown_keys"]))
    if rep.get("conflicts"):
        bits.append("与进程环境冲突 %s" % ", ".join(c["key"] for c in rep["conflicts"]))
    return "%s（%s）" % (rep["path"], "，".join(bits))
