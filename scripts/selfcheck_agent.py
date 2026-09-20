"""一次性导入自检（写完新模块立刻跑，避免把语法/导入错误留到 pytest 里）。

用法：
    <venv python> scripts\\selfcheck_agent.py
输出写到 logs/_agent_selfcheck.txt（本机 PowerShell 的 stdout 不回传）。
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "logs" / "_agent_selfcheck.txt"
lines: list[str] = []


def step(label, fn):
    try:
        value = fn()
        lines.append("OK   %-42s %s" % (label, value))
    except Exception as exc:  # noqa: BLE001
        lines.append("FAIL %-42s %s: %s" % (label, type(exc).__name__, exc))
        lines.append(traceback.format_exc())


step("import llm", lambda: __import__("llm"))
step("import llm.adapter", lambda: __import__("llm.adapter", fromlist=["x"]))
step("import llm.schema", lambda: __import__("llm.schema", fromlist=["x"]))
step("import llm.render", lambda: __import__("llm.render", fromlist=["x"]))
step("import agents", lambda: __import__("agents"))
step("import agents.executor", lambda: __import__("agents.executor", fromlist=["x"]))
step("import agents.memory", lambda: __import__("agents.memory", fromlist=["x"]))
step("import agents.synthesizer", lambda: __import__("agents.synthesizer", fromlist=["x"]))
step("import agents.loop", lambda: __import__("agents.loop", fromlist=["x"]))
step("import agents.prompts.system", lambda: __import__("agents.prompts.system", fromlist=["x"]))

from llm.schema import docs_text, tool_names  # noqa: E402

step("tool_names()", lambda: "%d 个: %s" % (len(tool_names()), ", ".join(tool_names())))
step("docs_text() 字符数", lambda: len(docs_text(modules=("math",))))
step("docs_text() 预览", lambda: "\n----\n" + docs_text(modules=("math",)) + "\n----")

from agents.executor import build_namespace, ALLOWED_MODULES  # noqa: E402
from tools.registry import ToolContext  # noqa: E402


def ns_probe():
    ns = build_namespace(ToolContext(), lambda *a, **k: None)
    leaked = sorted(set(ns) & {"ctx", "scene", "TOOL_REGISTRY", "open", "__import__"})
    return "命名空间 %d 个名字；泄漏 %s；可导入 %s" % (len(ns), leaked or "无", ALLOWED_MODULES)


step("build_namespace()", ns_probe)

from agents.prompts.system import build_system_prompt, build_user_prompt  # noqa: E402

step("build_system_prompt()", lambda: "%d 字符" % len(build_system_prompt(docs_text(), ALLOWED_MODULES)))
step("build_user_prompt()", lambda: build_user_prompt("哪把椅子离门最近？", {"objects": {"chair": 2}}, answer_type="float"))

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(lines), encoding="utf-8")
print("wrote %s" % OUT)
