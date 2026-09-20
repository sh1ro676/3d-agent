"""`agents/memory.py` 的单测 + **动作空间三方一致性** + 分层禁令的 AST 检查。

这组测试里有三条是"架构级"的，值得单独说：

**① 动作空间必须三方一致。**
提示词列出的工具 / 静态检查放行的工具 / 运行期命名空间里的工具 ——
三者不一致的后果是**静默的**：提示词里列了但命名空间没有 → 模型照着写 → `NameError`
→ 表现成「模型不会写程序」。所以这里直接断言三个集合相等。

**② `tools/` 与 `scene_graph/` 不许 import `llm`**（§13.3(6) 禁令三：
几何层永不调用 LLM）。这条用 AST 静态检查守，不靠人记。

**③ `agents/` 不许 import `evaluation/`。**
evaluation 是**测量** agent 的器械；被测方依赖测量方会把方向反过来。

为什么用 AST 而不是"跑一下看看"：这类约束破掉的时候**不会报错**，
只会在某次重构之后悄悄多出一条依赖边。静态检查是唯一能提前发现它的方式。
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.executor import META_TOOLS, QA_TOOLSET, build_namespace  # noqa: E402
from agents.memory import MAX_FEEDBACK_ATTEMPTS, WorkingMemory  # noqa: E402
from agents.synthesizer import static_check  # noqa: E402
from llm.schema import docs_text, tool_docs  # noqa: E402
from tools.registry import ToolContext, tool_names  # noqa: E402

#: 不写返回值形状的工具 —— 目前**没有**，这个集合是留给后来者的显式出口。
#: 有了它，「这条工具没写形状」就必须是一次有意识的决定（写明理由），
#: 而不是一次遗漏 —— 遗漏的代价实测是 5 倍冗余调用（见
#: `test_every_tool_documents_its_return_shape` 的 docstring）。
DOC_SHAPE_EXEMPT: frozenset[str] = frozenset()


# ---------------------------------------------------------------------------
# 工作记忆
# ---------------------------------------------------------------------------


class TestWorkingMemory:
    def test_empty_memory_has_no_feedback(self):
        assert WorkingMemory().feedback() is None

    def test_feedback_keeps_only_the_recent_attempts(self):
        mem = WorkingMemory()
        for i in range(5):
            mem.record_attempt("execute:runtime", "错误 %d" % i)
        text = mem.feedback()
        assert "%s" % "错误 4" in text and "错误 3" in text
        assert "错误 0" not in text
        assert "还有 3 次更早的失败" in text
        assert len(mem.attempts) == 5          # 记录本身是全量保留的

    def test_feedback_max_attempts_is_configurable(self):
        mem = WorkingMemory()
        for i in range(3):
            mem.record_attempt("static_check", "e%d" % i)
        assert "e0" in mem.feedback(max_attempts=3)

    def test_long_messages_are_clipped(self):
        mem = WorkingMemory()
        a = mem.record_attempt("execute:runtime", "x" * 5000)
        assert len(a.message) <= 400 and a.message.endswith("（已截断）")

    def test_attempt_line_carries_line_code_and_recovery(self):
        mem = WorkingMemory()
        a = mem.record_attempt("execute:runtime", "boom", code="NOT_IN_SCENE",
                               lineno=12, recovery=("read_scene",), hint="用 list_objects")
        text = a.line()
        for needle in ("第 1 次", "第 12 行", "NOT_IN_SCENE", "read_scene", "list_objects"):
            assert needle in text

    def test_learn_from_trace_collects_ids_from_successful_reads(self):
        mem = WorkingMemory()
        trace = [
            {"tool": "list_objects", "result": {"ok": True, "value": [
                {"object_id": "chair_1"}, {"object_id": "door_1"}]}},
            {"tool": "find_nearest", "result": {"ok": True, "value": [
                {"object_id": "chair_2"}]}},
        ]
        learned = mem.learn_from_trace(trace)
        assert set(learned) == {"chair_1", "door_1", "chair_2"}
        assert mem.confirmed_ids == {"chair_1", "door_1", "chair_2"}

    def test_learn_from_trace_uses_known_ids_from_the_hallucination_error(self):
        """`NOT_IN_SCENE` 的 `context["known_ids"]` 是最权威的一份清单 —— 也要学。"""
        mem = WorkingMemory()
        mem.learn_from_trace([{"tool": "get_3d_position", "result": {
            "ok": False, "error": {"code": "NOT_IN_SCENE",
                                   "context": {"known_ids": ["sofa_1", "table_1"]}}}}])
        assert mem.confirmed_ids == {"sofa_1", "table_1"}

    def test_learn_from_trace_deduplicates_and_reports_only_new(self):
        mem = WorkingMemory()
        first = mem.learn_from_trace([{"tool": "list_objects", "result": {
            "ok": True, "value": [{"object_id": "a"}]}}])
        again = mem.learn_from_trace([{"tool": "list_objects", "result": {
            "ok": True, "value": [{"object_id": "a"}]}}])
        assert first == ["a"] and again == []

    def test_confirmed_ids_reach_the_feedback(self):
        mem = WorkingMemory()
        mem.record_attempt("execute:runtime", "no such id")
        mem.learn_from_trace([{"tool": "list_objects", "result": {
            "ok": True, "value": [{"object_id": "chair_1"}]}}])
        assert "chair_1" in mem.feedback()

    def test_confirmed_ids_are_truncated_but_counted(self):
        mem = WorkingMemory()
        mem.record_attempt("execute:runtime", "x")
        mem.confirmed_ids.update("obj_%03d" % i for i in range(40))
        text = mem.feedback()
        assert "共 40 个" in text

    def test_reset(self):
        mem = WorkingMemory()
        mem.record_attempt("static_check", "x")
        mem.confirmed_ids.add("a")
        mem.reset()
        assert mem.attempts == [] and mem.confirmed_ids == set()

    def test_to_dict(self):
        mem = WorkingMemory(question="q", scene_hint={"objects": {"chair": 2}})
        mem.record_attempt("static_check", "e")
        d = mem.to_dict()
        assert d["question"] == "q" and d["n_attempts"] == 1 and d["attempts"][0]["stage"] == "static_check"

    def test_max_feedback_attempts_default(self):
        assert MAX_FEEDBACK_ATTEMPTS == 2


# ---------------------------------------------------------------------------
# 动作空间三方一致
# ---------------------------------------------------------------------------


class TestToolsetAgreement:
    @pytest.fixture
    def ctx(self):
        from scene_graph.schema import SceneGraph

        return ToolContext(scene=SceneGraph(scene_id="s", image_id="i"))

    def test_prompt_tools_equal_static_check_tools(self, ctx):
        ns = build_namespace(ctx, lambda *a, **k: None)
        exposed = {n for n in ns if not n.startswith("__") and n in set(tool_names())}
        assert exposed == set(QA_TOOLSET)

    def test_static_check_rejects_everything_not_in_the_namespace(self, ctx):
        ns = build_namespace(ctx, lambda *a, **k: None)
        for name in tool_names():
            src = "r = %s()\nsubmit(1, evidence=['x'])" % name
            allowed = static_check(src, tools=QA_TOOLSET).ok
            present = name in ns
            assert allowed == present, (
                "%s：静态检查放行=%s 但命名空间存在=%s —— 三者必须一致" % (name, allowed, present))

    def test_prompt_lists_exactly_the_action_space(self):
        text = docs_text(tools=QA_TOOLSET)
        for name in QA_TOOLSET:
            assert "%s(" % name in text, name
        for name in META_TOOLS:
            assert "%s(" % name not in text, name

    def test_default_docs_are_stable_across_calls(self):
        """逐字节稳定 → 提示词前缀可被服务端缓存，也保证两次实验可比。"""
        assert docs_text(tools=QA_TOOLSET) == docs_text(tools=QA_TOOLSET)

    def test_qa_toolset_and_meta_tools_are_disjoint(self):
        assert not (set(QA_TOOLSET) & set(META_TOOLS))
        assert set(QA_TOOLSET) | set(META_TOOLS) == set(tool_names())

    def test_docs_are_compact(self):
        """prompt 长度 = 成本。VADAR 的 program prompt 是 6965 字符，这里必须显著更短。"""
        assert len(docs_text(tools=QA_TOOLSET)) < 2500

    def test_every_tool_documents_its_return_shape(self):
        """★ 契约必须写在**模型会读的那一段**里。

        实测（2026-09-19，`reports/combination_run_*_reanalyzed.md`）：
        `list_objects` 的文档当时只写「列出场景中的物体」——一个字都没提 value 里带
        `extent_m`。模型于是为每个物体又调一次 `get_3d_extent`：10 道组合题用了
        **106 次**调用，最少 **26 次**就够，冗余倍数中位数 **5.0**。
        形状其实写在 system prompt 的规则 2 里，但**每条工具文档自己不说**；
        两条信息源在具体程度上不一致时，模型跟的是更具体的那条。

        ⚠ 这条断言只能保证「写了 `res.value`」，保证不了「写对了」——
        写对与否由 `scripts/probe_combination.py` 的字段清点（第四节）与
        `tests/test_tools.py` 里钉住 `_brief` 键集的用例共同守着。
        """
        missing = [d.name for d in tool_docs(QA_TOOLSET)
                   if d.name not in DOC_SHAPE_EXEMPT and "res.value" not in d.summary]
        assert not missing, (
            "这些工具的第一段没写 `res.value` 的形状：%s\n"
            "第一段正是渲染进提示词的那一段（`llm/schema.py::first_paragraph`），"
            "写在下面的设计说明模型一个字也看不到。\n"
            "确实不返回形状的工具（例如只产生副作用的），请显式加进 `DOC_SHAPE_EXEMPT` "
            "并写明理由 —— 让它成为一次有意识的决定。" % missing)

    def test_the_two_fields_that_caused_five_x_redundancy_are_named(self):
        """`list_objects` 的文档必须点名 `centroid_m` 与 `extent_m`。

        这正是那个 5 倍冗余的直接原因：返回值里已经有这两个量，文档却没说，
        于是模型去调 `get_3d_extent`（那条文档明确承诺了 `(w, h, l)`）。
        """
        by_name = {d.name: d.summary for d in tool_docs(QA_TOOLSET)}
        for field in ("centroid_m", "extent_m"):
            assert field in by_name["list_objects"], field


# ---------------------------------------------------------------------------
# 分层禁令（AST 静态检查）
# ---------------------------------------------------------------------------


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                found.add(node.module.split(".")[0])
            elif node.level:                       # from . import x → 包内
                found.add(".")
    return found


def _py_files(pkg: str):
    return sorted((ROOT / pkg).rglob("*.py"))


class TestLayering:
    def test_geometry_layer_never_imports_llm(self):
        """禁令三：几何层永不调用 LLM —— 它只读 SceneGraph，不知道 LLM 存在。"""
        offenders = []
        for pkg in ("tools", "scene_graph"):
            for f in _py_files(pkg):
                if "llm" in _imported_modules(f) or "agents" in _imported_modules(f):
                    offenders.append(str(f.relative_to(ROOT)))
        assert offenders == [], "几何层/工具层不得依赖 llm 或 agents：%s" % offenders

    def test_agents_never_import_evaluation(self):
        """被测方不许依赖测量器械。"""
        offenders = [str(f.relative_to(ROOT)) for f in _py_files("agents")
                     if "evaluation" in _imported_modules(f)]
        assert offenders == []

    def test_pure_llm_modules_do_not_know_the_domain(self):
        """`adapter` / `render` 只负责跟端点说话，不认识工具库与场景图。"""
        for name in ("llm/adapter.py", "llm/render.py"):
            mods = _imported_modules(ROOT / name)
            assert "tools" not in mods, name
            assert "scene_graph" not in mods, name

    def test_no_new_module_pulls_in_torch_at_import(self):
        """新模块必须能在没有 torch 的进程里 import —— 8GB 机器上这是几秒的差别。"""
        import subprocess

        code = (
            "import sys;"
            "sys.path.insert(0, r'%s');"
            "import llm.adapter, llm.schema, llm.render, agents.executor, agents.memory,"
            " agents.synthesizer, agents.loop, agents.prompts.system;"
            "print('torch' in sys.modules)"
        ) % ROOT
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
        assert out.returncode == 0, out.stderr[-2000:]
        assert out.stdout.strip().endswith("False"), out.stdout
