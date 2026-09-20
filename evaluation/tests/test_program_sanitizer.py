#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成程序路径字面量消毒的测试。

背景（2026-09-18 首次端到端冒烟实测）
-------------------------------------
VADAR 把输出路径**插进要执行的源码**：

    with open("{result_file}", "w+") as result_file:

Windows 上 `result_file` 是 `D:\\3D_Spatial_Agent\\...`，于是这段字面量被
Python 按转义序列解析：`\\3`→\\x03、`\\202`→\\x82、`\\a`→BEL、`\\t`→TAB。
生成程序打不开自己的结果文件 → `[Errno 22] Invalid argument` → 该题记 0 分。

这套测试的重心不是「消毒函数有没有被调用」，而是：
**未消毒的程序在语义上到底打开哪条路径** —— 那条断言才是这个 bug 本身。

⚠ 本文件里所有 Windows 路径都必须写成 raw 字符串（`r"D:\\..."`）。
   写漏一个 `r`，测试自己就会变成那个 bug 的又一个受害者，
   而且症状是「断言莫名其妙不成立」，不会指向真正的原因。
"""

from __future__ import annotations

import ast
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evaluation import vadar_compat as vc  # noqa: E402

#: 故意用会踩满四种转义的路径：八进制 `\3`、`\202`，BEL `\a`，TAB `\t`。
NASTY = r"D:\3D_Spatial_Agent\results\A\2026-09-18_15-54-15\api_generator\trace.html"

PROGRAM = (
    "# WRITE NAMESPACE\n"
    "import json\n"
    'with open("' + NASTY + '", "w+") as result_file:\n'
    "    json.dump({}, result_file)\n"
)


def _opened_path(text):
    """把程序里第一处 `open(<字面量>` 的字面量取出来 —— 即**运行时真正用的值**。"""
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "open":
            return node.args[0].value
    raise AssertionError("程序里没有 open() 调用")


class TestTheBugItself:
    """先证明 bug 是真的，再证明修复是对的。少了前半段，后半段就只是自说自话。"""

    def test_unsanitized_program_opens_a_different_path(self):
        """未消毒 = 打开的不是那条路径。这条断言就是 bug 本身。

        注意 `\\3` 和 `\\202` 是**合法**转义（八进制），所以既不报错也不告警，
        只是悄悄换了一条路径 —— 这才是它危险的地方。
        """
        opened = _opened_path(PROGRAM)
        assert opened != NASTY
        assert "\x03" in opened, "\\3 应被解析成 \\x03"
        assert "\x82" in opened, "\\202 应被解析成 \\x82"
        assert "\x07" in opened, "\\a 应被解析成 BEL"
        assert "\t" in opened, "\\t 应被解析成 TAB"

    def test_sanitized_program_opens_exactly_the_intended_path(self):
        fixed, n = vc.sanitize_program_text(PROGRAM)
        assert n == 1
        assert _opened_path(fixed) == NASTY


class TestSanitizeText:
    def test_already_raw_is_not_double_prefixed(self):
        """已经 raw 的不能再加一个 `r` —— 否则 `rr"D:..."` 是语法错。"""
        already = 'with open(r"' + NASTY + '", "w+"):\n    pass\n'
        fixed, n = vc.sanitize_program_text(already)
        assert n == 0
        assert "rr" not in fixed
        assert _opened_path(fixed) == NASTY

    def test_single_quotes_are_covered(self):
        src = "open('" + NASTY + "', 'w+')\n"
        fixed, n = vc.sanitize_program_text(src)
        assert n == 1 and _opened_path(fixed) == NASTY

    def test_posix_and_relative_paths_are_left_alone(self):
        """只动带盘符的绝对路径。Linux 风格路径本来就是对的，改它只会引入风险。"""
        src = 'open("/home/u/results/a.json", "w+")\nopen("rel/b.json", "w+")\n'
        fixed, n = vc.sanitize_program_text(src)
        assert n == 0 and fixed == src

    def test_ordinary_escapes_untouched(self):
        """`\\n` 这种正文里的转义不能被当成路径。"""
        src = 'PREFIX = "line1\\nline2"\nSUFFIX = "a\\tb"\n'
        fixed, n = vc.sanitize_program_text(src)
        assert n == 0 and fixed == src

    def test_two_paths_on_one_line(self):
        src = 'a = "' + NASTY + '"; b = "' + NASTY + 'x"\n'
        fixed, n = vc.sanitize_program_text(src)
        assert n == 2


class TestSanitizeFile:
    def test_rewrites_in_place_and_counts(self, tmp_path):
        p = tmp_path / "executable_program.py"
        p.write_text(PROGRAM, encoding="utf-8")
        before = dict(vc.SANITIZER_STATS)
        vc.SANITIZER_STATS.update({"files": 0, "replacements": 0,
                                   "namespace_coercions": 0, "samples": []})

        res = vc.sanitize_program_file(str(p))
        assert res["paths"] == 1
        assert _opened_path(p.read_text(encoding="utf-8")) == NASTY
        assert vc.SANITIZER_STATS["files"] == 1
        assert vc.SANITIZER_STATS["replacements"] == 1
        assert vc.SANITIZER_STATS["samples"], "样本要留证，否则报告里无法回答「修了几个」"

        vc.SANITIZER_STATS.update(before)

    def test_already_clean_file_is_not_rewritten(self, tmp_path):
        p = tmp_path / "clean.py"
        src = 'open("rel/a.json", "w+")\n'
        p.write_text(src, encoding="utf-8")
        mtime = p.stat().st_mtime_ns
        assert vc.sanitize_program_file(str(p)) == {"paths": 0, "namespace_coercions": 0}
        assert p.stat().st_mtime_ns == mtime, "没有可修的就别动文件（避免假改动量）"

    def test_missing_file_returns_zero_instead_of_raising(self, tmp_path):
        """执行前消毒不能变成新的失败点 —— 文件不在就是 0，让 VADAR 自己报它的错。"""
        assert vc.sanitize_program_file(str(tmp_path / "nope.py")) \
            == {"paths": 0, "namespace_coercions": 0}

    def test_both_fixes_can_be_switched_off_independently(self, tmp_path):
        """两个修复各自可关 —— 保真对照臂需要能只开一个。"""
        p = tmp_path / "prog.py"
        p.write_text(PROGRAM + vc._NAMESPACE_LINE + "\n", encoding="utf-8")
        res = vc.sanitize_program_file(str(p), fix_coercion=False)
        assert res == {"paths": 1, "namespace_coercions": 0}

        p.write_text(PROGRAM + vc._NAMESPACE_LINE + "\n", encoding="utf-8")
        res = vc.sanitize_program_file(str(p), fix_paths=False)
        assert res == {"paths": 0, "namespace_coercions": 1}


class TestNamespaceCoercion:
    """「静默归零」的机制本身 —— 11/11 数值题空答案的根因。

    VADAR 用 `json.dumps` 能不能过来决定哪些变量写进 `result.json`。
    `np.float32` / `np.int64` / torch 0 维张量**过不去**，于是经过深度推出来的
    中间量和 `final_result` 一起消失；VADAR 的兜底是「没有 final_result 就记空答案」。

    ⟹ 正确算出来的答案被静默变成空字符串 —— 不报错、不重试、直接 0 分。
    所以这里必须断言**替换前确实丢、替换后确实留**，只断言「修完了」是不够的。
    """

    def _program(self, namespace_line):
        return ("import json\n"
                "def is_serializable(obj):\n"
                "    try:\n"
                "        json.dumps(obj)\n"
                "    except (TypeError, OverflowError):\n"
                "        return False\n"
                "    return True\n\n"
                + namespace_line + "\n")

    def _written(self, src, ns):
        g = dict(ns)
        exec(compile(src, "<generated_program>", "exec"), g)  # noqa: S102
        return g.get("serializable_globals", {})

    def test_numpy_scalars_are_dropped_without_the_repair(self):
        np = pytest.importorskip("numpy")
        got = self._written(self._program(vc._NAMESPACE_LINE),
                            {"final_result": np.float32(2.5), "k": 1.0})
        assert "final_result" not in got, "未修复时它就该被丢掉 —— 这就是 bug 本身"
        assert got.get("k") == 1.0, "纯 Python 的值照旧保留，所以表面上『一切正常』"

    def test_after_repair_the_answer_survives_as_a_python_scalar(self):
        np = pytest.importorskip("numpy")
        fixed, n = vc.repair_namespace_writer(self._program(vc._NAMESPACE_LINE))
        assert n == 1
        got = self._written(fixed, {"final_result": np.float32(2.5), "k": 1.0})
        assert got["final_result"] == pytest.approx(2.5)
        assert isinstance(got["final_result"], float), "必须是 Python float，不能只是能比较"

    def test_int64_and_torch_like_scalars_too(self):
        np = pytest.importorskip("numpy")
        fixed, _ = vc.repair_namespace_writer(self._program(vc._NAMESPACE_LINE))
        got = self._written(fixed, {"final_result": np.int64(7)})
        assert got["final_result"] == 7
        assert isinstance(got["final_result"], int)

    def test_arrays_are_still_filtered(self):
        """只解包 0 维标量。数组照旧过滤 —— 否则 result.json 会被整个点云撑爆。"""
        np = pytest.importorskip("numpy")
        fixed, _ = vc.repair_namespace_writer(self._program(vc._NAMESPACE_LINE))
        got = self._written(fixed, {"arr": np.zeros(3), "final_result": np.float32(1.5)})
        assert "arr" not in got
        assert got["final_result"] == pytest.approx(1.5)

    def test_absent_line_is_a_noop(self):
        fixed, n = vc.repair_namespace_writer("x = 1\n")
        assert n == 0 and fixed == "x = 1\n"

    def test_repair_is_idempotent(self):
        once, n1 = vc.repair_namespace_writer(self._program(vc._NAMESPACE_LINE))
        twice, n2 = vc.repair_namespace_writer(once)
        assert n1 == 1 and n2 == 0 and once == twice
        assert vc._NAMESPACE_LINE not in once, "原过滤行必须已经不在"


class TestWritePayloadIsActuallySerializable:
    """写盘那一步必须真的能写出来 —— 这是**上一版补丁自己踩的坑**。

    2026-09-18 实测：把 VADAR 的**字典推导**改写成
    `serializable_globals = {}` + `for` 循环之后，名字在读取 `globals()`
    **之前**就被绑定了，于是 `globals()["serializable_globals"]` 拿到的
    正是「要构建的那个字典」本身，循环体写下自引用：

        serializable_globals["serializable_globals"] = serializable_globals

    `is_serializable` 拦不住它（检查的那一瞬间那个字典还是空的），于是
    `json.dump` 走到最后一个键时抛 `ValueError: Circular reference detected`，
    把 `result.json` **截断在半路** —— 9/9 道题的文件都停在
    `"serializable_globals": ` 处。VADAR 读不出来 → 判为执行失败 →
    **重试整个程序生成**，每题多烧约 5 轮 LLM + 5 轮模型推理。

    上面那组用例只检查了「字典里有没有 `final_result`」，**没有把它序列化**，
    所以整类 bug 从断言底下溜了过去。这里补的就是那一刀。
    """

    HEAD = ("import json\n"
            "def is_serializable(obj):\n"
            "    try:\n"
            "        json.dumps(obj)\n"
            "    except (TypeError, OverflowError):\n"
            "        return False\n"
            "    return True\n\n")

    def _write_and_load(self, tmp_path, namespace_line, ns):
        """把 VADAR 那段写盘逻辑原样跑一遍，再**真的把文件读回来**。"""
        body = vc.repair_namespace_writer(self.HEAD + namespace_line + "\n")[0]
        out = tmp_path / "result.json"
        src = body + ("with open(%r, 'w+') as result_file:\n"
                      "    json.dump(serializable_globals, result_file)\n"
                      % str(out))
        exec(compile(src, "<generated_program>", "exec"), dict(ns))  # noqa: S102
        return json.loads(out.read_text(encoding="utf-8"))

    def test_repaired_program_writes_valid_json(self, tmp_path):
        payload = self._write_and_load(
            tmp_path, vc._NAMESPACE_LINE, {"final_result": 1.5, "cabinet": [[1, 2]]})
        assert payload["final_result"] == 1.5

    def test_payload_never_contains_itself(self, tmp_path):
        """自引用是**静默**的：它不改变任何正数值，只让整份文件写不出来。"""
        payload = self._write_and_load(tmp_path, vc._NAMESPACE_LINE, {"final_result": 1})
        assert "serializable_globals" not in payload, \
            "写盘产物里不该出现 serializable_globals 这个键 —— 它会指向自己"
        assert payload.get("serializable_globals") is not payload

    def test_vendor_original_line_is_also_clean(self, tmp_path):
        """VADAR 原版的字典推导没有这个毛病 —— 说明坑是**我们**引入的，
        不是它自带的。这条断言的意义在于：以后谁再动这段代码，
        会先看到「原版是干净的」。"""
        payload = self._write_and_load(
            tmp_path, vc._NAMESPACE_LINE, {"final_result": 1.5})
        assert "serializable_globals" not in payload

    def test_replacement_is_a_comprehension_not_a_prebound_loop(self):
        """结构性前提：**名字必须在读 `globals()` 之后才绑定**。

        这条守的是「将来有人为了可读性把它改回 for 循环」这件事 ——
        那是一次看起来完全无害的重构，代价是整轮实验慢 4 倍。
        """
        body = vc._NAMESPACE_REPLACEMENT
        assert "serializable_globals = {" in body
        assert "serializable_globals = {}" not in body
        assert "serializable_globals[_k] = " not in body


class TestExecuteFileWrapper:
    """包装器要同时吃下 VADAR 的两套签名，且**必须在原方法之前**消毒。"""

    def test_signature_with_path_argument(self, tmp_path, monkeypatch):
        seen = {}

        class A:
            def _execute_file(self, program_executable_path):
                seen["opened"] = _opened_path(
                    open(program_executable_path, encoding="utf-8").read())
                return "ran"

        monkeypatch.setattr(vc, "SANITIZER_STATS",
                            {"files": 0, "replacements": 0,
                             "namespace_coercions": 0, "samples": []})
        wrapped = vc._wrap_execute_file(A._execute_file)
        A._execute_file = wrapped

        p = tmp_path / "prog.py"
        p.write_text(PROGRAM, encoding="utf-8")
        assert A()._execute_file(str(p)) == "ran"
        assert seen["opened"] == NASTY, "原方法看到的必须已经是修好的路径"

    def test_signature_without_argument(self, tmp_path, monkeypatch):
        """engine.py 的版本不带参数，路径挂在 self 上。"""

        class E:
            def __init__(self, path):
                self.program_executable_path = path

            def _execute_file(self):
                return _opened_path(open(self.program_executable_path, encoding="utf-8").read())

        monkeypatch.setattr(vc, "SANITIZER_STATS",
                            {"files": 0, "replacements": 0,
                             "namespace_coercions": 0, "samples": []})
        E._execute_file = vc._wrap_execute_file(E._execute_file)

        p = tmp_path / "prog.py"
        p.write_text(PROGRAM, encoding="utf-8")
        assert E(str(p))._execute_file() == NASTY

    def test_wrapper_is_marked_for_idempotency(self):
        class A:
            def _execute_file(self, p):
                return p

        w = vc._wrap_execute_file(A._execute_file)
        assert getattr(w, "__vadar_path_sanitized__", False) is True


class TestInstallSwitch:
    def test_can_be_disabled_for_fidelity_runs(self, monkeypatch):
        """`VADAR_FIX_PATH_ESCAPE=0` 要能关掉，且**不去 import agents/engine**。

        这条顺带守住一件事：本模块关掉后必须零副作用，否则「保真对照臂」
        就没法在纯净环境里跑。
        """
        monkeypatch.setenv("VADAR_FIX_PATH_ESCAPE", "0")
        rep = vc.install_program_sanitizer(verbose=False)
        assert rep == {"enabled": False, "patched": []}


class TestTemplateShapeIsWhatWeThinkItIs:
    """把「VADAR 的模板长什么样」也钉住。

    如果哪天 vendor 升级、模板改成 `open({path!r}, ...)` 或 `os.fspath`，
    消毒器会变成**空操作而不报错**。那时应该有人来看一眼，
    而不是让消毒计数悄悄归零、实验分数悄悄变 0。
    """

    def test_vendor_template_still_interpolates_a_bare_path(self):
        root = os.path.join(vc.VADAR_REPO_ROOT, "agents", "agents.py")
        # 这里**故意不用 skip**：本项目的规矩是「看到 skipped 当失败处理」，
        # 而一条自己会退化成 skip 的哨兵用例，正是这条规矩要防的东西。
        assert os.path.isfile(root), "vendor/VADAR 缺失 —— 本仓库不该出现这种检出"
        text = open(root, encoding="utf-8", errors="replace").read()
        assert 'with open("{result_file}", "w+")' in text, \
            "VADAR 模板变了：消毒器的前提不再成立，需要重新评估（不是简单改断言）"
        assert vc._NAMESPACE_LINE in text.replace("{{", "{").replace("}}", "}"), \
            "VADAR 的写盘过滤行变了：静默丢 numpy 标量这件事可能已不复存在（或换了形式）"
