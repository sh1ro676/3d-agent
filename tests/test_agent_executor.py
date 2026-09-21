"""`agents/executor.py` 的单元测试 —— 零 GPU、零联网、零 LLM。

这组测试盯的是**四个「静默归零」级的缺陷**（都不报错，直接把正确答案变成 0 分）：

    ① 程序能读到 `ctx.scene` → 「坐标只能来自工具返回值」变成提示词祈祷
    ② `open` 可用 → 绝对路径被当转义字符（Windows `\\3D`→`\\x03`），生成程序整段作废
    ③ 程序正常结束但没有 `submit` → 早期基线静默给空串并算错
    ④ 几何算飞了给出 `nan` → 一路活到最后，变成一个"看着像数字"的答案

另外还钉住三件事：`submit` 的证据下限、`answer_type` 的前置约束、
以及失败归类（`stage`）必须区分得开 —— 混成一个异常字符串就没法做失败诊断了。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.executor import (  # noqa: E402
    ALLOWED_MODULES,
    QA_TOOLSET,
    _Submitted,
    build_namespace,
    execute_program,
    file_line_of,
)
from scene_graph.schema import BBox3D, Node, SceneGraph  # noqa: E402
from tools import load_tools  # noqa: E402
from tools.registry import ToolContext  # noqa: E402

load_tools()


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------


def mk(node_id, label, xyz, size=(0.5, 0.9, 0.5), score=0.8) -> Node:
    x, y, z = xyz
    w, h, l = size
    return Node(
        id=node_id, label=label, score=score,
        bbox_2d=(10.0, 10.0, 50.0, 60.0), mask_ref=f"masks/{node_id}.png",
        centroid_3d=(x, y, z), extent_3d=(w, h, l),
        bbox_3d=BBox3D(min=(x - w / 2, y - h / 2, z - l / 2), max=(x + w / 2, y + h / 2, z + l / 2)),
        n_points=1000,
    )


@pytest.fixture
def scene() -> SceneGraph:
    return SceneGraph(
        scene_id="unit_scene", image_id="rgb.png", up_axis="-y",
        nodes=(
            mk("door_1", "door", (-1.2, 0.0, 1.4), (0.9, 2.0, 0.1), 0.91),
            mk("chair_1", "chair", (0.8, 0.1, 2.9)),
            mk("chair_2", "chair", (1.6, 0.1, 2.2)),
            mk("sofa_1", "sofa", (0.31, -0.42, 2.86), (1.92, 0.78, 0.91)),
        ),
        build_meta={"image_size": [640, 480]},
    )


@pytest.fixture
def ctx(scene) -> ToolContext:
    return ToolContext(scene=scene, record_trace=True)


# ---------------------------------------------------------------------------
# 沙箱不变量
# ---------------------------------------------------------------------------


class TestSandbox:
    def test_ctx_is_not_reachable(self, ctx):
        """★ 最重要的那条：程序拿不到 SceneGraph，也就抄不到真坐标。"""
        ns = build_namespace(ctx, lambda *a, **k: None)
        assert "ctx" not in ns
        assert not (set(ns) & {"scene", "SceneGraph", "TOOL_REGISTRY", "np", "torch"})
        # 连间接路径也堵上
        out = execute_program(
            "submit(len(list_objects().value), evidence=['count'])", ctx)
        assert out.ok and out.submission.answer == 4

    def test_open_is_unavailable(self, ctx):
        """反面：早期基线的 `open("{result_file}",...)` 就是从这里炸的。"""
        out = execute_program("f = open('x.txt', 'w')", ctx)
        assert out.ok is False and out.stage == "runtime"
        assert "NameError" in out.message or "open" in out.message

    def test_import_outside_allowlist_is_a_clear_error(self, ctx):
        out = execute_program("import os\nsubmit(1, evidence=['x'])", ctx)
        assert out.ok is False and out.stage == "runtime"
        assert "白名单" in out.message

    def test_import_inside_allowlist_works(self, ctx):
        out = execute_program(
            "submit(round(math.sqrt(2), 3), evidence=['math.sqrt(2)'])", ctx)
        assert out.ok and out.submission.answer == 1.414

    def test_all_allowed_modules_are_importable(self, ctx):
        for mod in ALLOWED_MODULES:
            out = execute_program("import %s\nsubmit(1, evidence=['ok'])" % mod, ctx)
            assert out.ok, "%s 应该可导入，实际 %s" % (mod, out.stage)

    def test_toolset_is_exactly_what_is_exposed(self, ctx):
        ns = build_namespace(ctx, lambda *a, **k: None)
        present = {n for n in ns if not n.startswith("__")} & set(QA_TOOLSET)
        assert present == set(QA_TOOLSET)
        assert "describe_scene" not in ns          # L5 元工具不进动作空间

    def test_unknown_tool_in_toolset_raises_immediately(self, ctx):
        """提示词与运行期不一致必须当场响，不能等模型写出 NameError。"""
        with pytest.raises(KeyError, match="未注册的工具"):
            build_namespace(ctx, lambda *a, **k: None, toolset=("list_objects", "no_such_tool"))


# ---------------------------------------------------------------------------
# submit 契约
# ---------------------------------------------------------------------------


class TestSubmitContract:
    def test_happy_path_records_evidence_and_targets(self, ctx):
        out = execute_program(
            "res = list_objects()\n"
            "ids = [o['object_id'] for o in res.value]\n"
            "submit(ids[0], target_ids=ids[:2], evidence=['list_objects() 第 0 个'])", ctx)
        assert out.ok and out.stage == "ok"
        s = out.submission
        assert s.answer == "door_1" and s.target_ids == ("door_1", "chair_1")
        assert s.evidence == ("list_objects() 第 0 个",)

    def test_evidence_is_mandatory(self, ctx):
        out = execute_program("submit(1.0)", ctx)
        assert out.ok is False and out.stage == "contract"
        assert "evidence" in out.message

    def test_empty_evidence_list_is_also_rejected(self, ctx):
        out = execute_program("submit(1.0, evidence=[])", ctx)
        assert out.ok is False and out.stage == "contract"

    def test_none_answer_is_rejected_with_a_way_out(self, ctx):
        out = execute_program("submit(None, evidence=['x'])", ctx)
        assert out.ok is False and out.stage == "contract"
        assert "unknown" in out.message

    def test_submit_ends_the_program(self, ctx):
        """`submit` 是 NoReturn —— 后面的代码不许再跑。"""
        out = execute_program(
            "submit(1, evidence=['x'])\nprint('SHOULD NOT PRINT')", ctx)
        assert out.ok and "SHOULD NOT PRINT" not in out.stdout

    def test_submit_inside_a_loop_still_ends_it(self, ctx):
        out = execute_program(
            "for i in range(100):\n"
            "    if i == 3:\n"
            "        submit(i, evidence=['found at i=3'])\n", ctx)
        assert out.ok and out.submission.answer == 3

    def test_unknown_target_ids_are_reported_but_not_fatal(self, ctx):
        """`target_ids` 驱动 Viewer 高亮，编的 id 不该让答案作废，但必须留下痕迹。"""
        out = execute_program("submit(1, target_ids=['ghost_1'], evidence=['x'])", ctx)
        assert out.ok and out.submission.unknown_targets == ("ghost_1",)

    def test_abstain_is_a_first_class_outcome(self, ctx):
        out = execute_program('submit("unknown", evidence=["场景里没有 door"])', ctx)
        assert out.ok and out.submission.abstained is True
        assert out.submission.answer == "unknown"

    def test_abstain_skips_answer_type_coercion(self, ctx):
        out = execute_program('submit("unknown", evidence=["无"]); ', ctx, answer_type="float")
        assert out.ok and out.submission.abstained is True


class TestControlFlowSignalIsNotCatchable:
    """`submit()` 靠抛异常终止程序 —— 那个异常绝不能被**程序自己**接住。

    为什么值得单独一个类：`SAFE_BUILTIN_NAMES` 主动把 `Exception` 交给了模型
    （"错误恢复"是本项目的观测点，不给异常类型就写不出 try/except），
    于是模型很自然会写出这种兜底：

        try:
            submit(calculate_distance(a, b), evidence=["..."])
        except Exception:
            submit("unknown", evidence=["算不出来"])

    控制流信号一旦被它接住，两个后果都是**静默的**（本机实测）：

      · 算对的答案被兜底改写成 `"unknown"` ⟹ 落盘 `abstained=true`。
        模型没有弃答，是**记录说它弃答了** —— 直接污染"弃答率"这个硬指标。
      · 或者程序跑到结尾落进 `no_submit` ⟹ **已提交的答案被判成没提交**。

    三道防线，各挡一种写法（见 `agents/executor.py` 的 `_Submitted` docstring）：
      · `except Exception`            → `_Submitted` 继承 `BaseException` 挡住；
      · 裸 `except:` + 重新 submit    → `_SubmitBox` 的"已提交不再改写"守卫挡住；
        （裸 `except:` 是 Python 里唯一能捕获 `BaseException` 且**不需要异常名**的
          写法，所以"白名单里没有 BaseException"拦不住它 —— 这才需要守卫。）
      · 裸 `except: pass` 后跑完      → `execute_program` 认 `box.submission` 兜住。
    """

    def test_except_exception_cannot_swallow_submit(self, ctx):
        """最常见的兜底写法。修复前这里拿到的是 answer='unknown' / abstained=True。"""
        out = execute_program(
            "try:\n"
            "    submit(1.5, evidence=['d = 1.5'])\n"
            "except Exception:\n"
            "    submit('unknown', evidence=['fallback'])\n", ctx)
        assert out.ok, out.message
        assert out.submission.answer == 1.5
        assert out.submission.abstained is False

    def test_bare_except_cannot_rewrite_the_answer(self, ctx):
        """守卫：保留**第一份**答案，兜底分支里的 'unknown' 不许覆盖它。"""
        out = execute_program(
            "try:\n"
            "    submit(1.5, evidence=['d = 1.5'])\n"
            "except:\n"
            "    submit('unknown', evidence=['fallback'])\n", ctx)
        assert out.ok, out.message
        assert out.submission.answer == 1.5
        assert out.submission.abstained is False

    def test_bare_except_pass_still_returns_the_answer(self, ctx):
        """控制流被吞、程序跑完 —— 落进 `no_submit` 之前先认 `box.submission`。
        修复前这里是 ok=False / stage='no_submit' / answer=None。"""
        out = execute_program(
            "try:\n"
            "    submit(1.5, evidence=['d = 1.5'])\n"
            "except:\n"
            "    pass\n"
            "x = 1 + 1\n", ctx)
        assert out.ok and out.stage == "ok", out.message
        assert out.submission.answer == 1.5
        # 记录里必须留下痕迹：不能让它看起来像"什么都没发生过"。
        assert "吞掉" in out.message

    def test_never_submitted_is_still_a_failure(self, ctx):
        """反向边界：兜底逻辑**不许**把"一次都没提交"救回来 —— 那才是真·没交答卷。"""
        out = execute_program("x = 1 + 1\n", ctx)
        assert out.ok is False and out.stage == "no_submit"
        assert out.submission is None

    def test_base_exception_is_not_exposed_to_programs(self, ctx):
        """白名单里没有 `BaseException` ⟹ 除裸 `except:` 外没有别的捕获路径。
        这条不是"顺便测一下"：它决定了上面那条守卫是**最后一道**防线。"""
        ns = build_namespace(ctx, lambda *a, **k: None)
        builtins_ns = ns["__builtins__"]
        assert "BaseException" not in builtins_ns
        assert "SystemExit" not in builtins_ns
        assert issubclass(_Submitted, BaseException)
        assert not issubclass(_Submitted, Exception)


class TestAnswerTypeCoercion:
    def test_int_accepts_float_string(self, ctx):
        assert execute_program('submit("3", evidence=["x"])', ctx, answer_type="int") \
            .submission.answer == 3

    def test_float_truncation_is_not_silent(self, ctx):
        out = execute_program('submit("大约 2.5 米", evidence=["x"])', ctx, answer_type="float")
        assert out.ok is False and out.stage == "contract"
        assert "无法解析成数" in out.message

    def test_bool_is_refused_for_numeric_answers(self, ctx):
        """bool 是 int 的子类 —— 静默当成 1/0 会把「判了真假」伪装成「算出了数」。"""
        out = execute_program("submit(True, evidence=['x'])", ctx, answer_type="float")
        assert out.ok is False and "bool" in out.message

    def test_nan_is_refused(self, ctx):
        """④ 挡 nan：它会一路活到最后，变成一个看着像数字的答案。"""
        out = execute_program("submit(float('nan'), evidence=['x'])", ctx, answer_type="float")
        assert out.ok is False and "有限数" in out.message

    def test_inf_is_refused(self, ctx):
        out = execute_program("submit(1e999, evidence=['x'])", ctx, answer_type="float")
        assert out.ok is False and out.stage == "contract"

    def test_bool_answer_type(self, ctx):
        assert execute_program('submit("true", evidence=["x"])', ctx, answer_type="bool") \
            .submission.answer is True

    def test_str_answer_type_keeps_text(self, ctx):
        out = execute_program('submit("blue", evidence=["vlm"])', ctx, answer_type="str")
        assert out.submission.answer == "blue"


# ---------------------------------------------------------------------------
# 失败归类
# ---------------------------------------------------------------------------


class TestStages:
    def test_syntax_error_has_lineno(self, ctx):
        out = execute_program("def broken(:\n    pass", ctx)
        assert out.stage == "syntax" and out.lineno is not None
        assert out.tool_calls == 0

    def test_no_submit_is_an_explicit_failure(self, ctx):
        """③ 这是「静默给 0 分」的那条路，这里必须明确判失败。"""
        out = execute_program("x = 1 + 1", ctx)
        assert out.ok is False and out.stage == "no_submit"
        assert "final_result" in out.message or "submit" in out.message

    def test_runtime_error_points_at_the_user_line(self, ctx):
        """报错行必须是**模型写的那一行**，不是工具内部的行。"""
        src = "res = list_objects()\nvalues = res.value[999]\nsubmit(1, evidence=['x'])"
        out = execute_program(src, ctx, source_label="<unit_program>")
        assert out.ok is False and out.stage == "runtime"
        assert out.lineno == 2
        assert "IndexError" in out.message

    def test_hallucinated_id_is_recorded_in_the_trace(self, ctx):
        """程序即使**不判 `res.ok`** 继续往下走，幻觉事实也留在了 trace 里。"""
        src = "res = get_3d_position(object_id='chair_9')\nsubmit(1, evidence=['x'])"
        out = execute_program(src, ctx)
        assert out.to_dict()["trace_errors"].get("NOT_IN_SCENE") == 1

    def test_ignoring_a_tool_failure_is_loud_not_silent(self, ctx):
        """不判 `res.ok` 就取 `res.value` → None → 契约报错。

        ★ 这一条是「静默失败」的正面反例：早期基线会拿一个空值继续算，
        这里必须在交答案的那一刻停住。同时 trace 里留着真实的错误码。
        """
        src = "res = get_3d_position(object_id='chair_9')\nsubmit(res.value, evidence=['x'])"
        out = execute_program(src, ctx)
        assert out.ok is False and out.stage == "contract"
        assert out.to_dict()["trace_errors"] == {"NOT_IN_SCENE": 1}

    def test_program_can_recover_from_hallucination(self, ctx):
        """错误可恢复性是本项目的观测点：程序读到错误码后换路走通。"""
        src = (
            "res = get_3d_position(object_id='chair_9')\n"
            "if not res.ok:\n"
            "    good = list_objects(label='chair').value[0]['object_id']\n"
            "    res = get_3d_position(object_id=good)\n"
            "submit(len(res.value), evidence=['fallback after NOT_IN_SCENE'])"
        )
        out = execute_program(src, ctx)
        assert out.ok and out.submission.answer == 3
        assert out.to_dict()["trace_errors"] == {"NOT_IN_SCENE": 1}

    def test_timeout_is_caught_by_the_watchdog(self, ctx):
        out = execute_program("n = 0\nwhile True:\n    n += 1", ctx, timeout_s=0.4)
        assert out.ok is False and out.stage == "timeout"
        assert "超时" in out.message or "未结束" in out.message

    def test_timeout_does_not_leak_into_the_next_call(self, ctx):
        """看门狗必须只打一枪 —— 否则超时的那一次会污染下一次执行。"""
        execute_program("while True:\n    pass", ctx, timeout_s=0.4)
        out = execute_program("submit(7, evidence=['after timeout'])", ctx)
        assert out.ok and out.submission.answer == 7

    def test_recursion_error_is_classified(self, ctx):
        out = execute_program(
            "def f(n):\n    return f(n + 1)\nf(0)", ctx)
        assert out.ok is False and out.stage == "runtime"
        assert "Recursion" in out.message

    def test_print_does_not_pollute_stdout_but_is_kept(self, ctx):
        out = execute_program("print('debug 42')\nsubmit(1, evidence=['x'])", ctx)
        assert out.ok and "debug 42" in out.stdout


class TestTraceAndTiming:
    def test_trace_slice_only_covers_this_run(self, ctx):
        execute_program("list_objects()", ctx)
        out = execute_program("list_objects(label='chair')", ctx)
        assert out.tool_calls == 1 and len(out.trace) == 1

    def test_duration_is_recorded(self, ctx):
        out = execute_program("submit(1, evidence=['x'])", ctx)
        assert out.duration_ms >= 0.0

    def test_file_line_of_ignores_frames_outside_the_source(self, ctx):
        """工具内部抛的异常不该被算成"用户代码第 N 行"。"""
        try:
            raise ValueError("x")
        except ValueError as exc:
            assert file_line_of(exc, "<unit_program>") is None
