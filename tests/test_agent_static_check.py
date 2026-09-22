"""`agents/synthesizer.static_check()` 的单元测试 —— 零成本、不执行、不联网。

静态检查是**在花钱之前**回答「模型守不守协议」的那个东西，所以它的价值全在准确：
**漏报**会让坏程序进沙箱（白跑一次 GPU），**误报**会让好程序被拒（模型开始瞎改）。
两端都各写了反面用例。

独立于工具集：用 `tools=` / `params=` 注入一个只有两个工具的假世界 ——
检查逻辑与真实注册表解耦，注册表加减工具不会让这组测试失效。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.executor import ALLOWED_MODULES, QA_TOOLSET  # noqa: E402
from agents.synthesizer import extract_code, static_check  # noqa: E402
from llm.schema import params_of  # noqa: E402

FAKE_TOOLS = ("list_objects", "calculate_distance")
FAKE_PARAMS = {
    "list_objects": ("scene_id", "label", "limit"),
    "calculate_distance": ("scene_id", "a", "b"),
}


def check(src: str):
    return static_check(src, tools=FAKE_TOOLS, params=FAKE_PARAMS)


GOOD = (
    "objs = list_objects(label='chair')\n"
    "if not objs.ok:\n"
    "    submit('unknown', evidence=['no chair'])\n"
    "ids = [o['object_id'] for o in objs.value]\n"
    "d = calculate_distance(a=ids[0], b=ids[1])\n"
    "submit(d.value, evidence=['distance'])\n"
)


# ---------------------------------------------------------------------------
# 四项检查
# ---------------------------------------------------------------------------


class TestFourChecks:
    def test_good_program_passes(self):
        r = check(GOOD)
        assert r.ok and r.parse_ok and r.has_submit
        assert r.tool_calls == ("list_objects", "calculate_distance")

    def test_check_1_syntax(self):
        r = check("def f(:\n  pass")
        assert r.parse_ok is False and not r.ok
        assert r.syntax_error and "第" in r.syntax_error
        assert "语法错误" in r.feedback()

    def test_check_2_unknown_function(self):
        r = check("x = compute_distance('a', 'b')\nsubmit(1, evidence=['x'])")
        assert not r.ok and r.unknown_calls == ("compute_distance",)
        assert "未知函数" in r.feedback()

    def test_check_3_unknown_kwarg_lists_the_real_ones(self):
        r = check("res = list_objects(labels='chair')\nsubmit(1, evidence=['x'])")
        assert not r.ok and r.bad_kwargs == (("list_objects", "labels"),)
        assert "可用参数" in r.feedback()

    def test_check_4_missing_submit(self):
        r = check("objs = list_objects()\nfinal_result = len(objs.value)")
        assert r.parse_ok and r.has_submit is False and not r.ok
        assert "submit()" in r.feedback()

    def test_all_four_reported_together(self):
        r = check("res = list_objects(labels='x')\ny = mystery()")
        assert r.has_submit is False and r.bad_kwargs and r.unknown_calls
        assert len(r.errors) >= 3


class TestStricterRules:
    def test_positional_tool_args_are_refused(self):
        """工具第一个参数是 scene_id —— 位置传参会静默绑错，运行期不会报错。"""
        r = check("res = list_objects('chair')\nsubmit(1, evidence=['x'])")
        assert not r.ok and r.positional_tool_calls == ("list_objects",)
        assert "位置参数" in r.feedback()

    def test_kwargs_expansion_is_refused(self):
        r = check("kw = {'label': 'chair'}\nres = list_objects(**kw)\nsubmit(1, evidence=['x'])")
        assert not r.ok
        assert "参数名无法静态核对" in r.feedback()

    def test_import_outside_allowlist(self):
        r = check("import os\nsubmit(1, evidence=['x'])")
        assert not r.ok and "os" in r.bad_imports

    def test_relative_import_refused(self):
        r = check("from . import foo\nsubmit(1, evidence=['x'])")
        assert not r.ok

    def test_allowed_import_passes(self):
        r = check("import math\nsubmit(math.pi, evidence=['pi'])")
        assert r.ok, r.errors


class TestNoFalsePositives:
    """误报同样致命：好程序被拒 → 模型开始改一个没坏的地方。"""

    def test_user_defined_function_is_not_unknown(self):
        src = ("def extent_diag(w, h, l):\n"
               "    return math.sqrt(w * w + h * h + l * l)\n"
               "ext = get_extent_helper()\n")  # 故意留一个未知调用，下面单独断言
        r = check(src)
        assert "extent_diag" not in r.unknown_calls
        assert "extent_diag" not in r.unknown_calls
        assert "get_extent_helper" in r.unknown_calls

    def test_local_variable_call_and_comprehension_names_are_fine(self):
        src = (
            "objs = list_objects()\n"
            "ids = [o['object_id'] for o in objs.value]\n"
            "pairs = [ids[i] for i in range(len(ids))]\n"
            "for idx, val in enumerate(pairs):\n"
            "    pass\n"
            "submit(len(pairs), evidence=['pairs'])\n"
        )
        r = check(src)
        assert r.ok, r.errors

    def test_method_calls_on_results_are_fine(self):
        src = ("res = list_objects()\n"
               "n = res.value[0].get('score')\n"
               "submit(n, evidence=['score'])\n")
        r = check(src)
        assert r.ok, r.errors

    def test_attribute_calls_are_fine(self):
        r = check("submit(math.sqrt(16), evidence=['math.sqrt'])")
        assert r.ok, r.errors

    def test_builtin_calls_are_fine(self):
        src = ("res = list_objects()\n"
               "items = res.value\n"
               "best = min(items, key=lambda o: o['score'])\n"
               "submit(sorted(o['label'] for o in items)[0], evidence=['sorted'])\n")
        r = check(src)
        assert r.ok, r.errors

    def test_lambda_params_do_not_shadow_check(self):
        r = check("f = lambda a, b: a\ntotal = f(1, 2)\nsubmit(total, evidence=['x'])")
        assert r.ok, r.errors

    def test_except_binding_name_is_known(self):
        src = ("try:\n"
               "    res = list_objects()\n"
               "except ValueError as err:\n"
               "    submit(str(err), evidence=['caught'])\n"
               "submit(1, evidence=['ok'])\n")
        r = check(src)
        assert r.ok, r.errors


class TestAgainstRealRegistry:
    def test_default_toolset_is_the_qa_toolset(self):
        r = static_check(GOOD)
        assert r.ok
        assert r.tool_calls == ("list_objects", "calculate_distance")

    def test_meta_tools_are_not_in_the_default_action_space(self):
        """L5 元工具不在动作空间里 —— 给了它，工具选择这个指标就没观测点了。"""
        r = static_check("r = describe_scene()\nsubmit(1, evidence=['x'])")
        assert not r.ok and "describe_scene" in r.unknown_calls

    def test_every_qa_tool_exists_and_has_params(self):
        for name in QA_TOOLSET:
            assert params_of(name), name

    def test_scene_id_is_a_real_parameter_name(self):
        r = static_check("r = list_objects(scene_id='x')\nsubmit(1, evidence=['x'])")
        assert r.ok, r.errors


class TestExtractCode:
    def test_python_fence(self):
        src, fenced = extract_code("说明文字\n```python\nsubmit(1, evidence=['x'])\n```\n尾巴")
        assert fenced and src == "submit(1, evidence=['x'])"

    def test_bare_fence(self):
        src, fenced = extract_code("```\nsubmit(1, evidence=['x'])\n```")
        assert fenced and "submit" in src

    def test_longest_block_wins(self):
        text = "```python\nx = 1\n```\n中间\n```python\nsubmit(x, evidence=['y'])\n```"
        src, fenced = extract_code(text)
        assert fenced and "submit" in src

    def test_no_fence_falls_back_to_whole_text(self):
        """模型没按提示词给代码块时**不当成失败**，但 `fenced=False` 会被记下来。"""
        src, fenced = extract_code("x = 1\nsubmit(x, evidence=['y'])")
        assert fenced is False and src.startswith("x = 1")

    def test_empty_reply(self):
        src, fenced = extract_code("")
        assert src == "" and fenced is False
        assert static_check(src, tools=FAKE_TOOLS, params=FAKE_PARAMS).ok is False


class TestFeedbackIsActionable:
    def test_error_messages_carry_line_numbers(self):
        r = check("objs = list_objects(labels='x')")
        joined = "\n".join(r.errors)
        assert "第 1 行" in joined

    def test_feedback_mentions_every_problem(self):
        r = check("import os\nx = mystery()\ny = list_objects(bogus=1)")
        text = r.feedback()
        for needle in ("os", "mystery", "bogus", "submit"):
            assert needle in text, needle


# ---------------------------------------------------------------------------
# 提示词契约：改提示词等于改实验条件，所以把「必须出现的东西」也钉住
# ---------------------------------------------------------------------------


class TestPromptContract:
    def _docs(self):
        from llm.schema import docs_text

        return docs_text(tools=FAKE_TOOLS)

    def test_placeholder_leftover_is_loud(self):
        """漏填占位符时模型会收到字面量 `{{X}}` 并自己猜 —— 必须当场炸。"""
        from agents.prompts.system import render_template

        with pytest.raises(KeyError, match="占位符"):
            render_template("前置 {{A}} 后置 {{B}}", {"A": "1"})

    def test_system_prompt_has_no_placeholders_left(self):
        """★ 不许出现的是**未替换的占位符**，不是「任何花括号」。

        这里原先断言 `"{{" not in text and "}}" not in text` —— 那是个**代理指标**，
        而且它拦下的东西与真正要守的无关：工具文档一写返回值形状就必然带花括号
        （`extent_m:{w,h,l}}`），于是「把形状写进工具文档」这个修法会被它误杀
        （2026-09-19 实测：12 条工具文档补形状后，这条断言红，而提示词本身完全正常）。

        真正的不变量由渲染器自己守：`_PLACEHOLDER_RE` 只认 `{{大写字母}}`，
        `render_template` 发现残留就抛 KeyError（见上一条用例）。
        所以这里改用**同一个正则**检查 —— 口径只有一处实现，且与渲染器一致；
        另外保留对两个已知键的显式检查，免得正则哪天被改宽了没人发现。
        """
        from agents.prompts.system import _PLACEHOLDER_RE, build_system_prompt

        text = build_system_prompt(self._docs(), ALLOWED_MODULES)
        assert _PLACEHOLDER_RE.findall(text) == []
        assert "{{TOOL_DOCS}}" not in text and "{{MODULES}}" not in text

    def test_system_prompt_names_the_object_id_field(self):
        """★ 这条是**实跑发现**的：模型按 `id` 取字段 → 报错 → 弃答。

        实测记录见 `logs/agent_runs/20260918_172113_living_room.json`（第 2 题）：
        模型用 `single_object` 拿到 dict，却按 `id` 取字段，于是 `a=None` 一路传到
        `calculate_distance`，最后如实弃答。提示词里必须点明字段名。

        ⚠ **2026-09-22 更正**：原文把两种写错混成了一句「写 `obj["id"]` / `obj.get("id")`
        会得到 `None`」。真跑里的报错其实是 **`KeyError: 'id'`**，`None` 只有 `.get()` 才有。
        不改判定，但错描述会把模型引向错误的调试方向。下面
        `test_keyerror_claim_matches_measured_behaviour` 专门盯住旧串归零。
        """
        from agents.prompts.system import build_system_prompt

        text = build_system_prompt(self._docs(), ALLOWED_MODULES)
        assert "object_id" in text
        assert "不是 `id`" in text

    def test_keyerror_claim_matches_measured_behaviour(self):
        """提示词关于 `obj["id"]` 的说法必须与运行期**实测**一致。

        「文档写了、实现没做」是本项目反复吃的形态；这条是它的**镜像**：
        **文档写错了，实现没问题**。所以断言必须成对出现 —— 一边测量，一边查文本。
        """
        import agents.prompts.system as ps

        obj = {"object_id": "chair_1"}
        with pytest.raises(KeyError):
            obj["id"]                                   # noqa: B018 —— 故意触发
        assert obj.get("id") is None                    # 只有 .get() 才是 None

        text = ps.build_system_prompt(self._docs(), ALLOWED_MODULES)
        assert "KeyError" in text, "提示词必须写明下标访问抛的是 KeyError"
        flat = " ".join(text.split())
        assert '`obj.get("id")` 会得到' not in flat, "旧串「会得到 None」必须归零"

    def test_system_prompt_says_tools_need_no_import(self):
        """★ 真跑 12 题里有 **2 题**因为写 `from tools import ...` 多跑了一轮。

        根因不是模型乱来：旧规则 5 的措辞是「只能 `import`：math, ...」，
        它在**暗示「功能要靠 import 拿到」**。所以修法是改提示词，不是怪模型 ——
        这一条与「写了→看到了→没照做」不同型，它属于**提示词诱发**。
        """
        import agents.prompts.system as ps

        text = ps.build_system_prompt(self._docs(), ALLOWED_MODULES)
        assert "不要 import" in text
        assert "已经在你的命名空间里" in text

    def test_importing_tools_really_fails_with_a_readable_message(self):
        """提示词说「`from tools import ...` 一定会失败」—— **这句话本身要被验证**。

        只断言提示词里有这句话，就是又一次「文档写了」。所以这里真跑一遍：
        失败要真的发生，且报错要**可读**（点名白名单），否则模型看不懂该怎么改。
        """
        from agents.executor import QA_TOOLSET, execute_program
        from scene_graph.schema import SceneGraph
        from tools.registry import ToolContext

        scene = SceneGraph(scene_id="s", image_id="i", nodes=())
        out = execute_program(
            "from tools import list_objects\nsubmit(1, evidence=['x'])\n",
            ToolContext(scene=scene, record_trace=True), toolset=QA_TOOLSET,
        )
        assert out.ok is False and out.stage == "runtime"
        assert "白名单" in out.message, out.message

    def test_prompt_fingerprint_covers_the_templates(self):
        """版本号会忘升，**指纹不会** —— 所以指纹必须真的覆盖模板内容。

        只断言「长度 16、全是十六进制」不够：把函数实现成返回常量也能过。
        这里改一个字符再要求它变，才证明它盯着模板。
        """
        import agents.prompts.system as ps

        before = ps.prompt_fingerprint()
        assert len(before) == 16
        assert all(c in "0123456789abcdef" for c in before)
        assert ps.prompt_fingerprint() == before            # 确定性

        original = ps.SYSTEM_TEMPLATE
        try:
            ps.SYSTEM_TEMPLATE = original + "\n第七行：多余的改动"
            assert ps.prompt_fingerprint() != before, "指纹没有覆盖 SYSTEM_TEMPLATE"
        finally:
            ps.SYSTEM_TEMPLATE = original
        assert ps.prompt_fingerprint() == before            # 可复原，不污染后续用例

    def test_prompt_version_is_a_semver_string(self):
        """版本号要能被机械核对 —— 只靠人眼看「1.2.0 之后是 1.1.0 吗」一定会出错。"""
        from agents.prompts.system import PROMPT_VERSION

        parts = PROMPT_VERSION.split(".")
        assert len(parts) == 3 and all(p.isdigit() for p in parts), PROMPT_VERSION

    def test_system_prompt_requires_evidence_and_keyword_args(self):
        from agents.prompts.system import build_system_prompt

        text = build_system_prompt(self._docs(), ALLOWED_MODULES)
        for needle in ("submit(", "evidence", "关键字", "至少 1 条"):
            assert needle in text, needle

    def test_user_prompt_states_the_answer_type(self):
        from agents.prompts.system import build_user_prompt

        assert "float" in build_user_prompt("问", {}, answer_type="float")
        assert "bool" in build_user_prompt("问", {}, answer_type="bool")
        assert "str" in build_user_prompt("问", {})

    def test_scene_hint_carries_no_coordinates(self):
        """§13.3(2) 的信息约束：坐标只能来自工具返回值。"""
        from agents.prompts.system import scene_hint_for
        from scene_graph.schema import Node, SceneGraph

        scene = SceneGraph(
            scene_id="s", image_id="i",
            nodes=(Node(id="chair_1", label="chair", centroid_3d=(1.5, 2.5, 3.5)),),
        )
        hint = scene_hint_for(scene)
        assert hint["objects"] == {"chair": 1}
        blob = repr(hint)
        for leaked in ("1.5", "2.5", "3.5", "centroid"):
            assert leaked not in blob, leaked

    def test_scene_hint_does_not_leak_intrinsics(self):
        """内参决定横向米制尺度（§21）—— 把它写进提示词会让不同题目拿到不同的暗示。"""
        from agents.prompts.system import scene_hint_for
        from scene_graph.schema import SceneGraph

        scene = SceneGraph(scene_id="s", image_id="i",
                           camera_intrinsics=[[606.0, 0.0, 320.0], [0.0, 606.0, 240.0], [0.0, 0.0, 1.0]])
        assert "camera_intrinsics" not in scene_hint_for(scene)

    def test_scene_hint_is_blind_to_build_meta(self):
        """★ 回归守卫：`build_meta` 的**任何**字段都不许进提示词。

        ⚠ 这条守卫的上一版是**失效**的 —— 它叫 `..._does_not_leak_intrinsics`、
        注释也写对了，但只断言 `"camera_intrinsics" not in hint`，而夹具**没有构造
        `build_meta`** ⟹ 真正有缺陷的那条分支（`build_meta.intrinsics_source` 透传）
        **从未被执行过**。这是「守卫没生效」而不是「守卫失败」：
        后者会变红，前者只是安静地什么都不查，于是一个错误能瞒过两轮。

        所以断言写的是**白名单**（hint 的键只能来自 `SCENE_HINT_KEYS`），
        不是「某个具体字段缺席」—— 下一个想透传 `fov` / `scale_calibrated` /
        `up_axis_reliable` 的人也会被这条拦下。

        ★ **已做 revert-check（把漏洞临时塞回）**：本测试与 `..._byte_identical_...`
        双双变红，而旧的 `..._does_not_leak_intrinsics` **依然绿** ——
        这就是"守卫失效"的直接证据，不是推断。
        """
        from agents.prompts.system import SCENE_HINT_KEYS, scene_hint_for
        from scene_graph.schema import SceneGraph

        # 键名取真实 builder 的输出（26 个），值用哨兵串 —— 便于连"值泄露"一起断言。
        meta = {
            "config": {"prompt": "sofa. chair.", "box_threshold": 0.3},
            "timings_ms": {"depth_ms": 471.6, "detect_ms": 927.1},
            "image_hw": [480, 640],                      # ← 真实落盘是 (高, 宽)
            "grid_hw": [480, 640],
            "grid_scale_vs_image": [1.0, 1.0],
            "depth_range_m": [1.3763, 3.9738],
            "n_detections_raw": 9, "n_detections_kept": 9,
            "n_nodes": 9, "n_edges": 133, "n_dropped": 0, "n_fallbacks": 0,
            "dropped": [], "fallbacks": [],
            "mask_box_coverage_mean": 0.8041, "mask_box_coverage_min": 0.4624,
            "label_counts": {"chair": 1},
            "up_axis": "-y", "up_axis_tilt_deg": 63.04,
            "up_axis_reliable": False, "up_axis_reason": "band_not_horizontal",
            "scale_calibrated": False,
            "intrinsics_source": "SENTINEL_source",       # ← 事故就是这一个字段
            "intrinsics": [[163.737762, 0.0, 322.0],
                           [0.0, 163.418274, 248.110352],
                           [0.0, 0.0, 1.0]],
            "fov": {"hfov_deg": 125.8, "vfov_deg": 111.5,
                    "plausible": False, "reason": "hfov_out_of_range"},
            "perception": {"device": "cuda", "device_total_mib": 8188.0},
        }
        hint = scene_hint_for(SceneGraph(scene_id="s", image_id="i", build_meta=meta))

        # ① 契约：键只能来自白名单
        extra = set(hint) - SCENE_HINT_KEYS
        assert not extra, "scene_hint 出现白名单外的键：%s" % sorted(extra)
        # ② 键名不许出现
        import json as _json

        blob = repr(hint) + _json.dumps(hint, ensure_ascii=False)
        for k in meta:
            assert k not in blob, "build_meta 的键泄露进 scene_hint：%s" % k
        # ③ 值也不许出现（值泄露比键泄露更隐蔽：键名可以不出现，值照样能带着信息）
        assert "SENTINEL" not in blob
        # ④ 两个"看起来像尺寸"的量必须分开：image_size 来自归一化，不是 build_meta 透传
        assert hint["image_size"] == [640, 480]           # (宽, 高) —— 不是 image_hw 的 (480, 640)

    def test_scene_hint_is_byte_identical_across_intrinsics_sources(self):
        """★ 可比性契约：只改 `build_meta`（内参来源那一档）必须给出**逐字节相同**的 hint。

        这是探针实验的前提。它一旦不成立，「只改内参」的对照就同时改了提示词，
        结论不可归因 —— 而那种污染**不会报错**，只会让两档的差异看起来更大或更小。
        """
        from agents.prompts.system import scene_hint_for
        from scene_graph.schema import Node, SceneGraph

        import json as _json

        shared = {"image_hw": [480, 640],
                  "intrinsics": [[163.737762, 0.0, 322.0],
                                 [0.0, 163.418274, 248.110352],
                                 [0.0, 0.0, 1.0]]}
        nodes = (Node(id="chair_1", label="chair", centroid_3d=(1.5, 2.5, 3.5)),)

        def hint_with(source: str) -> str:
            sc = SceneGraph(scene_id="probe", image_id="i", nodes=nodes,
                            build_meta=dict(shared, intrinsics_source=source))
            return _json.dumps(scene_hint_for(sc), ensure_ascii=False, sort_keys=True)

        assert hint_with("predicted") == hint_with("provided")
