"""`scripts/compare_intrinsics_probe.py` 的单元测试 —— 钉住**尺子**，不钉数字。

为什么值得给一把"比对脚本"写单测
--------------------------------
这个项目的判定逻辑已经错过一次：第一版比对只认中文，而模型答的是
`left` / `back` / `behind`，于是**把 3 道全对的题判成错**，还顺手标了「参照档也错」
（等于把锅推给模型）。**同一份数据、只换尺子，结论就变，而且不会报错。**

所以这里测的不是"脚本跑得通"，是"尺子在**已知**输入上给出**已知**答案"：
容差边界、中英文别名、以及**弃答必须返回 `None` 而不是 `False`**
（`False` 是"答错"，`None` 是"没答" —— 混起来就是那次
「把 30% 正确答案判 unsupported」的同型缺陷）。

数字本身（abs 0/8、rel 8/8、cnt 4/4）**不进单测** —— 它是实测结果，
会随模型/场景变；进单测只会变成"改数据就得改测试"。这里只钉判定规则与契约。

导入方式：`scripts/` 不是包（没有 `__init__.py`），所以按路径加载，不动包结构。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module():
    path = ROOT / "scripts" / "compare_intrinsics_probe.py"
    spec = importlib.util.spec_from_file_location("compare_intrinsics_probe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cmp = _load_module()

REAL_QUESTIONS = ROOT / "dataset" / "probes" / "intrinsics_sensitivity_20q.json"


def _questions() -> list[dict]:
    return json.loads(REAL_QUESTIONS.read_text(encoding="utf-8"))["questions"]


def _fake_scene(**overrides) -> dict:
    """够 `build_truth` 跑起来的最小场景（只放它必需的四个 id + 两个陪跑）。"""
    base = {
        "table_1": {"id": "table_1", "label": "table", "centroid_3d": [0.0, 0.0, 2.0]},
        "chair_1": {"id": "chair_1", "label": "chair", "centroid_3d": [0.5, 0.0, 2.0]},
        "mirror_1": {"id": "mirror_1", "label": "mirror", "centroid_3d": [-1.0, 0.0, 4.0]},
        "sofa_1": {"id": "sofa_1", "label": "sofa", "centroid_3d": [0.0, 0.0, 1.0],
                   "extent_3d": [2.0, 0.8, 0.9]},
        "picture_1": {"id": "picture_1", "label": "picture", "centroid_3d": [0.0, 0.0, 9.0]},
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 题集契约：顺序即契约，靠硬校验
# ---------------------------------------------------------------------------


class TestOrderContract:
    def test_real_question_file_matches_the_truth_builder(self):
        """真实题集与 `build_truth` 必须逐题对齐 —— 这是"顺序即契约"的执行体。"""
        cmp._assert_order_contract(cmp.build_truth(_fake_scene()), _questions())   # 不抛即通过

    def test_a_swapped_question_is_caught(self):
        qs = _questions()
        qs[0], qs[1] = qs[1], qs[0]
        with pytest.raises(ValueError, match="顺序即契约"):
            cmp._assert_order_contract(cmp.build_truth(_fake_scene()), qs)

    def test_a_wrong_group_is_caught(self):
        qs = _questions()
        qs[0] = {**qs[0], "category": "rel"}
        with pytest.raises(ValueError, match="分组"):
            cmp._assert_order_contract(cmp.build_truth(_fake_scene()), qs)

    def test_a_short_question_file_is_caught(self):
        with pytest.raises(ValueError, match="题数"):
            cmp._assert_order_contract(cmp.build_truth(_fake_scene()), _questions()[:5])

    def test_missing_required_id_fails_loudly(self):
        """缺 id 说明场景换了、题目语义已不成立 —— 宁可报错，不要出一张看似正常的表。"""
        scene = _fake_scene()
        del scene["table_1"]
        with pytest.raises(KeyError, match="table_1"):
            cmp.build_truth(scene)


# ---------------------------------------------------------------------------
# 尺子：数值
# ---------------------------------------------------------------------------


class TestNumericRuler:
    ITEM = {"kind": "num", "truth": 2.0, "cat": "abs"}

    def test_tolerance_is_five_percent(self):
        assert cmp.judge(self.ITEM, 2.09)["ok"] is True          # 4.5% ⟹ 过
        assert cmp.judge(self.ITEM, 2.11)["ok"] is False         # 5.5% ⟹ 不过

    def test_the_exact_boundary_is_float_fuzzy_and_that_is_accepted(self):
        """**已知性质，不是待修的 bug**：恰好落在 ±5% 上的结果由二进制浮点表示决定
        —— `2.1 - 2.0 == 0.10000000000000009 > 0.05 * 2.0` ⟹ 判不过。

        刻意**不**给尺子加 epsilon：报告只印到 0.1%，边界那一位不承载信息，
        而一个"看起来更友好"的魔数会让尺子多一处无法从数据推出的自由度
        （本项目对这类"大概是这样"的换算因子有明确纪律：要么有据可依，要么
        **让误差落在安全的一侧并写明**）。

        本题集上它不影响任何结论：8 道绝对题里离 5% 最近的是 12.7%，有 2.5 倍余量。
        """
        assert cmp.judge(self.ITEM, 2.1)["ok"] is False          # 浮点上恰好越界
        assert cmp.judge(self.ITEM, 1.9)["ok"] is False          # 另一侧同理（差值同样略大于 0.1）

    def test_units_and_chinese_are_stripped(self):
        """实测模型回的是 `'3.07 m'` 这种带单位的串 —— 尺子必须能读它。"""
        for raw in ("3.07 m", "3.07米", "约 3.07", 3.07):
            assert cmp.as_number(raw) == pytest.approx(3.07), raw

    def test_zero_truth_does_not_divide_by_zero(self):
        hit = cmp.judge({"kind": "num", "truth": 0.0, "cat": "cnt"}, 0)
        assert hit["ok"] is True and hit["rel_err"] == 0.0
        assert cmp.judge({"kind": "num", "truth": 0.0, "cat": "cnt"}, 1)["ok"] is False


# ---------------------------------------------------------------------------
# 尺子：二元题（这里就是曾经出错的地方）
# ---------------------------------------------------------------------------


class TestBinaryRulers:
    def test_side_accepts_the_english_the_model_actually_answers(self):
        """★ 回归点：模型答 `left`/`back`/`behind`。第一版尺子只认中文 ⟹ 3 道全对判错。"""
        back = {"kind": "side", "truth": "后", "cat": "rel"}
        assert cmp.judge(back, "behind")["ok"] is True          # 参照档实测答案
        assert cmp.judge(back, "back")["ok"] is True            # pred 档实测答案
        assert cmp.judge(back, "front")["ok"] is False
        left = {"kind": "side", "truth": "左", "cat": "rel"}
        assert cmp.judge(left, "left")["ok"] is True
        assert cmp.judge(left, "right")["ok"] is False

    def test_obj_matches_by_canonical_label_and_alias(self):
        pic = {"kind": "obj", "truth": "picture", "cat": "rel"}
        assert cmp.judge(pic, "picture")["ok"] is True
        assert cmp.judge(pic, "画")["ok"] is True
        assert cmp.judge(pic, "chair")["ok"] is False

    def test_obj_matcher_is_deliberately_generous_on_the_glued_label(self):
        """**已知偏宽**（docstring 里已写明，这里把它钉成"故意"而不是"意外"）：

        真值是 `sofa chair`（两个检测粘连出的一个标签）时，答案只答 `chair` 也算对。
        偏宽的方向是**有利于模型**，所以它不会把"模型不会做题"说过头 ——
        但反过来，它也不会让分数虚高到掩盖真问题（那批题的真值标签都是单词）。
        """
        glued = {"kind": "obj", "truth": "sofa chair", "cat": "rel"}
        assert cmp.judge(glued, "chair")["ok"] is True
        assert cmp.judge(glued, "sofa")["ok"] is True
        assert cmp.judge(glued, "table")["ok"] is False

    def test_set_accepts_the_comma_separated_string(self):
        """★ 回归点：模型回的是逗号分隔字符串，不是 list。第一版只认 list 就漏判了。"""
        item = {"kind": "set", "truth": ["chair", "mirror", "table"], "cat": "cnt"}
        assert cmp.judge(item, "chair, mirror, table")["ok"] is True
        assert cmp.judge(item, "chair、mirror、table")["ok"] is True      # 顿号也认
        assert cmp.judge(item, ["chair", "mirror", "table"])["ok"] is True
        assert cmp.judge(item, ["chair", "mirror"])["ok"] is False

    def test_yesno(self):
        item = {"kind": "yesno", "truth": "no", "cat": "cnt"}
        assert cmp.judge(item, "no")["ok"] is True
        assert cmp.judge(item, "没有")["ok"] is True
        assert cmp.judge(item, "yes")["ok"] is False


# ---------------------------------------------------------------------------
# 弃答 ≠ 答错（这是本项目反复踩的那类坑）
# ---------------------------------------------------------------------------


class TestAbstentionIsNotWrong:
    def test_unparseable_numbers_return_none_not_false(self):
        """`None` = 没答 / 尺子覆盖不到；`False` = 答错。混起来就是把"未测到"当地 0 分。"""
        num = {"kind": "num", "truth": 2.0, "cat": "abs"}
        for junk in ("unknown", "", None, True, [1, 2], "不知道"):
            assert cmp.judge(num, junk)["ok"] is None, junk

    def test_unknown_kind_is_not_counted_as_wrong(self):
        assert cmp.judge({"kind": "wat", "truth": 1, "cat": "abs"}, 1)["ok"] is None

    def test_non_string_answer_to_a_side_question_is_not_counted_as_wrong(self):
        assert cmp.judge({"kind": "side", "truth": "左", "cat": "rel"}, 3)["ok"] is None


# ---------------------------------------------------------------------------
# 可比性：两档提示词必须逐字节相同（用盘上的真实资产验）
# ---------------------------------------------------------------------------


class TestHintAlignment:
    def test_probe_scenes_have_byte_identical_prompts(self):
        """★ 探针实验的前提。这条一破，两档的差异就无法归因到几何。"""
        ok, info = cmp.check_hint_alignment(cmp.DEFAULT_PRED_SCENE, cmp.DEFAULT_GT_SCENE)
        assert ok is True, info

    def test_the_unprepared_living_room_pair_drifts_only_on_scene_id(self):
        """未处理过的 `living_room_pred` / `_gt` 对**不是**干净对照 —— 差异只在 `scene_id`。

        写这条是为了把 `scene_hint` 的现状钉住：`scene_id` 确实进了提示词，
        所以"直接拿这两个场景跑对照"会带着一个非几何差异。
        （若将来 `scene_id` 从 hint 里去掉，差异集会变空 —— 这条断言仍然成立。）
        """
        ok, info = cmp.check_hint_alignment("living_room_pred", "living_room_gt")
        keys_a = json.loads(info.split("\n")[0].split("=", 1)[1].strip())
        keys_b = json.loads(info.split("\n")[1].split("=", 1)[1].strip())
        assert set(keys_a) == set(keys_b)                       # 键集合一致
        diff = {k for k in keys_a if keys_a[k] != keys_b[k]}
        assert diff <= {"scene_id"}, diff
        assert not ok if diff else True                         # 有差异就必须报未对齐
