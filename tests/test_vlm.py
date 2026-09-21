"""`llm/vlm.py`（角色② 视觉语义）的单测 —— 零联网、零 GPU。

这里最值钱的不是「解析 JSON 对不对」，而是三条**结构性断言**：

    1. `describe()` 的签名里**不能出现任何空间词汇** —— 这是 §13.3(3) 那句
       「空间幻觉在类型层面就写不出来」的可执行版本。测它而不是测提示词，
       是因为提示词可以被绕过，签名不行。
    2. 发给视觉模型的提示词里**一个数字都不能有** —— 我们刚把 bbox 裁掉再送出去，
       如果又把它写进提示词，等于白裁，而且角色② 就多了一条看到坐标的路径。
    3. **闭集违规与缺置信度必须被记录**，不能悄悄取个近似值 ——
       「闭集」的价值全在于它让「模型瞎编」变成一个可计数的数字。
"""

from __future__ import annotations

import inspect
import json
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.adapter import LLMClient, LLMError, LLMSettings  # noqa: E402
from llm.vlm import (  # noqa: E402
    ATTRS,
    DEFAULT_CONFIDENCE_THRESHOLD,
    SPATIAL_PARAM_TERMS,
    Attribute,
    Region,
    VLM,
    VLMError,
    check_signature,
    image_size_from_meta,
    to_data_url,
)


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------

#: 单元测试的调用日志落在**系统临时目录**，不落仓库。
#: ⚠ 2026-09-21 修复：这里原先默认写 `ROOT/logs/_unit_vlm_calls.jsonl`，
#: 而那个路径是**被 git 跟踪的** ⟹ 每跑一次测试就往仓库里追加 60 行
#: （入库时已积累 417 行 / 161 KB），于是 `git status` 永远不干净。
#: 后果不只是脏：**一个每次都脏的工作树会训练人忽略 `git status`**，
#: 而「改动有没有生效」正是靠它判断的（本项目的排查常态）。
_UNIT_LOG_DIR = Path(tempfile.mkdtemp(prefix="spatial-unit-vlm-"))


def fake_client(text: str, *, finish_reason: str = "stop", model: str = "fake-vision",
                log_path: Path | None = None):
    """造一个**零联网**的 `LLMClient`，并把请求体抓出来供断言。"""
    seen: dict = {}

    def transport(url, payload, headers, timeout):
        seen["url"] = url
        seen["payload"] = payload
        seen["headers"] = dict(headers)
        return {
            "model": model,
            "choices": [{"message": {"content": text}, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30},
        }

    st = LLMSettings(base_url="https://vision.invalid/v1", api_key="sk-unit-test",
                     model=model, label="vision")
    client = LLMClient(st, transport=transport,
                       log_path=str(log_path or (_UNIT_LOG_DIR / "vlm_calls.jsonl")))
    return client, seen


def image_file(tmp_path: Path, size=(200, 120), color=(200, 30, 30)) -> Path:
    pil = pytest.importorskip("PIL.Image")
    p = tmp_path / "frame.png"
    pil.new("RGB", size, color).save(p)
    return p


def region(**kw) -> Region:
    kw.setdefault("label", "chair")
    return Region(**kw)


def attrs_payload(*, name="color", value="black", confidence=0.9) -> str:
    return json.dumps({"attributes": [{"name": name, "value": value,
                                       "confidence": confidence}]}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# ★ 结构性断言：签名里没有空间参数
# ---------------------------------------------------------------------------


class TestSignatureHasNoSpatialParameter:
    def test_contract_parameters_are_exactly_the_four(self):
        params = list(inspect.signature(VLM.describe).parameters)
        assert params == ["self", "image", "region", "attrs", "candidates"]

    def test_check_signature_is_clean(self):
        assert check_signature() == ()

    def test_check_signature_detects_a_violation(self):
        """故意造一个带空间参数的函数 —— 检查器必须抓得住。

        没有这条用例的话，`check_signature` 可能永远返回 ()，
        而「架构保证」就退化成一个永远为真的断言（最坏的一种测试）。
        """
        def describe(image, region, attrs, candidates=None, left_of=None):
            return []

        bad = check_signature(describe)
        assert bad == ("left_of",)

    def test_checker_covers_every_declared_term(self):
        for term in SPATIAL_PARAM_TERMS:
            def fn(**kw):
                return []

            fn.__signature__ = inspect.Signature(
                [inspect.Parameter(term + "_x", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
            )
            assert check_signature(fn) == (term + "_x",), term

    def test_import_time_guard_is_actually_installed(self):
        """模块底部的导入期断言必须存在 —— 否则「违规代码根本跑不起来」这句话是空的。"""
        src = (ROOT / "llm" / "vlm.py").read_text(encoding="utf-8")
        assert "_violations = check_signature()" in src
        assert "raise RuntimeError" in src

    def test_attrs_are_the_five_semantic_ones(self):
        assert ATTRS == ("color", "material", "texture", "state", "shape")

    def test_attrs_contain_no_spatial_word(self):
        for a in ATTRS:
            for term in SPATIAL_PARAM_TERMS:
                assert term not in a, (a, term)


# ---------------------------------------------------------------------------
# 图像 → data URL
# ---------------------------------------------------------------------------


class TestImageEncoding:
    def test_path_becomes_a_data_url(self, tmp_path):
        url = to_data_url(image_file(tmp_path))
        assert url.startswith("data:image/png;base64,")

    def test_data_url_passes_through_unchanged(self):
        raw = "data:image/jpeg;base64,AAAA"
        assert to_data_url(raw) == raw

    def test_bytes_are_encoded(self):
        assert to_data_url(b"\x89PNG\r\n") .startswith("data:image/png;base64,")

    def test_mapping_with_path_is_accepted(self, tmp_path):
        p = image_file(tmp_path)
        assert to_data_url({"path": str(p)}).startswith("data:image/png;base64,")

    def test_missing_file_raises_vlm_error(self):
        with pytest.raises(VLMError, match="不存在"):
            to_data_url("/definitely/not/here.png")

    def test_none_raises(self):
        with pytest.raises(VLMError, match="没有图像"):
            to_data_url(None)


# ---------------------------------------------------------------------------
# describe：正常路径
# ---------------------------------------------------------------------------


class TestDescribeHappyPath:
    def test_parses_a_fenced_json_reply(self, tmp_path):
        c, seen = fake_client("```json\n" + attrs_payload() + "\n```")
        out = VLM(c).describe(image_file(tmp_path), region(), ["color"])
        assert len(out) == 1 and out[0] == Attribute(
            name="color", value="black", confidence=0.9, raw="black")

    def test_parses_a_bare_json_reply(self, tmp_path):
        c, _ = fake_client(attrs_payload())
        assert VLM(c).describe(image_file(tmp_path), region(), ["color"])[0].value == "black"

    def test_output_order_follows_attrs_not_the_model(self, tmp_path):
        payload = json.dumps({"attributes": [
            {"name": "material", "value": "leather", "confidence": 0.8},
            {"name": "color", "value": "black", "confidence": 0.9},
        ]})
        c, _ = fake_client(payload)
        out = VLM(c).describe(image_file(tmp_path), region(), ["color", "material"])
        assert [a.name for a in out] == ["color", "material"]

    def test_low_confidence_is_returned_not_swallowed(self, tmp_path):
        """低置信度是**结果**，不是异常 —— 值照给，但必须进 `low_confidence` 清单。"""
        c, _ = fake_client(attrs_payload(confidence=0.2))
        v = VLM(c)
        out = v.describe(image_file(tmp_path), region(), ["color"])
        assert out[0].value == "black" and out[0].confidence == 0.2
        assert v.last_call.low_confidence == ("color",)

    def test_threshold_is_respected_by_the_low_confidence_list(self, tmp_path):
        """把阈值调低之后，同一个置信度就不再算「低」—— 两边必须用同一个阈值。"""
        c, _ = fake_client(attrs_payload(confidence=0.2))
        v = VLM(c, threshold=0.1)
        v.describe(image_file(tmp_path), region(), ["color"])
        assert v.last_call.low_confidence == ()
        assert v.last_call.threshold == 0.1

    def test_last_call_report_records_model_and_usage(self, tmp_path):
        c, _ = fake_client(attrs_payload())
        v = VLM(c)
        v.describe(image_file(tmp_path), region(), ["color"])
        rep = v.last_call.to_dict()
        assert rep["model"] == "fake-vision" and rep["endpoint_label"] == "vision"
        assert rep["usage"]["prompt_tokens"] == 120
        assert rep["requested"] == ["color"]

    def test_the_request_really_carries_an_image(self, tmp_path):
        c, seen = fake_client(attrs_payload())
        VLM(c).describe(image_file(tmp_path), region(), ["color"])
        parts = seen["payload"]["messages"][0]["content"]
        kinds = [p["type"] for p in parts]
        assert kinds == ["text", "image_url"]
        assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


# ---------------------------------------------------------------------------
# ★ 结构性断言：提示词里一个数字都不能有
# ---------------------------------------------------------------------------


class TestPromptLeaksNothing:
    def test_prompt_is_byte_identical_for_two_different_bboxes(self, tmp_path):
        """裁剪用的 bbox **不进提示词** —— 否则角色② 就多了一条看到坐标的路径。

        送出去的是「一张裁出来的小图 + 里面有个椅子」，不是「bbox=(50,20,100,60) 的物体」。
        直接断言"提示词里没有数字"是**错的**（提示词本身有序号 `1. 2. 3.`、
        `confidence` 的取值区间 `0~1`）；能证明这件事的正确断言是：
        **换一个 bbox，提示词逐字节不变**。
        """
        img = image_file(tmp_path, size=(200, 120))
        c1, seen1 = fake_client(attrs_payload())
        VLM(c1).describe(img, region(bbox=(10.0, 10.0, 40.0, 40.0), label="chair"), ["color"])
        c2, seen2 = fake_client(attrs_payload())
        VLM(c2).describe(img, region(bbox=(150.0, 80.0, 199.0, 119.0), label="chair"), ["color"])

        p1 = seen1["payload"]["messages"][0]["content"][0]["text"]
        p2 = seen2["payload"]["messages"][0]["content"][0]["text"]
        assert p1 == p2

    def test_no_bbox_number_appears_in_the_prompt(self, tmp_path):
        c, seen = fake_client(attrs_payload())
        VLM(c).describe(
            image_file(tmp_path, size=(200, 120)),
            region(bbox=(137.0, 91.0, 173.0, 119.0), label="chair"),
            ["color"],
        )
        prompt = seen["payload"]["messages"][0]["content"][0]["text"]
        for token in ("137", "91", "173", "119"):
            assert token not in prompt

    def test_prompt_names_the_label_and_the_attributes(self, tmp_path):
        c, seen = fake_client(attrs_payload())
        VLM(c).describe(image_file(tmp_path), region(label="sofa"), ["color", "material"])
        prompt = seen["payload"]["messages"][0]["content"][0]["text"]
        assert "sofa" in prompt and "color" in prompt and "material" in prompt

    def test_closed_set_is_spelled_out_for_the_model(self, tmp_path):
        c, seen = fake_client(attrs_payload())
        VLM(c).describe(image_file(tmp_path), region(), ["color"],
                        {"color": ["black", "white"]})
        prompt = seen["payload"]["messages"][0]["content"][0]["text"]
        assert "black" in prompt and "white" in prompt

    def test_prompt_never_asks_for_place_orientation_or_size(self, tmp_path):
        c, seen = fake_client(attrs_payload())
        VLM(c).describe(image_file(tmp_path), region(), ["color"])
        prompt = seen["payload"]["messages"][0]["content"][0]["text"]
        for word in ("位置", "朝向", "远近", "大小"):
            assert word in prompt          # 这些词只出现在「不要回答」那句禁令里
        assert "不要描述它的位置" in prompt


class TestCropping:
    def test_crop_happens_and_is_recorded(self, tmp_path):
        c, _ = fake_client(attrs_payload())
        v = VLM(c)
        v.describe(image_file(tmp_path, size=(200, 120)),
                   region(bbox=(60.0, 30.0, 120.0, 90.0)), ["color"])
        assert v.last_call.cropped is True

    def test_crop_works_when_the_image_is_a_Path_not_a_str(self, tmp_path):
        """★ 回归：调用方最自然的写法是传 `Path`，而 `isinstance(Path, str)` 是 False。

        曾经只认 `str` ⟹ `Path` 静默退化成整图（`cropped=False`），
        看起来一切正常，实际是"小物件属性全靠模型猜"。
        """
        c, _ = fake_client(attrs_payload())
        v = VLM(c)
        p = image_file(tmp_path, size=(200, 120))
        assert not isinstance(p, str)
        v.describe(p, region(bbox=(60.0, 30.0, 120.0, 90.0)), ["color"])
        assert v.last_call.cropped is True

    def test_no_bbox_means_no_crop(self, tmp_path):
        c, _ = fake_client(attrs_payload())
        v = VLM(c)
        v.describe(image_file(tmp_path), region(bbox=None), ["color"])
        assert v.last_call.cropped is False

    def test_bbox_larger_than_the_image_degrades_instead_of_raising(self, tmp_path):
        c, _ = fake_client(attrs_payload())
        v = VLM(c)
        out = v.describe(image_file(tmp_path, size=(40, 40)),
                         region(bbox=(0.0, 0.0, 900.0, 900.0)), ["color"])
        assert out[0].value == "black"      # 没崩，退化成整图

    def test_crop_sends_a_different_image_than_the_full_frame(self, tmp_path):
        """真的换了像素 —— 否则「裁了」只是报告里的一个 True。"""
        c1, seen1 = fake_client(attrs_payload())
        VLM(c1).describe(image_file(tmp_path), region(bbox=None), ["color"])
        c2, seen2 = fake_client(attrs_payload())
        VLM(c2).describe(image_file(tmp_path), region(bbox=(60.0, 30.0, 120.0, 90.0)), ["color"])
        assert (seen1["payload"]["messages"][0]["content"][1]["image_url"]["url"]
                != seen2["payload"]["messages"][0]["content"][1]["image_url"]["url"])


# ---------------------------------------------------------------------------
# ★ 闭集与置信度：必须被记录，不许静默兜回
# ---------------------------------------------------------------------------


class TestClosedSet:
    def test_value_outside_the_closed_set_is_flagged_and_zeroed(self, tmp_path):
        c, _ = fake_client(attrs_payload(value="charcoal", confidence=0.95))
        v = VLM(c)
        out = v.describe(image_file(tmp_path), region(), ["color"],
                         {"color": ["black", "white"]})
        assert out[0].value == "charcoal"           # 原话保留，供人工看
        assert out[0].in_closed_set is False
        assert out[0].confidence == 0.0             # ★ 不许当确定值
        assert v.last_call.violations == ("color",)
        assert "color" in v.last_call.low_confidence

    def test_case_difference_is_normalised_into_the_closed_set(self, tmp_path):
        c, _ = fake_client(attrs_payload(value="BLACK "))
        out = VLM(c).describe(image_file(tmp_path), region(), ["color"],
                              {"color": ["black", "white"]})
        assert out[0].value == "black" and out[0].in_closed_set is True

    def test_no_candidates_means_no_closed_set_check(self, tmp_path):
        c, _ = fake_client(attrs_payload(value="charcoal"))
        out = VLM(c).describe(image_file(tmp_path), region(), ["color"])
        assert out[0].in_closed_set is True and out[0].value == "charcoal"


class TestConfidenceIsNeverFaked:
    def test_missing_confidence_defaults_to_zero_not_one(self, tmp_path):
        payload = json.dumps({"attributes": [{"name": "color", "value": "black"}]})
        c, _ = fake_client(payload)
        v = VLM(c)
        out = v.describe(image_file(tmp_path), region(), ["color"])
        assert out[0].confidence == 0.0
        assert "color" in v.last_call.low_confidence

    def test_unparsable_confidence_defaults_to_zero(self, tmp_path):
        payload = json.dumps({"attributes": [
            {"name": "color", "value": "black", "confidence": "very sure"}]})
        c, _ = fake_client(payload)
        assert VLM(c).describe(image_file(tmp_path), region(), ["color"])[0].confidence == 0.0

    def test_confidence_is_clamped_to_unit_interval(self, tmp_path):
        c, _ = fake_client(attrs_payload(confidence=7.5))
        assert VLM(c).describe(image_file(tmp_path), region(), ["color"])[0].confidence == 1.0

    def test_missing_attribute_is_reported_not_invented(self, tmp_path):
        """模型漏答一条 → 空值 + 0 置信度 + 记进违规，**不补默认值**。"""
        c, _ = fake_client(attrs_payload())          # 只答了 color
        v = VLM(c)
        out = v.describe(image_file(tmp_path), region(), ["color", "material"])
        assert out[1].name == "material" and out[1].value == ""
        assert out[1].confidence == 0.0
        assert "material" in v.last_call.violations

    def test_truncated_output_marks_everything_uncertain(self, tmp_path):
        """截断的 JSON 会解析成"少了几条属性"，看起来像模型漏答 —— 必须区分开。"""
        c, _ = fake_client(attrs_payload(), finish_reason="length")
        v = VLM(c)
        v.describe(image_file(tmp_path), region(), ["color", "material"])
        assert v.last_call.low_confidence == ("color", "material")
        assert "FINISH_REASON_LENGTH" in v.last_call.error


# ---------------------------------------------------------------------------
# 参数与失败
# ---------------------------------------------------------------------------


class TestArgumentValidation:
    def test_unknown_attribute_raises(self, tmp_path):
        c, _ = fake_client(attrs_payload())
        with pytest.raises(VLMError, match="不支持的属性"):
            VLM(c).describe(image_file(tmp_path), region(), ["color", "brand"])

    def test_spatial_attribute_is_refused_with_the_reason(self, tmp_path):
        """`left_of` / `position` 这类请求必须被拒，而且要说清为什么。"""
        c, _ = fake_client(attrs_payload())
        with pytest.raises(VLMError, match="空间关系一律由几何层计算"):
            VLM(c).describe(image_file(tmp_path), region(), ["position"])

    def test_empty_attrs_raises(self, tmp_path):
        c, _ = fake_client(attrs_payload())
        with pytest.raises(VLMError, match="attrs 为空"):
            VLM(c).describe(image_file(tmp_path), region(), [])

    def test_attrs_are_case_insensitive(self, tmp_path):
        c, _ = fake_client(attrs_payload())
        assert VLM(c).describe(image_file(tmp_path), region(), ["COLOR"])[0].name == "color"


class TestBackendFailures:
    def test_llm_error_becomes_vlm_error(self, tmp_path):
        def transport(url, payload, headers, timeout):
            raise RuntimeError("boom")

        st = LLMSettings(base_url="https://vision.invalid/v1", api_key="sk-x",
                         model="fake-vision", max_retries=0, label="vision")
        v = VLM(LLMClient(st, transport=transport, log_path=str(tmp_path / "c.jsonl")))
        with pytest.raises(VLMError, match="视觉后端调用失败"):
            v.describe(image_file(tmp_path), region(), ["color"])
        assert v.last_call.error.startswith("BACKEND_UNAVAILABLE")

    def test_non_json_reply_raises_and_is_recorded(self, tmp_path):
        c, _ = fake_client("我觉得它是黑色的。")
        v = VLM(c)
        with pytest.raises(VLMError, match="不是 JSON"):
            v.describe(image_file(tmp_path), region(), ["color"])
        assert "不是 JSON" in v.last_call.error

    def test_empty_reply_raises(self, tmp_path):
        c, _ = fake_client("")
        with pytest.raises(VLMError, match="空内容"):
            VLM(c).describe(image_file(tmp_path), region(), ["color"])

    def test_bad_image_does_not_reach_the_backend(self, tmp_path):
        c, seen = fake_client(attrs_payload())
        v = VLM(c)
        with pytest.raises(VLMError, match="不存在"):
            v.describe(str(tmp_path / "nope.png"), region(), ["color"])
        assert "payload" not in seen          # 一次请求都没发出去


# ---------------------------------------------------------------------------
# 配置：不硬编码模型名
# ---------------------------------------------------------------------------


class TestSettingsComeFromEnv:
    def test_vision_endpoint_can_be_swapped_by_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SPATIAL_VISION_BASE_URL", "https://my-vlm.example/v1")
        monkeypatch.setenv("SPATIAL_VISION_MODEL", "qwen3-vl-flash")
        monkeypatch.setenv("SPATIAL_VISION_API_KEY", "sk-vision")
        monkeypatch.delenv("SPATIAL_API_KEY", raising=False)
        v = VLM()
        assert v.settings.model == "qwen3-vl-flash"
        assert v.settings.endpoint == "https://my-vlm.example/v1/chat/completions"

    def test_falls_back_to_the_text_endpoint_when_vision_is_unset(self, monkeypatch):
        for k in ("SPATIAL_VISION_MODEL", "SPATIAL_VISION_BASE_URL",
                  "SPATIAL_VISION_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("SPATIAL_MODEL", "deepseek-flash")
        monkeypatch.setenv("SPATIAL_API_KEY", "sk-text")
        v = VLM()
        assert v.settings.model == "deepseek-flash"
        assert v.settings.label == "text"

    def test_describe_dict_never_leaks_the_key(self, monkeypatch):
        monkeypatch.setenv("SPATIAL_VISION_MODEL", "qwen3-vl-flash")
        monkeypatch.setenv("SPATIAL_VISION_API_KEY", "sk-super-secret-value")
        d = VLM().describe_dict()
        assert "sk-super-secret-value" not in json.dumps(d, ensure_ascii=False)
        assert d["api_key"].startswith("***")

    def test_a_lone_vision_key_does_not_hijack_the_endpoint(self, monkeypatch):
        """只给了 `SPATIAL_VISION_API_KEY`、没给 model/base_url 时**不切换端点**。

        这是一条刻意的边界：单靠一个 key 无法判断端点是否另有一套，
        贸然切换会让"我明明填了 key"变成"它去连了一个不存在的服务"。
        端点由 `model` 或 `base_url` 触发。
        """
        for k in ("SPATIAL_VISION_MODEL", "SPATIAL_VISION_BASE_URL"):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("SPATIAL_VISION_API_KEY", "sk-vision-only")
        monkeypatch.setenv("SPATIAL_MODEL", "deepseek-flash")
        monkeypatch.setenv("SPATIAL_API_KEY", "sk-text")
        assert VLM().settings.label == "text"

    def test_threshold_is_configurable(self):
        assert VLM(fake_client("{}")[0], threshold=0.9).threshold == 0.9
        assert DEFAULT_CONFIDENCE_THRESHOLD == 0.6


class TestAttributeShape:
    def test_attribute_has_no_place_to_put_a_coordinate(self):
        """类型里没有位置字段 —— 这是「角色② 不得输出空间判断」的最后一道锁。"""
        fields = set(Attribute.__dataclass_fields__)
        assert fields == {"name", "value", "confidence", "source", "in_closed_set", "raw"}
        for bad in ("x", "y", "z", "xyz", "position", "bbox", "distance", "centroid"):
            assert bad not in fields


class TestImageSizeNormalisation:
    """`image_hw` 是 **H,W**，而 `image_size` 是 **W,H** —— 弄反了不会报错，只会悄悄裁错。

    所以这一节的每一条都在钉同一件事：**进门是 H,W，出门是 W,H**。
    """

    def test_image_hw_is_flipped_to_width_height(self):
        assert image_size_from_meta({"image_hw": [480, 640]}) == (640, 480)

    def test_explicit_width_height_keys_are_used_as_is(self):
        assert image_size_from_meta({"image_width": 640, "image_height": 480}) == (640, 480)

    def test_image_size_key_is_already_width_height(self):
        assert image_size_from_meta({"image_size": [640, 480]}) == (640, 480)

    def test_image_hw_wins_over_the_others_when_both_are_present(self):
        """builder 的输出（`image_hw`）优先 —— 它是权威来源，其余是历史写法。"""
        assert image_size_from_meta({"image_hw": [480, 640], "image_size": [1, 1]}) == (640, 480)

    def test_missing_meta_yields_none(self):
        assert image_size_from_meta({}) is None
        assert image_size_from_meta({"image_hw": [480]}) is None

    def test_region_for_node_uses_the_scene_meta(self):
        class Scene:
            build_meta = {"image_hw": [480, 640]}

        class Node:
            bbox_2d = (10.0, 20.0, 60.0, 90.0)
            mask_ref = "masks/chair_1.png"
            label = "chair"

        r = Region.for_node(Scene(), Node())
        assert r.image_size == (640, 480)
        assert r.bbox == (10.0, 20.0, 60.0, 90.0)
        assert r.label == "chair"

    def test_region_for_node_tolerates_a_scene_without_meta(self):
        class Scene:
            pass

        class Node:
            label = "chair"

        r = Region.for_node(Scene(), Node())
        assert r.image_size is None and r.bbox is None
