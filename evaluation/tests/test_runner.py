"""`evaluation/runner.py` 与 `evaluation/vadar_compat.py` 的单元测试。

零 torch、零 GPU、零联网、零模型：这里测的是**契约与选择逻辑**，
不是模型行为。真正跑模型的验证在 `--plan` + 一次 smoke 里。

## 为什么要测「题集选择」

VADAR 原版用无种子的 `random.sample` 抽 API 示例题（`evaluate.py:42`），
所以同一份数据两次跑会得到不同题 —— 数字无法归因。修掉它以后，
「确定性」本身就成了要守住的性质：一旦哪天有人顺手改回无种子版本，
测试必须立刻红。
"""

from __future__ import annotations

import json
import os
import sys

import pytest

from evaluation import vadar_compat
from evaluation.runner import metric_class, select_subset, subset_record


def q(i, at="int", ans="1"):
    return {"image_index": "img%d.jpg" % (i % 3), "question_index": i,
            "question": "q%d" % i, "answer_type": at, "answer": ans,
            "image_filename": "img%d.jpg" % (i % 3)}


class TestMetricClass:
    def test_maps_four_metric_classes(self):
        assert metric_class(q(0, "int")) == "numeric_count"
        assert metric_class(q(0, "float")) == "numeric_other"
        assert metric_class(q(0, "str", "yes")) == "yes_no"
        assert metric_class(q(0, "str", "no")) == "yes_no"
        assert metric_class(q(0, "str", "chair")) == "multi_choice"

    def test_unknown_type_is_labelled_not_silently_defaulted(self):
        assert metric_class(q(0, "bool", True)) == "unknown"


class TestSelectSubset:
    def test_vadarspec_matches_the_original_first_n_behaviour(self):
        pool = [q(i) for i in range(30)]
        got = select_subset(pool, 7, "vadarspec", 42)
        assert got == pool[:7]

    def test_vadarspec_n_larger_than_pool_returns_everything(self):
        pool = [q(i) for i in range(5)]
        assert select_subset(pool, 99, "vadarspec", 42) == pool

    def test_seeded_sample_is_deterministic(self):
        pool = [q(i) for i in range(100)]
        a = select_subset(pool, 10, "seeded-sample", 42)
        b = select_subset(pool, 10, "seeded-sample", 42)
        assert a == b
        assert len(a) == 10

    def test_different_seed_gives_different_selection(self):
        pool = [q(i) for i in range(100)]
        assert select_subset(pool, 10, "seeded-sample", 1) != \
               select_subset(pool, 10, "seeded-sample", 2)

    def test_stratify_fills_the_quota(self):
        pool = ([q(i, "int") for i in range(70)]
                + [q(100 + i, "float") for i in range(270)]
                + [q(400 + i, "str", "yes") for i in range(75)]
                + [q(500 + i, "str", "chair") for i in range(86)])
        got = select_subset(pool, 50, "stratify", 42)
        assert len(got) == 50
        classes = {c: 0 for c in ("numeric_count", "numeric_other", "yes_no", "multi_choice")}
        for item in got:
            classes[metric_class(item)] += 1
        # 按 70/270/75/86 的比例，50 题时每类都应有代表（这正是 stratify 的目的：
        # 原版取前 50 题会只覆盖 1 类，导致其余指标为 None）
        assert all(v > 0 for v in classes.values())

    def test_stratify_is_deterministic(self):
        pool = [q(i, "int") for i in range(10)] + [q(100 + i, "float") for i in range(10)]
        assert select_subset(pool, 6, "stratify", 7) == select_subset(pool, 6, "stratify", 7)

    def test_stratify_returns_no_duplicates(self):
        pool = [q(i, "int") for i in range(10)] + [q(100 + i, "float") for i in range(10)]
        got = select_subset(pool, 15, "stratify", 3)
        keys = [(x["image_index"], x["question_index"]) for x in got]
        assert len(keys) == len(set(keys))

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            select_subset([q(0)], 1, "nope", 0)


class TestSubsetRecord:
    def test_records_class_counts_and_selection(self):
        pool = [q(0, "int"), q(1, "float"), q(2, "str", "no"), q(3, "str", "table")]
        r = subset_record(pool, pool, "stratify", 42, 2, pool[:2])
        assert r["n_selected"] == 4
        assert r["metric_class_counts"] == {
            "numeric_count": 1, "numeric_other": 1, "yes_no": 1, "multi_choice": 1}
        assert len(r["selected"]) == 4
        assert len(r["api_subset"]) == 2
        json.dumps(r)          # 必须可落盘

    def test_selected_entries_carry_the_metric_class(self):
        pool = [q(0, "float")]
        r = subset_record(pool, pool, "vadarspec", 0, 0, [])
        assert r["selected"][0]["metric_class"] == "numeric_other"


# =====================================================================
# vadar_compat：只在**不导入 torch** 的范围内测
# =====================================================================
_STUB_NAMES = [
    "openai", "sam2", "sam2.build_sam", "sam2.sam2_image_predictor",
    "groundingdino", "groundingdino.datasets", "groundingdino.datasets.transforms",
    "groundingdino.util", "groundingdino.util.inference",
]


@pytest.fixture
def clean_stub_modules():
    """注册 stub 会改全局 sys.modules，测完必须还原，否则污染其他测试文件。"""
    saved = {n: sys.modules.get(n) for n in _STUB_NAMES}
    yield
    for n, old in saved.items():
        if old is None:
            sys.modules.pop(n, None)
        else:
            sys.modules[n] = old


class TestStubPackages:
    def test_registers_every_name_vadar_imports(self, clean_stub_modules):
        vadar_compat.ensure_stub_packages("all", verbose=False)
        for n in _STUB_NAMES:
            assert n in sys.modules, n

    def test_submodule_is_reachable_as_attribute(self, clean_stub_modules):
        """`import groundingdino.datasets.transforms as T` 要求
        父模块上有 `transforms` 属性 —— 只写 sys.modules 是不够的。"""
        vadar_compat.ensure_stub_packages("gdino", verbose=False)
        import groundingdino
        import groundingdino.datasets

        assert hasattr(groundingdino, "datasets")
        assert hasattr(groundingdino.datasets, "transforms")
        assert hasattr(groundingdino.util, "inference")

    def test_gdino_provides_load_model_and_predict(self, clean_stub_modules):
        vadar_compat.ensure_stub_packages("gdino", verbose=False)
        from groundingdino.util.inference import load_model, predict

        assert callable(load_model) and callable(predict)

    def test_transforms_expose_the_four_symbols_vadar_uses(self, clean_stub_modules):
        vadar_compat.ensure_stub_packages("gdino", verbose=False)
        from groundingdino.datasets import transforms as T

        for name in ("Compose", "RandomResize", "ToTensor", "Normalize"):
            assert hasattr(T, name), name

    def test_openai_stub_raises_instead_of_returning_none(self, clean_stub_modules):
        """静默返回 None 会让「Generator 没被替换」这件事拖到很晚才暴露。"""
        vadar_compat.ensure_stub_packages("all", verbose=False)
        import openai

        with pytest.raises(RuntimeError):
            openai.OpenAI(api_key="x")

    def test_sam2_stub_raises_with_an_explanation(self, clean_stub_modules):
        vadar_compat.ensure_stub_packages("all", verbose=False)
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        with pytest.raises(RuntimeError) as ei:
            SAM2ImagePredictor()
        assert "vision/segmentation.py" in str(ei.value)


class TestGdinoPredictContract:
    def test_rejects_non_handle_model(self, clean_stub_modules):
        vadar_compat.ensure_stub_packages("gdino", verbose=False)
        from groundingdino.util.inference import predict

        # 传一个「像模型但其实不是句柄」的对象 → 必须显式报错，
        # 而不是把 None 当成检测结果传下去（会静默变成 0 个框）
        with pytest.raises(TypeError):
            predict(model=object(), image=None, caption="sofa .")

    def test_load_model_reports_a_missing_weight_dir(self, clean_stub_modules, monkeypatch):
        vadar_compat.ensure_stub_packages("gdino", verbose=False)
        from groundingdino.util.inference import load_model

        monkeypatch.setenv("VADAR_GDINO_DIR", r"Z:\definitely\not\here")
        with pytest.raises(FileNotFoundError):
            load_model("cfg.py", "w.pth")


class TestCaptionMode:
    def test_default_keeps_vadar_caption_verbatim(self, monkeypatch):
        """臂 A 的意义就是保真：VADAR 的 caption 是 "sofa-table ."，
        默认必须原样送进去，不能「顺手修好看」。"""
        monkeypatch.delenv("VADAR_GDINO_CAPTION", raising=False)
        assert vadar_compat._caption_for_hf("sofa-table .") == "sofa-table ."

    def test_normalized_mode_offers_the_alternative(self, monkeypatch):
        monkeypatch.setenv("VADAR_GDINO_CAPTION", "normalized")
        assert vadar_compat._caption_for_hf("sofa-table .") == "sofa table."

    def test_normalized_mode_on_empty_caption(self, monkeypatch):
        monkeypatch.setenv("VADAR_GDINO_CAPTION", "normalized")
        assert vadar_compat._caption_for_hf(" .") == ""


class TestEvaluationModulesStayTorchFreeAtImport:
    """★ 守一条会**静默变绿**的回归。

    `tests/test_vision_exif.py` 里有一条哨兵用例，靠「本进程是否加载过 torch」
    判断某模块是否在模块级 import 了 torch。它的失败方式是 **skip**，
    而不是 fail —— 也就是说：只要 `evaluation/*` 里有一处在 import 期
    （或一被调用就）拉起 torch，那条断言就会安静地消失，
    整套测试照样全绿。这个坑真的踩过一次（`_gdino_predict` 把
    `import torch` 写在参数校验之前）。

    这里换成**静态检查**，不依赖进程状态、也不靠测试顺序。
    """

    MODULES = ["evaluation.runner", "evaluation.metrics",
               "evaluation.win_alarm", "evaluation.vadar_compat"]

    def _module_level_imports(self, path):
        import ast

        tree = ast.parse(open(path, "r", encoding="utf-8").read())
        names = set()
        for node in tree.body:                      # 只看模块级
            if isinstance(node, ast.Import):
                names.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split(".")[0])
        return names

    def test_no_module_level_torch_import(self):
        import os

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for mod in self.MODULES:
            fname = os.path.join(root, mod.split(".")[-1] + ".py")
            names = self._module_level_imports(fname)
            assert "torch" not in names, (
                "%s 在模块级 import 了 torch —— 请移到函数内" % mod)
            assert "numpy" not in names, (
                "%s 在模块级 import 了 numpy —— 这三个模块本该零第三方依赖" % mod)

    def test_gdino_predict_validates_before_importing_torch(self):
        """`predict()` 的参数校验必须在 `import torch` **之前**：

        顺序反了的话，一个「传错 model 参数」的失败调用也会把 torch
        拉进进程 —— 用异常测试根本看不出来，只有哨兵用例会变成 skip。
        这里用 AST 比较两条语句在函数体里的**先后**。
        （不能用字符串 find：注释里也会出现 `import torch` 这几个字，
        第一版就是这么写错的。）
        """
        import ast
        import os

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        src = open(os.path.join(root, "vadar_compat.py"), "r", encoding="utf-8").read()
        fn = next(n for n in ast.parse(src).body
                  if isinstance(n, ast.FunctionDef) and n.name == "_gdino_predict")

        def _is_torch_import(node):
            if isinstance(node, ast.Import):
                return any(a.name.split(".")[0] == "torch" for a in node.names)
            if isinstance(node, ast.ImportFrom):
                return (node.module or "").split(".")[0] == "torch"
            return False

        # 注意 raise 是嵌在 `if handle is None:` 里的，不是函数体的直接语句，
        # 所以必须用 ast.walk 取所有后代，并按行号比先后。
        lines_raise = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Raise)]
        lines_torch = [n.lineno for n in ast.walk(fn) if _is_torch_import(n)]
        assert lines_raise, "没找到参数校验的 raise"
        assert lines_torch, "没找到 import torch"
        assert min(lines_raise) < min(lines_torch), (
            "_gdino_predict 里 import torch 出现在参数校验之前（会污染 torch 哨兵用例）")


class TestRepoLayoutAssumptions:
    def test_vadar_repo_dir_must_be_named_vadar(self):
        """predefined_modules.py:17 硬编码 `from VADAR.prompts...`。
        路径写错的话这里会直接抛，而不是等到 import 时给一个难懂的 ImportError。"""
        import os

        assert os.path.basename(vadar_compat.VADAR_REPO_ROOT) == "VADAR"
        assert os.path.isfile(os.path.join(
            vadar_compat.VADAR_REPO_ROOT, "engine", "predefined_modules.py"))

    def test_default_gdino_dir_is_inside_the_project(self):
        """默认权重目录必须在项目内 —— 指到用户缓存会让实验不可移植。"""
        import os

        assert os.path.commonpath([vadar_compat.DEFAULT_GDINO_DIR,
                                   vadar_compat.PROJECT_ROOT]) \
            == os.path.normpath(vadar_compat.PROJECT_ROOT)


class TestArtifactRouting:
    """稳定入口的文件名分流。

    ## 为什么这值得单独一组测试

    2026-09-17 实测踩到：不带 `--plan` 直接跑，运行器在「缺 key」处以 return 2
    结束，但它**照旧写了 `latest_run.json`**。上一次 `--plan` 的记录里有
    `plan_summary.next_command`、有模型指纹 —— 全被一条 fatal 覆盖。
    想照着 next_command 复现，打开 latest_run.json 看到的是刚刚的失败现场。

    这不是「格式不对」，是**证据被销毁**。所以文件名分流要当成契约守住。
    """

    @staticmethod
    def _args(tmp_path, subset_file=None):
        import types

        return types.SimpleNamespace(results_root=str(tmp_path), arm="A",
                                     subset_file=subset_file)

    @staticmethod
    def _report(**kw):
        d = {"arm": "A", "started": "2026-09-17T21:00:00"}
        d.update(kw)
        return d

    def test_run_kind_writes_latest_run(self, tmp_path):
        from evaluation.runner import _write_report

        args = self._args(tmp_path)
        _write_report(args, self._report(), kind="run")
        assert (tmp_path / "A" / "latest_run.json").is_file()

    def test_fatal_kind_never_touches_latest_run(self, tmp_path):
        """核心回归：先有一次成功的 run，再来一次 fatal，前者必须还在。"""
        from evaluation.runner import _write_report

        args = self._args(tmp_path)
        _write_report(args, self._report(metrics={"submetrics": {"x": 1}}), kind="run")
        before = (tmp_path / "A" / "latest_run.json").read_text(encoding="utf-8")
        assert "submetrics" in before

        _write_report(args, self._report(fatal="缺少 VADAR_API_KEY"), kind="fatal")
        after = (tmp_path / "A" / "latest_run.json").read_text(encoding="utf-8")
        assert after == before, "fatal 态把上一次成功记录覆盖了"

        failed = json.loads((tmp_path / "A" / "latest_failed.json").read_text(encoding="utf-8"))
        assert "缺少 VADAR_API_KEY" in failed["fatal"]

    def test_plan_kind_has_its_own_entry(self, tmp_path):
        from evaluation.runner import _write_report

        args = self._args(tmp_path)
        _write_report(args, self._report(plan_summary={"next_command": "x"}),
                      kind="plan")
        plan = json.loads((tmp_path / "A" / "latest_plan.json").read_text(encoding="utf-8"))
        assert plan["plan_summary"]["next_command"] == "x"
        assert not (tmp_path / "A" / "latest_run.json").exists(), \
            "计划态不该凭空造出一个 latest_run.json"

    def test_reports_are_self_describing(self, tmp_path):
        """产物要自述类别，不能只靠文件名认人。"""
        from evaluation.runner import _write_report

        args = self._args(tmp_path)
        _write_report(args, self._report(), kind="fatal")
        got = json.loads((tmp_path / "A" / "latest_failed.json").read_text(encoding="utf-8"))
        assert got["mode"] == "fatal"

    def test_plan_mode_field_is_not_overwritten(self, tmp_path):
        """`--plan` 分支自己设了 mode="plan"，写入时不能把它改成别的。"""
        from evaluation.runner import _write_report

        args = self._args(tmp_path)
        _write_report(args, self._report(mode="plan"), kind="plan")
        got = json.loads((tmp_path / "A" / "latest_plan.json").read_text(encoding="utf-8"))
        assert got["mode"] == "plan"

    def test_fatal_does_not_overwrite_subset_json(self, tmp_path):
        """fatal 的题集从未执行过，不能顶掉上一次真跑过的题集。"""
        from evaluation.runner import _write_report

        args = self._args(tmp_path)
        _write_report(args, self._report(subset={"selected": [{"ran": True}]}), kind="run")
        sub_file = tmp_path / "A" / "subset.json"
        assert json.loads(sub_file.read_text(encoding="utf-8"))["selected"][0]["ran"]

        _write_report(args, self._report(subset={"selected": [{"ran": False}]}),
                      kind="fatal")
        assert json.loads(sub_file.read_text(encoding="utf-8"))["selected"][0]["ran"], \
            "fatal 态覆盖了 subset.json"

    def test_explicit_subset_file_is_never_rewritten(self, tmp_path):
        """`--subset-file` 复现模式：题集是**读进来的**，回写等于污染来源。"""
        from evaluation.runner import _write_report

        src = tmp_path / "origin.json"
        src.write_text('{"selected": []}', encoding="utf-8")
        args = self._args(tmp_path, subset_file=str(src))
        _write_report(args, self._report(subset={"selected": [{"x": 1}]}), kind="run")
        assert not (tmp_path / "A" / "subset.json").exists()



# =====================================================================
# 配置文件接线（2026-09-18）
#
# 背景：`06_deepseek_setup.ps1` 设的是**会话级**变量 —— 不落盘、传不进别的
# 进程，于是「运行器完整、--plan 全绿」也可能在真跑时因缺 key 直接 return 2。
# 改成读 `configs/llm_backend.env` 之后，「key 放在哪」变成一行可检查的事实。
# 这里守住的是**接线本身**：优先级、换文件、撤销、以及绝不外泄。
# =====================================================================
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#: 工作副本：含真 key，不进版本库/报告/截图。**不许**断言它「必须为空」。
LIVE = os.path.join(ROOT, "configs", "llm_backend.env")
#: 可分发模板：key 位留空，随仓库走。由 `tools/make_env_template.py` 生成。
TEMPLATE = os.path.join(ROOT, "configs", "llm_backend.env.template")
SECRET = "sk-0123456789abcdef0123456789abcdef"


@pytest.fixture
def isolated_env(monkeypatch):
    """隔离三样东西：`os.environ` 的内容、runner 记录的「文件写过的键」、
    以及上一份文件的原值表。

    为什么连 `os.environ` 本身一起换掉：

    * `use_env_file()` 按设计**真的**写环境变量（那正是它的职责），
      不隔离的话一个用例会污染其余用例；
    * `import evaluation.runner` 自己就会加载一次真模板 —— 于是 `VADAR_MODEL`
      这类键在**任何测试开始之前**就已经躺在 `os.environ` 里了。基线里混进
      「导入期残留」，会让「写坏的文件不许生效」变成在与噪声比对。所以先把
      导入期写进去的键从基线里剔掉，每个用例都从「不含配置文件的干净环境」出发。
    """
    from evaluation import runner

    polluted = set(runner._APPLIED_BY_FILE)      # 导入期由真模板写进去的键
    baseline = {k: v for k, v in os.environ.items() if k not in polluted}

    monkeypatch.setattr(os, "environ", baseline)
    monkeypatch.setattr(runner, "_APPLIED_BY_FILE", set())
    monkeypatch.setattr(runner, "_ENV_ORIG", {})
    monkeypatch.setattr(runner, "ENV_FILE_PATH", None)
    monkeypatch.setattr(runner, "ENV_FILE_REPORT", {})
    monkeypatch.setattr(runner, "ENV_FILE_ERROR", None)
    return runner


def env_file(tmp_path, text, name="e.env"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


class TestEnvFileWiring:
    def test_file_values_reach_environ_and_are_attributed_to_the_file(
            self, tmp_path, isolated_env):
        rep = isolated_env.use_env_file(
            env_file(tmp_path, "VADAR_API_KEY=%s\nVADAR_MODEL=m1\n" % SECRET))
        assert os.environ["VADAR_API_KEY"] == SECRET
        assert os.environ["VADAR_MODEL"] == "m1"
        assert rep["applied"] == ["VADAR_API_KEY", "VADAR_MODEL"]
        # 归因必须落在**文件**上：报告里要能回答"key 是哪来的"。
        assert isolated_env._api_key_source() == "env_file"
        # 指纹要有，明文不能有（下面还有一条专门扫序列化后的整份报告）。
        assert rep["secrets"]["VADAR_API_KEY"]["present"] is True
        assert rep["config_sha256"]

    def test_no_env_file_drops_what_the_file_had_written(self, tmp_path, isolated_env):
        """`--no-env-file` 必须真的"不用"，而不是"用了但假装没用"。"""
        runner = isolated_env
        runner.use_env_file(env_file(tmp_path, "VADAR_API_KEY=%s\n" % SECRET))
        assert os.environ.get("VADAR_API_KEY") == SECRET

        rep = runner.use_env_file(None)
        assert "VADAR_API_KEY" not in os.environ
        assert runner._api_key_source() == "missing"
        assert rep["exists"] is False

    def test_switching_env_files_clears_keys_only_the_old_one_had(
            self, tmp_path, isolated_env):
        """换文件时，上一份文件独有的键必须消失。

        反面：残留会让「换了配置却还在用旧值」静默发生 —— 而且看上去
        完全正常（新文件里那些键本来就没写，没人会去查一个不存在的键）。
        """
        runner = isolated_env
        runner.use_env_file(env_file(tmp_path, "VADAR_A=1\nVADAR_MODEL=old\n", "one.env"))
        runner.use_env_file(env_file(tmp_path, "VADAR_MODEL=new\n", "two.env"))
        assert os.environ["VADAR_MODEL"] == "new"
        assert "VADAR_A" not in os.environ

    def test_switching_files_restores_the_original_process_env_value(
            self, tmp_path, isolated_env):
        """被文件覆盖过的进程环境变量，撤销时要**还原**而不是删掉。

        反面：直接 pop 会把用户自己设的用户级/会话级变量一起抹掉，
        表现为"跑完一次实验，环境被改了"。
        """
        runner = isolated_env
        os.environ["VADAR_MODEL"] = "from_process"
        runner.use_env_file(env_file(tmp_path, "VADAR_MODEL=from_file\n"))
        assert os.environ["VADAR_MODEL"] == "from_file"

        runner.use_env_file(None)
        assert os.environ["VADAR_MODEL"] == "from_process"

    def test_broken_file_records_error_instead_of_raising(self, tmp_path, isolated_env):
        """写坏的文件 = 整轮停住，但**不带 traceback** 地停。

        `use_env_file()` 在导入期就会被调用（那时抛异常只会变成一段
        没人看的栈），所以它把错误**记下来**交给 main() 转成 fatal 产物。
        要求是：错误记得住、且一个键都不许生效。
        """
        runner = isolated_env
        rep = runner.use_env_file(env_file(tmp_path, "VADAR_MODEL=a\nVADAR_MODEL=b\n"))
        assert runner.ENV_FILE_ERROR and "重复" in runner.ENV_FILE_ERROR
        assert rep["applied"] == []
        assert "VADAR_MODEL" not in os.environ

    def test_missing_file_is_quiet(self, tmp_path, isolated_env):
        rep = isolated_env.use_env_file(str(tmp_path / "nope.env"))
        assert rep["exists"] is False and rep["error"] is None
        assert isolated_env.ENV_FILE_ERROR is None

    def test_serialized_report_never_contains_the_key(self, tmp_path, isolated_env):
        """整份报告序列化后逐字符搜 —— 这条拦的是**将来新增的字段**。

        断言"某个字段被掩码了"是点，断言"整份产物里搜不到明文"是面。
        """
        rep = isolated_env.use_env_file(env_file(tmp_path, "VADAR_API_KEY=%s\n" % SECRET))
        blob = json.dumps(rep, ensure_ascii=False)
        assert SECRET not in blob
        assert SECRET[:-4] not in blob


class TestTemplateHygiene:
    """模板本身的性质 —— 它是**要被人打开编辑**的文件，所以它的结构性缺陷
    会直接变成用户的操作失误。

    ⚠ 「key 必须为空」这条规矩只对**模板**成立，不对工作副本成立。

    最初这两条断言是直接打在 `configs/llm_backend.env` 上的，于是用户
    **正确填了 key 的那一刻，测试就永久变红**。永久红灯比没有测试更危险：
    它会训练所有人忽略红色，真正的泄露反倒没人看见。
    ⟹ 「谁的 key 必须空」由**文件名**回答（`.template` vs 无名后缀），
      不能由**内容**回答。
    """

    def test_the_documented_default_path_exists(self):
        assert os.path.isfile(LIVE), "configs/llm_backend.env 是配置的默认位置"

    def test_distributable_template_exists(self):
        assert os.path.isfile(TEMPLATE), \
            "模板应由 tools/make_env_template.py 生成（缺了就 difftool 一下再补）"

    def test_template_ships_with_an_empty_key_slot(self):
        """**模板**里的 key 必须是空的。

        反面（这条测试的真正目的）：一旦有人把自己的 key 存进了模板，
        这个文件就会跟着仓库/备份/截图一起扩散 —— 而它看起来「只是一份配置」。
        """
        import vadar_env
        values = vadar_env.parse_env_text(open(TEMPLATE, encoding="utf-8").read())
        assert values.get("VADAR_API_KEY") == ""
        assert values.get("VADAR_VISION_API_KEY") == ""

    def test_template_parses_without_errors(self):
        """模板必须能被自己的解析器读通（否则用户一打开就踩空）。"""
        import vadar_env
        values = vadar_env.parse_env_text(open(TEMPLATE, encoding="utf-8").read())
        assert "VADAR_MODEL" in values and "VADAR_BASE_URL" in values

    def test_template_keeps_non_secret_defaults(self):
        """`VADAR_MAX_TOKENS` 名字里有 TOKEN 但不是密钥，**不能被当成密钥清掉**。

        清掉的后果很隐蔽：模板能跑，但用的是代码默认值而不是文档写的值。
        """
        import vadar_env
        values = vadar_env.parse_env_text(open(TEMPLATE, encoding="utf-8").read())
        assert values.get("VADAR_MAX_TOKENS") == "8192"
        assert not vadar_env.is_secret("VADAR_MAX_TOKENS")

    def test_template_and_working_copy_agree_on_key_set(self):
        """模板与工作副本的**键集合**必须一致（密钥的值当然可以不同）。

        这条拦的是另一种静默腐化：给工作副本加了新选项、忘了重新生成模板，
        于是新用户拿到的配置缺少这个选项，而且**不会报错** —— 只会静默
        走代码默认值。修法就是重跑 `tools/make_env_template.py`。
        """
        if not os.path.isfile(LIVE) or not os.path.isfile(TEMPLATE):
            pytest.fail("工作副本或模板缺失，无法核对漂移")
        import vadar_env
        live = set(vadar_env.parse_env_text(open(LIVE, encoding="utf-8").read()))
        tpl = set(vadar_env.parse_env_text(open(TEMPLATE, encoding="utf-8").read()))
        assert live == tpl, ("模板已漂移：只在工作副本里 %s；只在模板里 %s；"
                             "重跑 tools/make_env_template.py"
                             % (sorted(live - tpl), sorted(tpl - live)))
