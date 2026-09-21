"""`llm_env.py` 的单元测试 —— 零依赖、零 GPU、零联网。

为什么这组测试值得单独立一套
============================
配置加载器是**所有实验的入口**，它的 bug 不会让程序崩溃，只会让实验结果
不可信：

* 值读错了 → 跑出来的数字没人能解释；
* 密钥泄进报告 → 一次不可逆的泄露（`results/` 会归档、会被贴进汇报）；
* 文件写坏了被"容错"跳过 → 拿默认值跑完，产出一份**看起来正常**的结果。

这三种失败里，只有第一种会被察觉。所以这里按「失败要响」来写断言：
**不是断言"能读到值"，而是断言"读错的方式都会浮出来"。**

下面每条 `# 反面` 注释标出的用例，都是**反着写会通过**的那种
—— 只测 happy path 的测试抓不到它们。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import llm_env  # noqa: E402

SECRET = "sk-0123456789abcdef0123456789abcdef"


def write(tmp_path, text, name="llm_backend.env", encoding="utf-8"):
    p = tmp_path / name
    p.write_text(text, encoding=encoding)
    return str(p)


# =====================================================================
# 1. 解析
# =====================================================================
class TestParse:
    def test_basic_key_value(self):
        v = llm_env.parse_env_text("A=1\nB=two\n")
        assert v == {"A": "1", "B": "two"}

    def test_spaces_around_equals_are_stripped(self):
        assert llm_env.parse_env_text("A = 1 ") == {"A": "1"}

    def test_comment_only_at_line_start(self):
        """`#` 在行首是注释；在行尾**不是**。"""
        v = llm_env.parse_env_text("#A=1\nB=2\n   # C=3\nD=4\n")
        assert v == {"B": "2", "D": "4"}

    def test_hash_inside_value_is_kept(self):
        # 反面：若支持行尾注释，这里会静默变成 "abc"（截断的配置）。
        # 一个值里带 # 的键真实存在（中转端点的 key、带 fragment 的 URL）。
        assert llm_env.parse_env_text("K=abc#def")["K"] == "abc#def"

    def test_whole_value_wrapped_in_quotes_loses_one_layer(self):
        v = llm_env.parse_env_text("A=\"1 2\"\nB='3 4'\n")
        assert v == {"A": "1 2", "B": "3 4"}

    def test_json_value_is_not_unquoted(self):
        """`SPATIAL_EXTRA_BODY={"thinking": ...}` 首字符是 `{`，必须原样保留。

        这是本项目里最要命的一个值：早期版本曾把它当命令行参数传给
        原生 exe，宿主 shell 剥掉内层双引号 → 不是合法 JSON → 整轮自检失败。
        在文件里它必须是**逐字节**安全的。
        """
        raw = '{"thinking": {"type": "disabled"}}'
        assert llm_env.parse_env_text("SPATIAL_EXTRA_BODY=" + raw)["SPATIAL_EXTRA_BODY"] == raw

    def test_empty_value_is_kept_as_empty_string(self):
        assert llm_env.parse_env_text("A=\n") == {"A": ""}

    def test_export_prefix_accepted(self):
        assert llm_env.parse_env_text("export A=1")["A"] == "1"

    def test_crlf_does_not_leave_carriage_return(self):
        # Windows 上编辑器很容易把文件存成 CRLF；`\r` 若留在值尾，
        # 会变成一个"看不见的字符"渗进 API key（401 却查不出原因）。
        assert llm_env.parse_env_text("A=1\r\nB=2\r\n") == {"A": "1", "B": "2"}

    def test_duplicate_key_raises(self):
        # 反面：若取最后一个（或第一个），改配置时忘注释旧行 = 静默用错值。
        with pytest.raises(llm_env.EnvFileError) as e:
            llm_env.parse_env_text("A=1\nB=2\nA=3\n")
        assert "重复" in str(e.value)

    def test_line_without_equals_raises(self):
        with pytest.raises(llm_env.EnvFileError):
            llm_env.parse_env_text("A=1\n这不是配置\n")

    def test_bad_key_name_raises(self):
        with pytest.raises(llm_env.EnvFileError):
            llm_env.parse_env_text("1BAD=1\n")


# =====================================================================
# 2. 密钥处理
# =====================================================================
class TestSecretHandling:
    def test_typical_secret_names_are_secrets(self):
        for k in ("SPATIAL_API_KEY", "DEEPSEEK_API_KEY", "ACCESS_TOKEN",
                  "CLIENT_SECRET", "DB_PASSWORD"):
            assert llm_env.is_secret(k), k

    def test_max_tokens_is_not_a_secret(self):
        """`TOKEN` 太容易误伤 —— `MAX_TOKENS` 是数量上限。

        2026-09-18 实测：报告里它被显示成 `***8192`，
        而且 `secrets.max_tokens.fingerprint` 里还写了一串哈希。
        """
        assert not llm_env.is_secret("SPATIAL_MAX_TOKENS")
        assert not llm_env.is_secret("SPATIAL_VISION_MAX_TOKENS")

    def test_mask_never_reveals_more_than_last_four(self):
        assert llm_env.mask(SECRET) == "***" + SECRET[-4:]
        assert SECRET[:-4] not in llm_env.mask(SECRET)

    def test_mask_of_empty_and_short(self):
        assert llm_env.mask("") == "(empty)"
        assert llm_env.mask(None) == "(empty)"
        assert llm_env.mask("abc") == "***"

    def test_fingerprint_is_stable_and_not_reversible(self):
        a = llm_env.fingerprint(SECRET)
        assert a == llm_env.fingerprint(SECRET)          # 稳定
        assert len(a) == 8
        assert SECRET[:8] not in a                          # 不是明文的截断
        assert a == hashlib.sha256(SECRET.encode()).hexdigest()[:8]
        assert llm_env.fingerprint("") is None

    def test_fingerprint_distinguishes_two_keys(self):
        assert llm_env.fingerprint("sk-aaa") != llm_env.fingerprint("sk-bbb")


# =====================================================================
# 3. 加载：优先级、留空、冲突
# =====================================================================
class TestLoad:
    def test_missing_file_is_not_an_error(self):
        env = {}
        rep = llm_env.load_env_file("does/not/exist.env", environ=env)
        assert rep["exists"] is False and rep["error"] is None
        assert env == {}                       # 什么都没写进去

    def test_none_path_is_tolerated(self):
        assert llm_env.load_env_file(None, environ={})["exists"] is False

    def test_values_land_in_environ(self, tmp_path):
        env = {}
        rep = llm_env.load_env_file(write(tmp_path, "A=1\nB=2\n"), environ=env)
        assert env == {"A": "1", "B": "2"}
        assert rep["applied"] == ["A", "B"] and rep["exists"] is True

    def test_file_wins_over_process_env_and_conflict_is_recorded(self, tmp_path):
        """文件优先，但**冲突必须留痕**。

        反面：如果静默覆盖，"我在文件里改了值，怎么没生效"与
        "我改了值，怎么生效了另一个"这两种现象都无法排查。
        """
        env = {"SPATIAL_MODEL": "deepseek-v4-pro"}
        rep = llm_env.load_env_file(write(tmp_path, "SPATIAL_MODEL=deepseek-flash\n"),
                                      environ=env)
        assert env["SPATIAL_MODEL"] == "deepseek-flash"
        assert len(rep["conflicts"]) == 1
        c = rep["conflicts"][0]
        assert c["key"] == "SPATIAL_MODEL" and c["took"] == "env_file"

    def test_identical_value_is_not_a_conflict(self, tmp_path):
        env = {"SPATIAL_MODEL": "deepseek-flash"}
        rep = llm_env.load_env_file(write(tmp_path, "SPATIAL_MODEL=deepseek-flash\n"),
                                      environ=env)
        assert rep["conflicts"] == [] and rep["kept_from_env"] == ["SPATIAL_MODEL"]

    def test_empty_value_does_not_clobber_process_env(self, tmp_path):
        """模板里留 `SPATIAL_API_KEY=` 是常态，它不该顶掉真实存在的环境变量。

        反面：如果空串按"有值"处理，那么"用户设了用户级环境变量 key"
        这条路会被一个空模板行静默废掉 —— 表现是「我明明设了却报缺 key」。
        """
        env = {"SPATIAL_API_KEY": SECRET}
        rep = llm_env.load_env_file(write(tmp_path, "SPATIAL_API_KEY=\n"), environ=env)
        assert env["SPATIAL_API_KEY"] == SECRET
        assert rep["empty_values"] == ["SPATIAL_API_KEY"]
        assert rep["applied"] == []

    def test_override_false_keeps_process_env(self, tmp_path):
        env = {"A": "old"}
        llm_env.load_env_file(write(tmp_path, "A=new\n"), environ=env, override=False)
        assert env["A"] == "old"

    def test_bom_is_tolerated(self, tmp_path):
        """PowerShell 的 `Out-File -Encoding utf8` 会写 BOM。

        反面：BOM 留在第一个键名前面 → `\\ufeffSPATIAL_API_KEY` 不是
        `SPATIAL_API_KEY`，键名整体错位，而报错方式只是"缺 key"。
        """
        p = write(tmp_path, "SPATIAL_API_KEY=" + SECRET + "\n", encoding="utf-8-sig")
        env = {}
        llm_env.load_env_file(p, environ=env)
        assert env == {"SPATIAL_API_KEY": SECRET}

    def test_unknown_key_is_reported_but_not_fatal(self, tmp_path):
        # 拼错一个键名否则完全静默（探针会显示默认值，看起来很正常）。
        rep = llm_env.load_env_file(write(tmp_path, "SPATIAL_BASE_URLL=x\n"), environ={})
        assert rep["unknown_keys"] == ["SPATIAL_BASE_URLL"]

    def test_all_template_keys_are_known(self):
        """模板里出现的每个键都必须在 KNOWN_KEYS 里 —— 否则用户一填就吃告警。

        这条同时守住"模板里不出现没有任何代码读的键"：这类键填了也不生效，
        是最典型的一类静默空操作。
        """
        tpl = ROOT / "configs" / "llm_backend.env.template"
        if not tpl.is_file():
            pytest.skip("模板不存在")
        keys = llm_env.parse_env_text(tpl.read_text(encoding="utf-8"))
        assert keys, "模板里应该至少有一个未注释的键"
        assert set(keys) <= llm_env.KNOWN_KEYS, sorted(set(keys) - llm_env.KNOWN_KEYS)

    def test_known_keys_cover_adapter_aliases(self):
        """`ENV_ALIASES` 里的每个键都必须在 KNOWN_KEYS 里（否则代码自己会告警一次）。

        用 `ast` 静态取键，**不 import adapter** —— 静态检查足够，
        而且不会在测试进程里留下任何副作用（环境变量、日志句柄）。

        ⚠ 2026-09-21 修复：这里原先只认 `ast.Assign`，而 `ENV_ALIASES` 是
        **带注解的赋值**（`AnnAssign`）⟹ 循环一次都没进过，`keys` 恒为空集，
        于是「每个键都在 KNOWN_KEYS 里」这句断言在**空集上永远成立**之前，
        先被最后那行 `assert keys` 挡住 —— 该守卫**从未真正生效**，
        而它失败时的表现是「测试挂了」，很容易被当成环境问题绕过。
        **教训同 §14：看不到，不是没有。** 所以下面补一条数量下限：
        只守护「找到了」不够，还要守护「找到的是全部 16 个」。
        """
        import ast
        tree = ast.parse((ROOT / "llm" / "adapter.py").read_text(encoding="utf-8"))
        keys = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign):
                targets: list = [node.target]           # `X: T = {...}`
            elif isinstance(node, ast.Assign):
                targets = list(node.targets)            # `X = {...}`
            else:
                continue
            if not any(getattr(t, "id", None) == "ENV_ALIASES" for t in targets):
                continue
            value = node.value
            assert isinstance(value, ast.Dict), "ENV_ALIASES 应是字面量 dict"
            for entry in value.values:
                for elt in getattr(entry, "elts", []):
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        keys.add(elt.value)
        assert keys, "没在 llm/adapter.py 里静态找到 ENV_ALIASES"
        #: 下限而非等值 —— 加键不该让这条守卫变红，但「只剩两三个」必须是红的。
        assert len(keys) >= 16, "只解析出 %d 个键，解析逻辑疑似又变瞎了：%s" % (
            len(keys), sorted(keys))
        assert not (keys - llm_env.KNOWN_KEYS), sorted(keys - llm_env.KNOWN_KEYS)


# =====================================================================
# 4. 指纹：能归因，但不泄密
# =====================================================================
class TestDigest:
    def test_digest_is_stable_across_line_order(self, tmp_path):
        a = llm_env.load_env_file(write(tmp_path, "A=1\nB=2\n", "a.env"), environ={})
        b = llm_env.load_env_file(write(tmp_path, "B=2\nA=1\n", "b.env"), environ={})
        assert a["config_sha256"] == b["config_sha256"]

    def test_digest_changes_when_a_value_changes(self, tmp_path):
        a = llm_env.load_env_file(write(tmp_path, "A=1\n", "a.env"), environ={})
        b = llm_env.load_env_file(write(tmp_path, "A=2\n", "b.env"), environ={})
        assert a["config_sha256"] != b["config_sha256"]

    def test_digest_does_not_contain_the_secret(self, tmp_path):
        """换 key **不**改指纹 —— 这是刻意的。

        指纹是给"这次和上次是不是同一套配置"用的；把密钥本身纳入摘要，
        就等于把一个可以公开的 8 位哈希变成了密钥的破解目标。
        「是不是同一把 key」由 `secrets[*].fingerprint` 回答。
        """
        a = llm_env.load_env_file(
            write(tmp_path, "SPATIAL_API_KEY=sk-aaaa\n", "a.env"), environ={})
        b = llm_env.load_env_file(
            write(tmp_path, "SPATIAL_API_KEY=sk-bbbb\n", "b.env"), environ={})
        assert a["config_sha256"] == b["config_sha256"]
        assert a["secrets"]["SPATIAL_API_KEY"]["fingerprint"] != \
            b["secrets"]["SPATIAL_API_KEY"]["fingerprint"]

    def test_secret_presence_is_visible_in_report(self, tmp_path):
        rep = llm_env.load_env_file(
            write(tmp_path, "SPATIAL_API_KEY=" + SECRET + "\n"), environ={})
        s = rep["secrets"]["SPATIAL_API_KEY"]
        assert s["present"] is True and s["fingerprint"]


# =====================================================================
# 5. ⭐ 报告里绝不出现密钥明文（本文件最重要的一组断言）
# =====================================================================
class TestNoLeak:
    def test_report_json_does_not_contain_the_secret(self, tmp_path):
        """把整份报告序列化后逐字符搜一遍 key 明文。

        这条之所以比「检查某个字段被掩码了」强：将来任何人往报告里加字段，
        都会自动被它拦住 —— 掩码是"点"的断言，泄漏是"面"的失败。
        """
        rep = llm_env.load_env_file(
            write(tmp_path, "SPATIAL_API_KEY=" + SECRET + "\nSPATIAL_MODEL=m\n"), environ={})
        blob = json.dumps(rep, ensure_ascii=False)
        assert SECRET not in blob
        # 连"大部分 key"也不许出现：掩码只能留最后 4 位
        assert SECRET[:-4] not in blob
        assert SECRET[:16] not in blob

    def test_conflict_record_masks_both_sides(self, tmp_path):
        env = {"SPATIAL_API_KEY": "sk-envside0000"}
        rep = llm_env.load_env_file(
            write(tmp_path, "SPATIAL_API_KEY=" + SECRET + "\n"), environ=env)
        blob = json.dumps(rep, ensure_ascii=False)
        assert SECRET not in blob and "sk-envside0000" not in blob
        assert rep["conflicts"][0]["env_file"].startswith("***")

    def test_describe_is_safe_to_print(self, tmp_path):
        rep = llm_env.load_env_file(
            write(tmp_path, "SPATIAL_API_KEY=" + SECRET + "\nSPATIAL_X=1\n"), environ={})
        text = llm_env.describe(rep)
        assert SECRET not in text and "SPATIAL_X" in text


# =====================================================================
# 6. locate()：把「填在第几行」指准
# =====================================================================
class TestLocate:
    def test_line_number_is_correct(self, tmp_path):
        p = write(tmp_path, "# c\n\nA=1\nB=2\n")
        assert llm_env.locate(p, "B") == 4
        assert llm_env.locate(p, "A") == 3

    def test_missing_key_returns_none(self, tmp_path):
        assert llm_env.locate(write(tmp_path, "A=1\n"), "ZZ") is None

    def test_missing_file_returns_none(self, tmp_path):
        assert llm_env.locate(str(tmp_path / "nope.env"), "A") is None

    def test_broken_file_returns_none_instead_of_raising(self, tmp_path):
        """定位失败不该把"报缺 key"的主错误盖成"文件坏了"。"""
        assert llm_env.locate(write(tmp_path, "A=1\nA=2\n"), "A") is None

    def test_real_template_points_at_the_api_key_line(self):
        """**模板**里那一行确实写着 `SPATIAL_API_KEY=`（否则缺 key 的提示会指错地方）。

        为什么打在模板上而不是工作副本上：工作副本是用户填 key 的地方，
        断言它「必须为空」会让用户一正确使用就永久红灯 —— 见
        `evaluation/tests/test_runner.py::TestTemplateHygiene` 的说明。
        """
        tpl = ROOT / "configs" / "llm_backend.env.template"
        # 不用 skip：本项目把 skipped 当失败处理，哨兵用例不许有「悄悄退场」的失败方式
        assert tpl.is_file(), "模板缺失，跑 tools/make_env_template.py 生成"
        line = llm_env.locate(str(tpl), "SPATIAL_API_KEY")
        assert line, "模板里必须有 SPATIAL_API_KEY 这一行"
        text = tpl.read_text(encoding="utf-8").splitlines()
        assert text[line - 1].strip() == "SPATIAL_API_KEY="

    def test_locate_works_on_the_working_copy_regardless_of_its_value(self):
        """工作副本里那一行也要定位得到 —— 但**只断言行为，不断言值**。

        这是「缺 key 提示指向第几行」这个功能真正需要的东西：
        行号对得上就够了，值是空是满不关测试的事。
        """
        live = ROOT / "configs" / "llm_backend.env"
        assert live.is_file(), "工作副本缺失"
        line = llm_env.locate(str(live), "SPATIAL_API_KEY")
        assert line, "工作副本里必须有 SPATIAL_API_KEY 这一行"
        text = live.read_text(encoding="utf-8").splitlines()
        assert text[line - 1].strip().startswith("SPATIAL_API_KEY=")


# =====================================================================
# 7. describe()：人读的一行摘要
# =====================================================================
class TestDescribe:
    def test_missing_file(self):
        assert "不存在" in llm_env.describe(
            llm_env.load_env_file("nope.env", environ={}))

    def test_no_path(self):
        assert llm_env.describe({"path": ""}) == "未使用配置文件"

    def test_error_is_surfaced(self):
        assert "有错" in llm_env.describe({"path": "x", "error": "坏了"})

    def test_summary_mentions_conflicts_and_unknown(self, tmp_path):
        env = {"A": "old"}
        rep = llm_env.load_env_file(
            write(tmp_path, "A=new\nZZZ=1\nB=\n"), environ=env)
        text = llm_env.describe(rep)
        assert "冲突" in text and "ZZZ" in text and "留空" in text
