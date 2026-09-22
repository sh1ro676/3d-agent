"""`scripts/serve_demo.py` 的守卫与任务登记表 —— 零 GPU、零联网、零 LLM。

盯三件事：

**B2 建图任务淘汰不能踢掉还在排队/运行中的任务。**
   原先的写法是 `while len(_BUILD_ORDER) > 8: pop(0)` —— **不看状态**。
   提交超过 8 个任务时，最旧的那个会被踢掉，而它很可能还在排队
   （`_BUILD_LOCK` 串行，一次只跑一个）。后果静默：`/api/build/status` 回 404
   「没有这个任务」，而 `_run_build_job` 开头的 `_BUILD_JOBS[job_id]` 会直接
   `KeyError` 死掉、任务永远停在 `"queued"`。

   这里用**真的 HTTP 服务端**来验，而不是直接调那个淘汰函数：缺陷在
   「提交路径有没有接上正确判据」这一层，只测函数会把它漏掉。

**B5 每个 HTTP 方法都要过同一道来源守卫。**
   `do_GET` / `do_HEAD` 原先没过 `_guard()`。于是 `/api/health`（带后端 base_url、
   model、env 文件路径）、`/api/runs`（全部历史问答）、建图日志尾部与**整个静态目录**
   对任何来源都是敞开的 —— 而 `--host` 改成 `0.0.0.0` 是演示投屏时很容易发生的事。

**A4 问答预算的取值与收紧规则。**
   客户端只能收紧不能放宽；非法值必须报 400 而不是静默忽略。
"""

from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import serve_demo  # noqa: E402
from serve_demo import (  # noqa: E402
    ANSWER_BUDGET_S,
    ASK_BUDGET_ENV,
    TERMINAL_JOB_STATES,
    DemoHandler,
    _ask_budget,
    _evict_finished_jobs_locked,
)


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registry():
    """每个用例前后都清空任务登记表 —— 它是模块级可变状态，会串味。"""
    serve_demo._BUILD_JOBS.clear()
    serve_demo._BUILD_ORDER.clear()
    yield
    serve_demo._BUILD_JOBS.clear()
    serve_demo._BUILD_ORDER.clear()


@pytest.fixture
def server(tmp_path, monkeypatch):
    """真的起一个服务端（绑 127.0.0.1:0 拿随机端口）。

    用真服务而不是直接调 handler 方法：`do_GET` 里那句 `super().do_GET()`
    在脱离 `DemoHandler` 实例时根本跑不通，而"守卫有没有真的接上"这件事
    只有走完整请求路径才算验过。
    """
    static = tmp_path / "demo"
    (static / "data").mkdir(parents=True)
    (static / "index.html").write_text("<html>stub</html>", encoding="utf-8")
    (static / "data" / "index.json").write_text(
        json.dumps({"scenes": [], "generated_at": "stub"}), encoding="utf-8")
    monkeypatch.setattr(serve_demo, "STATIC_ROOT", static)
    # 别把测试的日志写进仓库那份 demo_server.log
    monkeypatch.setattr(serve_demo, "LOG_FILE", tmp_path / "test.log")

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), DemoHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd.server_address[1]
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _request(port: int, path: str, *, method: str = "GET", body: dict | None = None):
    """返回 `(status, body_bytes)`；HTTP 错误码也当普通返回值，不抛。"""
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _deny_guard(monkeypatch, calls: list) -> None:
    """把守卫换成「一律拒绝」，并记录它被调用过。"""
    def deny(self):                     # noqa: ANN001
        calls.append(self.command)
        self._json(403, {"ok": False, "error": "只接受本机请求"})
        return False

    monkeypatch.setattr(DemoHandler, "_guard", deny)


# ---------------------------------------------------------------------------
# B5：每个方法都过守卫
# ---------------------------------------------------------------------------


class TestGuardCoversEveryMethod:
    def test_get_endpoints_work_when_the_guard_passes(self, server):
        """前提检查：正常本机请求下 GET 必须能用（否则下面的 403 没有意义）。"""
        status, body = _request(server, "/api/health")
        assert status == 200
        payload = json.loads(body)
        assert payload["ok"] is True

    def test_get_goes_through_the_guard(self, server, monkeypatch):
        """★ 修复前这条会红：`do_GET` 完全没调 `_guard()`。"""
        calls: list[str] = []
        _deny_guard(monkeypatch, calls)

        status, _ = _request(server, "/api/health")

        assert status == 403, "GET 没过守卫 —— /api/health 对任何来源都敞开"
        assert calls == ["GET"], "守卫必须被真的调用，而不是只存在于代码里"

    def test_static_files_go_through_the_guard(self, server, monkeypatch):
        """静态目录同样要守：它含全部场景图与前端资源。"""
        calls: list[str] = []
        _deny_guard(monkeypatch, calls)

        status, _ = _request(server, "/index.html")

        assert status == 403
        assert calls == ["GET"]

    def test_head_goes_through_the_guard(self, server, monkeypatch):
        """`do_HEAD` 是**单独**的方法（不走 `do_GET`），漏了就是一个缺口。"""
        calls: list[str] = []
        _deny_guard(monkeypatch, calls)

        status, _ = _request(server, "/index.html", method="HEAD")

        assert status == 403
        assert calls == ["HEAD"]

    def test_post_still_goes_through_the_guard(self, server, monkeypatch):
        """不回归：POST 原本就守，改完之后必须还守着。"""
        calls: list[str] = []
        _deny_guard(monkeypatch, calls)

        status, _ = _request(server, "/api/ask", method="POST", body={})

        assert status == 403
        assert calls == ["POST"]

    def test_the_guard_runs_before_reading_the_body(self, server, monkeypatch):
        """守卫必须在**读请求体之前** —— 否则一个非本机请求也能先让我们读满内存。"""
        order: list[str] = []
        real_read = DemoHandler._read_json

        def spy_read(self):             # noqa: ANN001
            order.append("read_body")
            return real_read(self)

        def deny(self):                 # noqa: ANN001
            order.append("guard")
            self._json(403, {"ok": False, "error": "只接受本机请求"})
            return False

        monkeypatch.setattr(DemoHandler, "_guard", deny)
        monkeypatch.setattr(DemoHandler, "_read_json", spy_read)

        _request(server, "/api/ask", method="POST", body={"question": "x"})

        assert order == ["guard"], "守卫必须在读请求体之前生效"


# ---------------------------------------------------------------------------
# B2：淘汰只看数量、不看状态 ⟹ 踢掉还在排队/运行的任务
# ---------------------------------------------------------------------------


def _put(job_id: str, state: str) -> dict:
    job = {"id": job_id, "state": state, "tail": [], "stage": state}
    serve_demo._BUILD_JOBS[job_id] = job
    serve_demo._BUILD_ORDER.append(job_id)
    return job


class TestJobEviction:
    def test_under_the_cap_nothing_is_dropped(self):
        for i in range(serve_demo._JOB_KEEP):
            _put("j%d" % i, "done")
        assert _evict_finished_jobs_locked() == []
        assert len(serve_demo._BUILD_ORDER) == serve_demo._JOB_KEEP

    def test_finished_jobs_are_dropped_oldest_first(self):
        _put("oldest", "done")
        for i in range(serve_demo._JOB_KEEP - 1):
            _put("j%d" % i, "done")
        _put("newest", "failed")

        dropped = _evict_finished_jobs_locked()

        assert dropped == ["oldest"]
        assert "oldest" not in serve_demo._BUILD_JOBS
        assert "newest" in serve_demo._BUILD_JOBS

    def test_queued_jobs_are_never_dropped(self):
        """★ 核心用例：排队中的任务不能因为「排在第 0 位」就被踢掉。"""
        _put("still_queued", "queued")
        for i in range(serve_demo._JOB_KEEP):
            _put("f%d" % i, "done")

        dropped = _evict_finished_jobs_locked()

        assert "still_queued" in serve_demo._BUILD_JOBS, \
            "排队中的任务被淘汰了 —— 它会 404，而线程会 KeyError"
        assert "still_queued" in serve_demo._BUILD_ORDER
        assert dropped, "该淘汰的是那些**已终结**的，不能一个都不淘汰"

    def test_running_jobs_are_never_dropped(self):
        _put("running_job", "running")
        for i in range(serve_demo._JOB_KEEP + 3):
            _put("f%d" % i, "done")

        dropped = _evict_finished_jobs_locked()

        assert "running_job" in serve_demo._BUILD_JOBS
        assert "running_job" in serve_demo._BUILD_ORDER
        assert "running_job" not in dropped
        assert len(dropped) == (serve_demo._JOB_KEEP + 4) - serve_demo._JOB_KEEP

    def test_all_live_jobs_are_kept_even_over_the_cap(self):
        """全是活动任务时**允许超编** —— 宁可多留几条 dict，也不能丢掉正在跑的活。"""
        for i in range(serve_demo._JOB_KEEP + 4):
            _put("live%d" % i, "queued")

        assert _evict_finished_jobs_locked() == []
        assert len(serve_demo._BUILD_JOBS) == serve_demo._JOB_KEEP + 4

    def test_every_listed_id_still_resolves(self):
        """不变量：`_BUILD_ORDER` 里的每个 id 都必须还能在表里查到。

        `_run_build_job` 一开头就按 id 查表，查不到就是 `KeyError` ——
        这条不变量正是那个 `KeyError` 的反面。
        """
        _put("queued_a", "queued")
        _put("done_a", "done")
        _put("running_b", "running")
        for i in range(serve_demo._JOB_KEEP):
            _put("done_%d" % i, "done")

        _evict_finished_jobs_locked()

        for jid in serve_demo._BUILD_ORDER:
            assert jid in serve_demo._BUILD_JOBS, "%s 在顺序表里但查不到" % jid

    def test_terminal_states_are_exactly_done_and_failed(self):
        """判据本身也要钉住：加新状态时必须显式决定它算不算「可淘汰」。"""
        assert set(TERMINAL_JOB_STATES) == {"done", "failed"}
        # `queued` / `running` 一定不在里面
        assert "queued" not in TERMINAL_JOB_STATES
        assert "running" not in TERMINAL_JOB_STATES


class TestJobEvictionThroughTheRealEndpoint:
    """走真的 `POST /api/build`，验「提交路径接上了正确的判据」。

    只测 `_evict_finished_jobs_locked()` 是不够的：缺陷也可以表现为
    「函数写对了但提交处没调它」。
    """

    @staticmethod
    def _payload(n: int) -> dict:
        # base64 合法但**不是图片**：worker 线程会在解码那一步就失败，
        # 于是它绝不会去起 `build_scene.py` 子进程（那要 GPU）。
        return {"data": "Tk9UX0FOX0lNQUdF" * (n + 1), "filename": "x.png"}

    def test_submitting_past_the_cap_keeps_queued_jobs_alive(self, server):
        """★ 缺陷的原始复现：先摆满 8 个「排队中」，再提交第 9 个。

        修复前：最旧的那个（`queued_0`）被淘汰 → `/api/build/status` 对它回 404。
        """
        for i in range(serve_demo._JOB_KEEP):
            _put("queued_%d" % i, "queued")

        status, body = _request(server, "/api/build", method="POST",
                                body=self._payload(3))
        assert status == 200, body
        new_id = json.loads(body)["job_id"]

        for jid in ["queued_%d" % i for i in range(serve_demo._JOB_KEEP)] + [new_id]:
            assert jid in serve_demo._BUILD_JOBS, "%s 被淘汰了" % jid

    def test_a_finished_job_is_the_one_that_gets_dropped(self, server):
        """有已终结的任务时，该淘汰的是它，而不是排队中的那些。"""
        _put("queued_keep_me", "queued")
        _put("finished_drop_me", "done")
        for i in range(serve_demo._JOB_KEEP - 2):
            _put("filler_%d" % i, "done")

        _request(server, "/api/build", method="POST", body=self._payload(3))

        assert "queued_keep_me" in serve_demo._BUILD_JOBS
        assert "finished_drop_me" not in serve_demo._BUILD_JOBS

    def test_status_of_a_queued_job_still_answers(self, server):
        """收尾检查：排队中的任务仍然查得到（不是 404）。"""
        _put("queued_1", "queued")
        status, body = _request(server, "/api/build/status?id=queued_1")
        assert status == 200
        assert json.loads(body)["state"] == "queued"


# ---------------------------------------------------------------------------
# A4（服务端）：预算的取值与收紧规则
# ---------------------------------------------------------------------------


class TestAskBudget:
    def test_default_is_used_when_the_client_asks_for_nothing(self, monkeypatch):
        monkeypatch.delenv(ASK_BUDGET_ENV, raising=False)
        assert _ask_budget({}) == (ANSWER_BUDGET_S, None)

    def test_client_can_only_tighten(self, monkeypatch):
        """★ 客户端调大预算必须被**收紧**，并在响应里如实标出。"""
        monkeypatch.delenv(ASK_BUDGET_ENV, raising=False)
        effective, requested = _ask_budget({"budget_s": ANSWER_BUDGET_S * 100})
        assert effective == ANSWER_BUDGET_S
        assert requested == ANSWER_BUDGET_S * 100

    def test_client_can_tighten(self, monkeypatch):
        monkeypatch.delenv(ASK_BUDGET_ENV, raising=False)
        assert _ask_budget({"budget_s": 30}) == (30.0, 30.0)

    def test_non_numeric_budget_is_rejected(self):
        with pytest.raises(ValueError, match="budget_s"):
            _ask_budget({"budget_s": "soon"})

    def test_non_positive_budget_is_rejected(self):
        with pytest.raises(ValueError, match="budget_s"):
            _ask_budget({"budget_s": 0})
        with pytest.raises(ValueError, match="budget_s"):
            _ask_budget({"budget_s": -5})

    def test_empty_string_means_unspecified(self, monkeypatch):
        monkeypatch.delenv(ASK_BUDGET_ENV, raising=False)
        assert _ask_budget({"budget_s": ""}) == (ANSWER_BUDGET_S, None)

    def test_env_override_changes_the_default(self, monkeypatch):
        monkeypatch.setenv(ASK_BUDGET_ENV, "60")
        assert _ask_budget({})[0] == 60.0

    def test_invalid_env_override_falls_back_to_the_default(self, monkeypatch):
        """非法覆盖值不抛（服务起不来比预算不对更糟），但**必须落日志**。"""
        logged: list[str] = []
        monkeypatch.setattr(serve_demo, "_log", lambda line: logged.append(line))
        monkeypatch.setenv(ASK_BUDGET_ENV, "not-a-number")
        assert _ask_budget({})[0] == ANSWER_BUDGET_S
        assert logged and ASK_BUDGET_ENV in logged[0], "静默回退会让原因查不到"

    def test_non_positive_env_override_falls_back(self, monkeypatch):
        monkeypatch.setattr(serve_demo, "_log", lambda line: None)
        monkeypatch.setenv(ASK_BUDGET_ENV, "0")
        assert _ask_budget({})[0] == ANSWER_BUDGET_S

    def test_the_budget_sits_between_measured_p100_and_the_unbounded_worst(self):
        """把取值理由变成一条守卫 —— 改动这个数会立刻红。

        下界来源：`logs/agent_runs/` 里 54 条真实 run 的 `elapsed_s`，
        中位数 1.8 s / p90 3.9 s / 最大 24.3 s（2026-09-22 实测）。
        预算必须远大于实测最大值，否则正常题会被误杀。

        上界来源：无预算时最坏 = 4 次 chat × (3 × 180 + 6) = 2184 s。
        预算必须远小于它，否则这个参数没有意义。
        """
        measured_p100_s = 24.3
        unbounded_worst_s = 2184.0
        assert measured_p100_s * 4 < ANSWER_BUDGET_S < unbounded_worst_s

    def test_health_reports_the_budget(self, server, monkeypatch):
        """前端要能直接说出「这次最多等多久」，而不是让人去读源码。"""
        monkeypatch.delenv(ASK_BUDGET_ENV, raising=False)
        status, body = _request(server, "/api/health")
        assert status == 200
        payload = json.loads(body)
        assert payload["ask_budget_s"] == ANSWER_BUDGET_S
        assert payload["ask_budget_env"] == ASK_BUDGET_ENV

    def test_ask_rejects_a_bad_budget_with_400_before_taking_the_lock(self, server):
        """非法预算必须在**抢锁之前**被拒：否则先占住锁再回 400，自己制造一次等待。"""
        status, body = _request(server, "/api/ask", method="POST",
                                body={"scene_id": "s", "question": "q", "budget_s": 0})
        assert status == 400
        assert "budget_s" in json.loads(body)["error"]
        assert serve_demo._ASK_LOCK.locked() is False
