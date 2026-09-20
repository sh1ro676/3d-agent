#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scripts/serve_demo.py —— 演示后端：静态四区界面 + 真跑 Agent 的 HTTP 接口。

为什么需要它
============
静态导出的界面只能**回放**已经走过的问答。而答辩现场最有说服力的一幕是
「我现场问一句，你当场算」—— 这需要把浏览器接到真的 `AgentLoop` 上。

它刻意做得很薄：**没有 Web 框架、没有异步、没有数据库**，只用标准库 `http.server`。
而且装配一行也不自己写，全部走 `scripts/run_agent.py::build_session`
—— 于是「界面上能跑」与「实验里能跑」是同一套开关、同一个工具集、同一个模型。
这一点比省几行代码重要得多：装配若有两份，两份会漂移，而漂移的方向总是
「演示看起来更顺」。

端点
====
    GET  /                     → 四区界面（demo/index.html）
    GET  /api/health           → 后端与场景就绪状态（前端启动时探它）
    POST /api/ask              → 真跑一次问答，返回整条 run（答案 / trace / 校验 / 成本）
    POST /api/counterfactual   → 移走若干物体后重算关系（纯图操作，不花 API 钱）
    GET  /api/runs             → 历史批次（回放用）
    POST /api/build            → 上传一张图 → 起一个后台建图任务（立刻返回 job_id）
    GET  /api/build/status?id= → 该任务的阶段、真实日志尾部、结果摘要

上传建图那条路
==============
浏览器传 base64 图片，后端落盘到 `uploads/`，然后**调已有的 CLI 脚本**跑完
「建图 → 报告 → 导出」，一行感知逻辑都不在这里重写。两个刻意的设计：

1. **异步 + 轮询**，不是同步阻塞。第一次建图要加载三个模型，几十秒到几分钟；
   卡住的 POST 会让「在跑」与「挂了」无法区分，而轮询能把子进程的**真实输出**
   逐行显示出来（进度条需要知道总进度，而这里并不知道）。
2. **结果里带内参来源与标定状态**。上传的照片绝大多数没有标定内参，
   这时横向米制尺度不可信。一个只报「9 个物体」的结果卡片，
   会让人以为它和标定过的场景一样可信 —— 所以摘要里必须写清这一点。

安全与边界
==========
- **只监听 127.0.0.1**：这是本机演示工具，不是服务，不该出现在局域网里。
- **问答串行**（一把非阻塞锁）：并发跑 Agent 会让成本台账互相覆盖；
  抢锁失败直接回 409，前端显示「上一题还在跑」——比排队静默等待诚实。
- 反事实那条路**不装视觉**：它是纯图操作，为它构造一个 VLM 只会让「它是不是偷偷
  看了图」变成需要解释的问题。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.parse
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
for _p in (str(ROOT), str(SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agents.loop import AgentLoop  # noqa: E402
from llm.adapter import LLMClient, LLMSettings, load_backend_env  # noqa: E402
from run_agent import build_session  # noqa: E402
from tools.scene_report import counterfactual  # noqa: E402
from tools.version import TOOLS_VERSION  # noqa: E402

#: 被 `main()` 覆盖，供 Handler 使用（Handler 由 http.server 实例化，拿不到 args）。
STATIC_ROOT = ROOT / "demo"
ENV_FILE = ROOT / "configs" / "llm_backend.env"
LOG_FILE = ROOT / "logs" / "demo_server.log"

#: 问答锁 —— 见模块 docstring「安全与边界」。**非阻塞**获取，抢不到就如实回 409。
_ASK_LOCK = threading.Lock()

PLANNER_MODES = ("off", "on")
DEFAULT_ANSWER_TYPES = ("float", "int", "str", "bool")

#: ---- 上传建图 ----------------------------------------------------------------
#: 落盘目录。刻意放在仓库内的 `uploads/`，而不是系统临时目录 ——
#: 「这张场景图是从哪张图建的」必须事后可查，临时目录会被清理掉。
UPLOAD_DIR = ROOT / "uploads"
#: 单张上限。定这个数的理由是**内参按实际像素算**：先把图缩小再推理会改变
#: 焦距与主点，所以不做前端压缩，而是直接给一个够用的上限。
#: 16 MB 能容纳绝大多数直出 JPG/PNG，又不至于让 base64 后的请求体撑爆内存。
MAX_UPLOAD_MB = 16
ALLOWED_IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
#: 建图子进程的总超时。第一次跑要下载/加载权重，国内链路下 15 分钟是合理上界。
BUILD_TIMEOUT_S = 1500
#: 建图默认词表 —— 与 `vision/grounding.py::DEFAULT_PROMPT` 保持一致。
DEFAULT_BUILD_PROMPT = "sofa. chair. table. picture. mirror."
#: 任务登记表只保留最近几个。这是演示工具，不是作业队列。
_JOB_KEEP = 8

#: 建图串行锁。显存只有 8188 MiB，两个建图同时跑必然 OOM；
#: 而且「排队等」比「两个都崩」对演示更友好，所以这里用阻塞式获取。
_BUILD_LOCK = threading.Lock()
_BUILD_JOBS: dict[str, dict[str, Any]] = {}
_BUILD_ORDER: list[str] = []
_JOBS_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _log(line: str) -> None:
    """写文件而不是只 print。

    ⚠ 本机 PowerShell 的 stdout 不回传，只 print 等于没有日志 —— 服务起来之后
    「刚才那次请求到底报了什么」将无处可查。这是本项目反复踩到的一条。
    """
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(f"[{_now()}] {line}\n")


def _backend_state() -> dict[str, Any]:
    """后端配置是否就绪 —— 前端据此决定「能不能问」，而不是让用户点下去才发现没 key。

    ⚠ 两处都踩过坑，改之前先读完这段：

    ① **必须先 `load_backend_env`，不能直接 `from_env`。**
    `LLMSettings.from_env` 只读 `os.environ`，而 key 是 `build_session`
    在跑问答时**注入**进去的。所以不主动加载 env 文件时，这个只读探针的输出
    会**取决于进程历史**：服务刚起、还没人问过 → `ready=false / api_key=(empty)`；
    随便问过一次 → `ready=true / api_key=***7dc8`。同一个函数两次调用给出相反答案，
    而配置根本没变。这里主动加载，是为了跟 `/api/ask` **同源**
    （它也走 `build_session` → `load_backend_env`），把"进程历史"这个变量消掉。
    ⚠ 若只改这一处而不改 ②，等于把"未就绪"变成真的。

    ② **`ready` 取内层 `info["ready"]`，不是恒 true。**
    原先写死 `"ready": True`，它只表示"配置对象构造成功"，不看 key 有没有 ——
    而本函数的 docstring 恰恰承诺"前端据此决定能不能问"。文档写了、实现没做到。
    现在两个缺陷互相掩盖（前端 `app.js` 读的正是外层字段，而问答确实能用），
    **只修一条就会把现在能用的功能弄坏**：只把判据改成 `info["ready"]` 而不做 ①，
    服务刚起、还没人问过的那个瞬间前端会显示"未就绪"并**禁用「提问」按钮**。
    ⟹ ①② 必须同批改。
    """
    try:
        env_report = load_backend_env(str(ENV_FILE))
        settings = LLMSettings.from_env("text")
    except Exception as exc:                # noqa: BLE001
        return {"ready": False, "error": str(exc)}
    info = settings.describe()
    # 未就绪时给一句人能读的原因。原先启动打印写 `state.get("error")`，那只在
    # `from_env` **抛异常**时才有值；缺 key 是**不抛异常**的（`ready()` 返回 False 而已）
    # ⟹ 会打印出「未就绪：None」，看上去像 bug 而不像诊断。
    missing = [name for name, val in (
        ("api_key", settings.api_key),
        ("base_url", settings.base_url),
        ("model", settings.model),
    ) if not val]
    return {
        "ready": bool(info.get("ready")),
        "reason": ("缺少 " + "、".join(missing)) if missing else None,
        "model": info.get("model"),
        "base_url": info.get("base_url"),
        # 只投影诊断用得上的几个字段：`env_report["secrets"]` 虽然只存 fingerprint，
        # 但没必要把密钥相关的任何东西送进 HTTP 响应 —— 少送一份，少一个泄漏面。
        "env_file": env_report.get("path"),
        "env_exists": env_report.get("exists"),
        "env_error": env_report.get("error"),
        "env_conflicts": env_report.get("conflicts") or [],
        "env_empty_values": env_report.get("empty_values") or [],
        "detail": info,
    }


class DemoHandler(SimpleHTTPRequestHandler):
    server_version = "SpatialAgentDemo/1.0"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(STATIC_ROOT), **kwargs)

    # -- 基础设施 ------------------------------------------------------------

    def log_message(self, fmt: str, *args: Any) -> None:     # noqa: A003
        _log("%s %s" % (self.address_string(), fmt % args))

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("请求体不是合法 JSON：%s" % exc) from exc
        if not isinstance(data, dict):
            raise ValueError("请求体必须是 JSON 对象")
        return data

    def _guard(self) -> bool:
        """只接受本机来源。

        `--host 127.0.0.1` 已经限制了监听地址，这里再挡一道是因为**改了 --host
        就顺带把「谁能花钱调模型」也改掉了** —— 一个参数不该有这种副作用。
        """
        origin = self.client_address[0] if self.client_address else ""
        if origin not in ("127.0.0.1", "::1", "localhost"):
            self._json(403, {"ok": False, "error": "只接受本机请求"})
            return False
        return True

    # -- 路由 ----------------------------------------------------------------

    def do_GET(self) -> None:                       # noqa: N802
        if self.path.startswith("/api/health"):
            return self._api_health()
        if self.path.startswith("/api/build/status"):
            return self._api_build_status()
        if self.path.startswith("/api/runs"):
            return self._api_runs()
        if self.path in ("/", ""):
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self) -> None:                      # noqa: N802
        if not self._guard():
            return
        try:
            payload = self._read_json()
        except ValueError as exc:
            return self._json(400, {"ok": False, "error": str(exc)})
        if self.path.startswith("/api/ask"):
            return self._api_ask(payload)
        if self.path.startswith("/api/build"):
            return self._api_build(payload)
        if self.path.startswith("/api/counterfactual"):
            return self._api_counterfactual(payload)
        return self._json(404, {"ok": False, "error": "未知端点 " + self.path})

    # -- 端点实现 ------------------------------------------------------------

    def _api_health(self) -> None:
        try:
            index = json.loads((STATIC_ROOT / "data" / "index.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            index = {"scenes": []}
        self._json(200, {
            "ok": True,
            "tools_version": TOOLS_VERSION,
            "backend": _backend_state(),
            "scenes": [s.get("scene_id") for s in index.get("scenes") or []],
            "generated_at": index.get("generated_at"),
            "running": _ASK_LOCK.locked(),
        })

    def _api_runs(self) -> None:
        path = STATIC_ROOT / "data" / "runs.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return self._json(500, {"ok": False, "error": "读不到回放数据：%s" % exc})
        self._json(200, {"ok": True, **data})

    def _api_ask(self, payload: dict[str, Any]) -> None:
        scene_id = str(payload.get("scene_id") or "").strip()
        question = str(payload.get("question") or "").strip()
        if not scene_id or not question:
            return self._json(400, {"ok": False, "error": "scene_id 与 question 都不能为空"})

        planner = str(payload.get("planner") or "off").lower()
        if planner not in PLANNER_MODES:
            return self._json(400, {"ok": False, "error": "planner 只能是 off / on"})

        answer_type = payload.get("answer_type") or None
        if answer_type not in (None, *DEFAULT_ANSWER_TYPES):
            return self._json(400, {"ok": False, "error": "answer_type 取值非法"})
        use_vlm = bool(payload.get("vlm", True))

        if not _ASK_LOCK.acquire(blocking=False):
            return self._json(409, {"ok": False, "error": "上一题还在跑，等它结束再问"})

        try:
            _log("ask scene=%s planner=%s vlm=%s q=%s" % (scene_id, planner, use_vlm, question[:40]))
            session = build_session(
                scene_ref=scene_id, env_file=str(ENV_FILE), no_vlm=not use_vlm,
            )
            if not session.ok:
                return self._json(400, {"ok": False, "error": session.error})

            settings = LLMSettings.from_env("text")
            client = LLMClient(settings)
            try:
                client.check_ready()
            except Exception as exc:        # noqa: BLE001  缺 key 秒级失败，不重试
                return self._json(503, {"ok": False, "error": str(exc)})

            loop = AgentLoop(client, ctx=session.ctx, planner=planner, toolset=session.toolset)
            run = loop.run(question, answer_type=answer_type)
            try:
                saved = _persist_live_run(scene_id, session, loop, run, client)
            except Exception as exc:        # noqa: BLE001  落盘失败不该把答案吞掉
                saved = None
                _log("persist live run failed: %r" % (exc,))
            _log("ask done status=%s elapsed=%.2fs saved=%s" % (run.status, run.elapsed_s, saved))
            self._json(200, {
                "ok": True,
                "run": run.to_dict(),
                "switches": loop.switches(),
                "vlm": session.vlm_report,
                "scene_hint": session.scene_hint,
                "usage": client.usage.snapshot(),
                "backend": {"model": settings.describe().get("model")},
                "saved_as": saved,
            })
        except Exception as exc:            # noqa: BLE001  演示后端不该把栈丢给浏览器
            _log("ask failed: %r" % (exc,))
            self._json(500, {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)})
        finally:
            _ASK_LOCK.release()

    def _api_counterfactual(self, payload: dict[str, Any]) -> None:
        scene_id = str(payload.get("scene_id") or "").strip()
        remove = payload.get("remove") or []
        if not scene_id:
            return self._json(400, {"ok": False, "error": "scene_id 不能为空"})
        if not isinstance(remove, list) or not all(isinstance(x, str) for x in remove):
            return self._json(400, {"ok": False, "error": "remove 必须是字符串数组"})
        try:
            session = build_session(
                scene_ref=scene_id, env_file=str(ENV_FILE), no_vlm=True,
            )
            if not session.ok:
                return self._json(400, {"ok": False, "error": session.error})
            result = counterfactual(session.ctx, scene_id, remove)
            _log("counterfactual scene=%s remove=%s ok=%s" % (scene_id, remove, result.ok))
            status = 200 if result.ok else 400
            self._json(status, {"ok": bool(result.ok), "result": result.to_dict()})
        except Exception as exc:            # noqa: BLE001
            _log("counterfactual failed: %r" % (exc,))
            self._json(500, {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)})

    def _api_build(self, payload: dict[str, Any]) -> None:
        """接收一张图，起一个后台建图任务，**立刻返回 job id**。

        为什么不做成同步阻塞请求：建图第一次要加载三个模型，几十秒到几分钟。
        一个挂在那里的 POST 会让「服务是不是死了」和「还在跑」变得无法区分，
        而且中间没有任何进度信息。改成 job + 轮询之后，前端能把**子进程的真实
        输出**一行行显示出来 —— 那是这段等待里唯一诚实的信息。
        """
        data = payload.get("data")
        if not isinstance(data, str) or len(data) < 64:
            return self._json(400, {"ok": False, "error": "缺少图片数据（base64）"})
        # base64 长度 → 原始字节的粗略上界。先在这里挡一道，
        # 免得为了拒绝一个 200 MB 的文件而先把它读进内存又解码一遍。
        if len(data) > int(MAX_UPLOAD_MB * 1024 * 1024 * 1.4):
            return self._json(413, {"ok": False,
                                    "error": "图片超过 %d MB" % MAX_UPLOAD_MB})

        intrinsics = str(payload.get("intrinsics") or "").strip()
        prompt = str(payload.get("prompt") or "").strip() or DEFAULT_BUILD_PROMPT

        with _JOBS_LOCK:
            job_id = "b%d" % int(time.time() * 1000)
            job: dict[str, Any] = {
                "id": job_id, "state": "queued", "stage": "queued",
                "stage_text": "排队中…", "queued_at": time.time(), "started_at": None,
                "ended_at": None, "scene_id": None, "tail": [],
                "result": None, "error": None,
                "_spec": {"data": data,
                          "filename": str(payload.get("filename") or "upload.png"),
                          "scene_id": str(payload.get("scene_id") or ""),
                          "prompt": prompt,
                          "intrinsics": intrinsics},
            }
            _BUILD_JOBS[job_id] = job
            _BUILD_ORDER.append(job_id)
            while len(_BUILD_ORDER) > _JOB_KEEP:
                _BUILD_JOBS.pop(_BUILD_ORDER.pop(0), None)
        _log("build job %s queued (prompt=%r intrinsics=%r)"
             % (job_id, prompt, intrinsics or "<未提供>"))
        # 排队发生在**工作线程**里（见 `_run_build_job` 里的 `_BUILD_LOCK`），
        # 所以这里必须立刻返回：否则第二个请求会让一个 HTTP 连接陪着一起等，
        # 前端就分不清"在排队"和"这个接口挂了"。
        threading.Thread(target=_run_build_job, args=(job_id,), daemon=True).start()

        self._json(200, {"ok": True, "job_id": job_id, "stage": "queued"})

    def _api_build_status(self) -> None:
        query = urllib.parse.urlparse(self.path).query
        job_id = (urllib.parse.parse_qs(query).get("id") or [""])[0]
        with _JOBS_LOCK:
            job = _BUILD_JOBS.get(job_id)
            snap = _job_snapshot(job) if job else None
        if snap is None:
            return self._json(404, {"ok": False, "error": "没有这个任务：%s" % job_id})
        self._json(200, snap)


def _safe_scene_id(raw: str, fallback: str) -> str:
    """把用户给的名字洗成安全的 `scene_id`。

    它同时是**目录名**（`dataset/scenes/<id>/`）和**URL 片段**
    （`data/<id>.scene.json`），所以只允许 `[A-Za-z0-9_-]`。
    这不是洁癖：一个叫 `../configs` 的 scene_id 会直接把场景写到别处去。
    """
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", (raw or "").strip()).strip("_-")
    return (s[:48] or fallback)


def _existing_scene_ids() -> set[str]:
    root = ROOT / "dataset" / "scenes"
    if not root.is_dir():
        return set()
    return {p.parent.name for p in root.glob("*/scene.json")}


def _stream_proc(cmd: list[str], job: dict[str, Any], stage: str,
                 stage_text: str, timeout_s: float | None = None) -> int:
    """跑一个子进程并把它的输出逐行喂进 job 的 `tail`。

    ⚠ 为什么要逐行读、而不是 `subprocess.run(capture_output=True)`：
    建图第一次要加载三个模型（几十秒），如果整段吃掉，前端在那几十秒里
    只能转一个不知道在干什么的圈。逐行读让「正在加载模型 ...」这种真实输出
    实时出现在界面上 —— 一个看得见的真实日志，比一个编出来的进度条诚实得多
    （进度条需要知道总进度，而这里并不知道）。

    ★ 超时**不能**写在读循环里面。先前正是那么写的：

        for line in proc.stdout:                        # 阻塞在 readline 上
            ...
            if time.time() - t_stage > BUILD_TIMEOUT_S:  # ← 只有"有输出"才走得到
                proc.kill()

    子进程一旦**不输出**（真死锁 / CUDA 卡住 / 网络挂起），`readline` 永久阻塞，
    那个判断永远不执行 ⟹ `BUILD_TIMEOUT_S` 形同虚设，而 `_BUILD_LOCK` 会被
    **永久占用**，此后所有建图永远停在"排队中"，前端只看到时间一直涨。
    所以超时改由**一个独立的 daemon 计时线程**负责，到点直接 `proc.kill()`：
    杀掉之后 stdout 读到 EOF，上面的循环自然结束，再回头看是不是超时了。

    这与 `agents/executor.py::_Watchdog` 是同一个道理的两处落地 ——
    那边在**本进程内**向执行线程注入异常，这边是**跨进程**发信号。
    `timeout_s` 只是为了单测能把它调小；生产路径一律走 `BUILD_TIMEOUT_S`。
    """
    limit = float(BUILD_TIMEOUT_S if timeout_s is None else timeout_s)
    job["stage"] = stage
    job["stage_text"] = stage_text
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1, env=env,
    )
    # 超时按**本阶段**自己计时，不按任务开始时刻：`started_at` 在排队期间是 None，
    # 而且用任务总时长当限值会让"排了很久队"的请求在刚开始跑时就被判超时 ——
    # 那是把队列的等待算进了子进程的账上。Timer 在本阶段开始时才起，就是这个粒度。
    timed_out = threading.Event()

    def _kill_on_timeout() -> None:
        if proc.poll() is None:          # 已经自己退出了就别再动它
            timed_out.set()
            try:
                proc.kill()
            except OSError:              # 竞态：恰好在同一刻它自己结束了
                pass

    watchdog = threading.Timer(limit, _kill_on_timeout)
    watchdog.daemon = True
    watchdog.start()
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if line.strip():
                tail = job["tail"]
                tail.append(line)
                del tail[:-_JOB_KEEP * 3]
                job["stage_text"] = line.strip()[:180]
        rc = proc.wait()
        # 先 wait 再判超时：等进程真正回收掉，免得留一个僵尸。
        if timed_out.is_set():
            raise RuntimeError(
                "「%s」阶段超时（%.0f s，已强杀）。可能是首次运行在下载权重，"
                "也可能子进程卡死且长时间无输出 —— 看上方日志尾部到哪一步停住。"
                % (stage, limit))
        return rc
    finally:
        watchdog.cancel()
        if proc.poll() is None:
            proc.kill()


def _run_build_job(job_id: str) -> None:
    """后台线程主体：解码 → 建图 → 报告 → 导出。

    每一步都复用**已有的 CLI 脚本**（`build_scene.py` / `report_scene.py` /
    `export_demo.py`），一行感知逻辑都不在这里重写。理由与后端装配那件事相同：
    两份实现必然漂移，而漂移的方向总是「这条路上的数字更漂亮」。
    """
    with _JOBS_LOCK:
        job = _BUILD_JOBS[job_id]
        spec = job.pop("_spec")
    # 串行点在这里，不在提交处 —— 排队是**建图**的属性，不该让一个 HTTP 连接
    # 替它承担等待。拿不到锁的任务停在 queued 状态，前端如实显示"排队中"。
    # （用 acquire/release 而不是 `with`，是为了不动下面那一大段的缩进：
    #   这段 body 的每一行都是"为什么这么写"的注释，重排缩进等于把它们全部重打一遍。）
    _BUILD_LOCK.acquire()
    job["started_at"] = time.time()
    job["state"] = "running"

    try:
        # ---- 1) 解码与落盘 --------------------------------------------------
        job["stage"] = "decode"
        job["stage_text"] = "解码上传的图片…"
        raw = base64.b64decode(spec["data"], validate=False)
        if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
            raise ValueError("图片超过 %d MB" % MAX_UPLOAD_MB)

        # 用 Pillow 真打开一次才算数 —— 只看扩展名等于信任客户端。
        # 同时这一步拿到真实尺寸：内参按**实际参与推理的像素网格**算，
        # 所以尺寸必须在落盘前确认。
        from io import BytesIO
        from PIL import Image as _Image

        try:
            probe = _Image.open(BytesIO(raw))
            probe.verify()
            probe = _Image.open(BytesIO(raw))
            width, height = probe.size
            img_fmt = (probe.format or "").upper()
        except Exception as exc:                    # noqa: BLE001
            raise ValueError("这不是一张能解析的图片：%s" % exc) from exc

        ext = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "BMP": ".bmp"}.get(
            img_fmt, Path(spec["filename"]).suffix.lower() or ".png")
        if ext not in ALLOWED_IMAGE_EXT:
            raise ValueError("不支持的图片格式：%s" % (img_fmt or ext))

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scene_id = _safe_scene_id(spec.get("scene_id"), "upload_" + stamp)
        if scene_id in _existing_scene_ids():
            scene_id = "%s_%s" % (scene_id, stamp[-6:])
        # 固定叫 `rgb.<ext>`，放在以 scene_id 命名的子目录里。
        # 不用原始文件名（哪怕它更"保真"）：这个文件会被 export_demo 原样拷进
        # `demo/assets/<scene_id>/`，而前面几个场景那里都是一张 `rgb.png`。
        # 让同一类产物在同一个位置叫同一个名字，前端与脚本都不必为"这张图叫什么"
        # 写分支 —— 那点"保真"不值得换来一处特例。
        img_dir = UPLOAD_DIR / scene_id
        img_dir.mkdir(parents=True, exist_ok=True)
        img_path = img_dir / ("rgb" + ext)
        img_path.write_bytes(raw)
        job["scene_id"] = scene_id
        job["image"] = {"path": str(img_path), "w": width, "h": height,
                        "format": img_fmt, "bytes": len(raw)}
        job["tail"].append("图片 %.1f KB  %dx%d  %s → uploads/%s/%s"
                           % (len(raw) / 1024, width, height, img_fmt,
                              scene_id, img_path.name))

        # ---- 2) 建图（子进程，复用 CLI） ------------------------------------
        py = sys.executable
        cmd = [py, str(SCRIPTS / "build_scene.py"),
               "--image", str(img_path), "--scene-id", scene_id]
        if spec.get("prompt"):
            cmd += ["--prompt", spec["prompt"]]
        if spec.get("intrinsics"):
            cmd += ["--intrinsics", spec["intrinsics"]]
        rc = _stream_proc(cmd, job, "build",
                          "建图：加载 GroundingDINO / SAM2 / UniDepth…")
        if rc != 0:
            raise RuntimeError("build_scene.py 退出码 %s（看上方日志尾部）" % rc)

        scene_dir = ROOT / "dataset" / "scenes" / scene_id
        if not (scene_dir / "scene.json").is_file():
            raise RuntimeError("建图结束但没有产出 scene.json")

        # ---- 3) 场景报告（可选，失败不致命） --------------------------------
        rc = _stream_proc(
            [py, str(SCRIPTS / "report_scene.py"), "--scene", str(scene_dir)],
            job, "report", "生成场景报告…")
        job["report_ok"] = rc == 0

        # ---- 4) 增量导出到 demo/data 与 demo/assets -------------------------
        rc = _stream_proc([py, str(SCRIPTS / "export_demo.py")],
                          job, "export", "导出到演示台…")
        if rc != 0:
            raise RuntimeError("export_demo.py 退出码 %s" % rc)

        # ---- 5) 汇总结果 ----------------------------------------------------
        job["result"] = _scene_summary(scene_id)
        job["state"] = "done"
    except Exception as exc:                        # noqa: BLE001  失败要如实回给前端
        job["state"] = "failed"
        job["error"] = "%s: %s" % (type(exc).__name__, exc)
    finally:
        job["ended_at"] = time.time()
        try:
            _log("build job %s → %s (%s)" % (job_id, job["state"], job.get("scene_id")))
        finally:
            # 锁必须在 log 之后也一定释放：泄漏一次，后面所有建图永久排队。
            _BUILD_LOCK.release()


def _scene_summary(scene_id: str) -> dict[str, Any]:
    """读场景，给前端一份「这个场景到底可信到什么程度」的摘要。

    ★ 读的是**权威**场景文件 `dataset/scenes/<id>/scene.json`，
    **不是**导出后的 `demo/data/<id>.scene.json`。

    为什么这条要写清楚：先前读的是导出的那份，结果摘要里
    `intrinsics_source` 恒为 `null`、`fov.hfov_deg` 也是 `null` ——
    因为那份副本是 `export_demo` 按"前端要什么"裁剪过的。
    后端去读自己导出的文件，等于把「导出器挑了哪些字段」变成了
    「后端能知道什么」。而这里丢掉的恰恰是最该被看见的一项：
    **内参来源决定横向米制尺度可不可信**（模型预测 vs EXIF，实测差 118 倍）。
    两份文件长得像，所以这个错误不会报错，只会让"尺度未标定"这条提示
    少掉它唯一的具体证据。
    """
    out: dict[str, Any] = {
        "scene_id": scene_id,
        "scene_url": "data/%s.scene.json" % scene_id,
        "image_url": "assets/%s/rgb.png" % scene_id,
    }
    raw = ROOT / "dataset" / "scenes" / scene_id / "scene.json"
    try:
        data = json.loads(raw.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        out["error"] = "读不到权威场景文件 %s：%s" % (raw, exc)
        return out
    # 权威文件带信封（`_format` / `_written_at` / `scene`），导出副本是扁平的。
    scene = data.get("scene", data) if isinstance(data, dict) else {}
    meta = scene.get("build_meta") or {}
    fov = meta.get("fov") or {}
    # ⚠ `build_meta.image_hw` 是 **[高, 宽]**，而本模块对外一律用 **[宽, 高]**。
    #   这里不自己拆一遍，而是复用 `vision/semantics.image_size_from_meta`
    #   —— 那个函数的存在理由就是这个键名与顺序：弄反了不报错，
    #   只是宽高比静默颠倒。上一轮在 `tools/attributes.py` 已经踩过一次。
    from vision.semantics import image_size_from_meta

    wh = image_size_from_meta(meta)
    out.update({
        "n_nodes": len(scene.get("nodes") or []),
        "n_edges": len(scene.get("edges") or []),
        "image_hw": list(wh) if wh else None,
        "intrinsics_source": meta.get("intrinsics_source"),
        "intrinsics": meta.get("intrinsics"),
        "fov": {"hfov_deg": fov.get("hfov_deg"), "vfov_deg": fov.get("vfov_deg"),
                "plausible": fov.get("plausible"), "reason": fov.get("reason")},
        "scale_calibrated": bool(meta.get("scale_calibrated")),
        "up_axis": meta.get("up_axis"),
        "up_axis_tilt_deg": meta.get("up_axis_tilt_deg"),
        "up_axis_reliable": bool(meta.get("up_axis_reliable")),
        "up_axis_reason": meta.get("up_axis_reason"),
        "n_detections_raw": meta.get("n_detections_raw"),
        "n_detections_kept": meta.get("n_detections_kept"),
        "mask_box_coverage_min": meta.get("mask_box_coverage_min"),
        "prompt": (meta.get("config") or {}).get("prompt"),
        "nodes": [{"id": n.get("id"), "label": n.get("label"),
                   "n_points": n.get("n_points")} for n in (scene.get("nodes") or [])],
    })
    return out


def _job_snapshot(job: dict[str, Any]) -> dict[str, Any]:
    """给前端的任务快照。

    ★ 时间要报**两个**，而且总时长必须单调递增。

    为什么：先前只有一个 `elapsed_s`，它是用 `started_at` 算的，
    而 `started_at` 在拿到建图锁时会被重置（排队结束后）。于是实测出现了这样一幕：
    一个任务排队等了 33 秒，刚开始执行就显示"已用 0.9 秒" —— 数字往回跳。
    那不是显示问题，是**信息错了**：它在告诉用户"你只等了 0.9 秒"。

    所以拆成：
      elapsed_s  从提交那一刻算起（含排队），只会变大
      stage_s    当前这个子进程跑了多久（建图/报告/导出分别计时）
    """
    now = time.time()
    queued_at = job.get("queued_at") or job.get("started_at") or now
    started_at = job.get("started_at") or now
    end = job.get("ended_at") or now
    return {
        "ok": True,
        "id": job["id"],
        "state": job["state"],
        "stage": job.get("stage"),
        "stage_text": job.get("stage_text"),
        "elapsed_s": round(end - queued_at, 1),
        "stage_s": round(end - started_at, 1),
        "scene_id": job.get("scene_id"),
        "image": job.get("image"),
        "tail": list(job.get("tail") or []),
        "report_ok": job.get("report_ok"),
        "result": job.get("result"),
        "error": job.get("error"),
    }


def _persist_live_run(scene_id: str, session: Any, loop: Any, run: Any,
                      client: Any) -> str:
    """把现场问答也落盘 —— 与批量实验**同目录、同格式、可分辨**。

    为什么值得写：演示时问过什么、答得对不对，如果不落盘就永远查不到了。
    而"现场提问"恰恰是最容易出意外的环节（模型抽风、代理掉线、超时），
    事后想复盘却只有浏览器上闪过的一屏。写下来之后：

    - 文件名带 `demo_live_` 前缀，与批量批次同目录但可一眼分辨；
    - 结构照抄 `run_agent.py` 的批次文件，于是 `export_demo.py` 会**自动**
      把它收进回放列表 —— 现场问过的题，下次打开就在"回放"里躺着。
    """
    payload = {
        "scene_id": scene_id,
        "scene_hint": session.scene_hint,
        "switches": loop.switches(),
        "toolset": list(session.toolset),
        "tools_version": TOOLS_VERSION,
        "usage": client.usage.snapshot(),
        "vlm": session.vlm_report,
        "summary": {"source": "demo_live", "n_runs": 1,
                    "status": {run.status: 1}},
        "runs": [run.to_dict()],
    }
    runs_dir = ROOT / "logs" / "agent_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = runs_dir / f"demo_live_{stamp}_{scene_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return path.name


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="演示后端：静态四区界面 + 真跑 Agent 的 HTTP 接口（只监听本机）",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--static", default=str(ROOT / "demo"),
                    help="静态根目录（由 scripts/export_demo.py 生成）")
    ap.add_argument("--env-file", default=str(ROOT / "configs" / "llm_backend.env"))
    return ap


def main(argv: list[str] | None = None) -> int:
    global STATIC_ROOT, ENV_FILE
    args = build_parser().parse_args(argv)
    STATIC_ROOT = Path(args.static)
    ENV_FILE = Path(args.env_file)

    if not (STATIC_ROOT / "index.html").exists():
        print("没有找到 %s/index.html —— 先跑：python scripts/export_demo.py" % STATIC_ROOT)
        return 2

    try:
        httpd = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    except OSError as exc:
        print("端口 %d 起不来（可能被占用）：%s" % (args.port, exc))
        return 2

    state = _backend_state()
    _log("server start http://%s:%d backend_ready=%s" % (args.host, args.port, state.get("ready")))
    print("演示界面  http://%s:%d/" % (args.host, args.port))
    print("后端状态  %s" % ("就绪 " + str(state.get("model")) if state.get("ready")
                            else "未就绪：" + str(state.get("error") or state.get("reason") or "原因未知")))
    print("静态根    %s" % STATIC_ROOT)
    print("日志      %s" % LOG_FILE)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n停止。")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
