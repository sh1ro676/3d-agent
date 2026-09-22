#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scripts/run_agent.py —— 自研 3D Spatial Agent 的命令行入口。

三种用法，成本从 0 到 几分钱，**按这个顺序用**：

    1) 看提示词（0 成本、不联网）—— 改完提示词先看这个
       python scripts/run_agent.py --scene living_room --question "..." --dry-run

    2) 跑一段手写程序（0 成本、不联网）—— 验执行器与工具库
       python scripts/run_agent.py --scene living_room --program-file my_prog.py

    3) 真跑一次问答（1 次 LLM 调用 ≈ 几千 token）
       python scripts/run_agent.py --scene living_room --question "哪把椅子离门最近？"

为什么 `--dry-run` 和 `--program-file` 值得做成一等公民
====================================================
调试一个「LLM + 沙箱 + 工具库」的循环时，最大的浪费是**把两类问题混在一起查**：
提示词写错了、还是执行器有 bug、还是模型不行？前两种各自都能在**不花钱**的前提下
单独验完。留下 `--dry-run` 是为了不让人为了看一眼 prompt 就发一次请求。

产物
====
`--out` 指定的 JSON（默认 `logs/agent_runs/<时间戳>_<场景>.json`）+ 同名 `.txt` 人读摘要。
两个都写，是因为本机 PowerShell 的 stdout 不回传，脚本必须自己落盘才有人看得见。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.executor import ALLOWED_MODULES, QA_TOOLSET, execute_program  # noqa: E402
from agents.loop import AgentLoop  # noqa: E402
from agents.prompts.system import (  # noqa: E402
    PROMPT_VERSION,
    build_system_prompt,
    prompt_fingerprint,
    scene_hint_for,
)
from agents.synthesizer import static_check  # noqa: E402
from llm.adapter import LLMClient, LLMSettings, load_backend_env  # noqa: E402
from llm.schema import docs_text  # noqa: E402
from llm.vlm import VLM  # noqa: E402
from scene_graph.store import load_scene, scene_dir  # noqa: E402
from tools.registry import ToolContext  # noqa: E402
from tools.version import TOOLS_VERSION  # noqa: E402

DEFAULT_OUT_DIR = ROOT / "logs" / "agent_runs"

#: 场景目录里可能出现的图像文件名（按优先级）。找不到就退化成"没有图"——
#: 后果是 `get_attributes` 返回 NOT_FOUND，而**不是**让整轮跑不起来：
#: 空间类题目（占绝大多数）根本不需要角色②。
_IMAGE_CANDIDATES = ("image.jpg", "image.jpeg", "image.png", "image.webp",
                     "rgb.jpg", "rgb.png", "input.jpg", "input.png")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="自研 3D Spatial Agent（程序合成范式）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--scene", default="living_room",
                    help="场景 id（读 dataset/scenes/<id>/scene.json）或 scene.json 的路径")
    ap.add_argument("--question", action="append", default=None,
                    help="可以给多次，每题独立跑一轮")
    ap.add_argument("--answer-type", action="append", default=None,
                    choices=["int", "float", "str", "bool"],
                    help="与 Omni3D-Bench 的 answer_type 对齐，便于与基线臂同表对比。"
                         "给一次 = 所有题都用它；给多次 = 按 --question 的顺序逐个配对")
    ap.add_argument("--program-file", default=None,
                    help="直接执行这个 .py（不调 LLM，零成本）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只渲染提示词与场景清单，不发请求（零成本）")
    ap.add_argument("--timeout", type=float, default=60.0, help="单次程序执行的硬超时（秒）")
    ap.add_argument("--max-retries", type=int, default=2, help="定向重新生成的上限（默认 2）")
    ap.add_argument("--render", default="template", choices=["template", "llm"],
                    help="答案渲染：模板（默认，零成本）或 LLM 润色（多一次调用）")
    ap.add_argument("--planner", default="off", choices=["off", "on"],
                    help="臂 G：planning-then-synthesis。on 时每题多一次 LLM 调用")
    ap.add_argument("--tools", default=None,
                    help="逗号分隔的动作空间覆盖（默认用 executor.QA_TOOLSET）")
    ap.add_argument("--image", default=None,
                    help="图像路径（默认在场景目录里自动找 image.jpg/png，或读场景的 build_meta）")
    ap.add_argument("--no-vlm", action="store_true",
                    help="★ 关闭视觉语义（角色②）：get_attributes 返回 CAPABILITY_DISABLED。"
                         "这是消融开关，不是「关掉一个工具」")
    ap.add_argument("--vision-model", default=None,
                    help="覆盖 SPATIAL_VISION_MODEL（不硬编码模型名的落地方式）")
    ap.add_argument("--vision-base-url", default=None, help="覆盖视觉端点 base_url")
    ap.add_argument("--questions-file", default=None,
                    help="批量题集 JSON：[{question, answer_type}, ...] 或字符串列表。"
                         "给了它就不用 --question")
    ap.add_argument("--env-file", default=None, help="后端配置文件（默认 configs/llm_backend.env）")
    ap.add_argument("--no-env-file", action="store_true", help="忽略后端配置文件，只用进程环境")
    ap.add_argument("--out", default=None, help="产物路径（默认 logs/agent_runs/<ts>_<scene>.json）")
    return ap


@dataclass
class Session:
    """一次问答所需的全部装配产物。**CLI 与演示后端共用同一份装配。**

    为什么值得单独抽出来：如果演示后端自己再写一遍装配顺序，那么「界面上跑的东西」
    与「实验里跑的东西」会慢慢分叉 —— 而分叉的表现通常是**演示看起来更顺**
    （没人会去比对两处开关的默认值）。这和「动作空间只能有一份」是同一条纪律。

    装配失败时 `error` 非空，其余字段是默认值：调用方据此回报，**不要**在这里打印或退出，
    因为两个调用方（命令行 / HTTP）的回报方式完全不同。
    """

    error: str | None = None
    error_hint: str = ""
    scene: Any = None
    scene_path: Path | None = None
    ctx: ToolContext | None = None
    toolset: tuple[str, ...] = ()
    scene_hint: dict[str, Any] = field(default_factory=dict)
    image: Path | None = None
    vlm_report: dict[str, Any] = field(default_factory=dict)
    env_report: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None


def build_session(*, scene_ref: str, env_file: str, no_env_file: bool = False,
                  no_vlm: bool = False, image_override: str | None = None,
                  tools: str | None = None, vision_model: str | None = None,
                  vision_base_url: str | None = None) -> Session:
    """把「配置 → 场景 → 上下文 → 动作空间 → 角色②」装配成一次会话。"""
    # ---- ① 后端配置 -------------------------------------------------------
    if no_env_file:
        env_report: dict[str, Any] = {"path": "", "exists": False, "skipped": True}
    else:
        try:
            env_report = load_backend_env(env_file)
        except Exception as exc:            # noqa: BLE001  配置文件写坏 = 整轮停住
            return Session(error="配置文件解析失败：%s" % exc,
                           error_hint="一个写坏的配置文件如果被容错跳过，实验会拿默认值跑完。",
                           env_report={"path": env_file, "exists": True, "skipped": False})

    # ---- ② 场景与上下文 ----------------------------------------------------
    scene_path = Path(scene_ref)
    if not scene_path.exists():
        scene_path = scene_dir(scene_ref)
    try:
        scene = load_scene(scene_path)
    except Exception as exc:                # noqa: BLE001
        return Session(error="读不到场景 %s：%s" % (scene_path, exc),
                       error_hint="先跑 scripts/build_scene.py 建场景图。",
                       env_report=env_report)

    ctx = ToolContext(scene=scene, record_trace=True)
    toolset = tuple(t.strip() for t in tools.split(",")) if tools else QA_TOOLSET
    hint = scene_hint_for(scene)

    # ---- ③ 角色②（视觉语义）：装配或如实关闭 --------------------------------
    # 关掉视觉**不是**把工具从动作空间里删掉 —— 那样模型不知道有这条路，
    # 「关掉视觉」就变成了「换了一个动作空间」，消融不再干净（见 §13.3(7)）。
    # 正确做法是让工具在运行期返回 CAPABILITY_DISABLED。
    vlm = None
    img: Path | None = None
    if no_vlm:
        vlm_report: dict[str, Any] = {"enabled": False, "reason": "--no-vlm（消融开关）"}
    else:
        _apply_vision_overrides(vision_model, vision_base_url)
        try:
            img, image_src = _find_image(scene, scene_path, image_override)
        except Exception as exc:            # noqa: BLE001
            img, image_src = None, "找图失败：%s" % exc

        if img is None:
            vlm_report = {"enabled": False, "reason": image_src + "；只有属性类题目需要它"}
        else:
            try:
                vlm = VLM()
            except Exception as exc:        # noqa: BLE001
                vlm = None
                vlm_report = {"enabled": False, "reason": "视觉端点组装失败：%s" % exc}
            else:
                vlm_report = {"enabled": True, "image": str(img),
                              "image_source": image_src, **vlm.describe_dict()}
    ctx.vlm = vlm
    if img is not None:
        ctx.images[str(getattr(scene, "image_id", "") or "image")] = img

    return Session(scene=scene, scene_path=scene_path, ctx=ctx, toolset=toolset,
                   scene_hint=hint, image=img, vlm_report=vlm_report, env_report=env_report)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # ---- ①③ 装配（配置 / 场景 / 上下文 / 动作空间 / 角色②）------------------
    # 这一段与演示后端共用 `build_session()` —— 两处各写一遍会让「界面里能跑」
    # 和「实验里能跑」悄悄分叉。
    session = build_session(
        scene_ref=args.scene, env_file=args.env_file, no_env_file=args.no_env_file,
        no_vlm=args.no_vlm, image_override=args.image, tools=args.tools,
        vision_model=args.vision_model, vision_base_url=args.vision_base_url,
    )
    if not session.ok:
        _emit(args, {"fatal": session.error, "hint": session.error_hint})
        return 2

    scene = session.scene
    ctx = session.ctx
    toolset = session.toolset
    hint = session.scene_hint
    vlm_report = session.vlm_report
    env_report = session.env_report

    # ---- ③ 零成本路径 -----------------------------------------------------
    if args.dry_run:
        # 模块白名单只在 system prompt 的规则 5 里出现一次（工具文档里不再重复），
        # 所以这里**必须**用 ALLOWED_MODULES —— 干跑的 prompt 要和真跑的逐字节一致，
        # 否则「我明明看过提示词了」这句话就失去了意义。
        docs = docs_text(tools=toolset,
                         heading="（返回类型统一是 ToolResult：先判 res.ok 再读 res.value）")
        payload = {
            "mode": "dry_run",
            "scene": scene.summary_line(),
            "scene_hint": hint,
            "tools_version": TOOLS_VERSION,
            "prompt_version": PROMPT_VERSION,
            "prompt_fingerprint": prompt_fingerprint(),
            "toolset": list(toolset),
            "tool_docs_chars": len(docs),
            "system_prompt_chars": len(build_system_prompt(docs, ALLOWED_MODULES)),
            "questions": _questions_of(args) or [],
            "planner": args.planner,
            "vlm": vlm_report,
            "llm_env": env_report,
            "prompt_preview": _prompt_preview(args, hint, docs, toolset),
        }
        if args.planner == "on":
            payload["plan_prompt_preview"] = _plan_prompt_preview(args, hint, docs, toolset)
        _emit(args, payload)
        return 0

    if args.program_file:
        source = Path(args.program_file).read_text(encoding="utf-8")
        check = static_check(source, tools=toolset)
        outcome = execute_program(
            source, ctx, answer_type=(args.answer_type or [None])[0],
            timeout_s=args.timeout, source_label=str(args.program_file),
            toolset=toolset,
        )
        payload = {
            "mode": "program_file",
            "scene": scene.summary_line(),
            "scene_hint": hint,
            "tools_version": TOOLS_VERSION,
            "static_check": check.to_dict(),
            "execution": outcome.to_dict(),
            "usage": {"calls": 0, "cost_cny": 0.0},
            "llm_env": env_report,
        }
        _emit(args, payload)
        return 0 if (check.ok and outcome.ok) else 1

    # ---- ④ 真跑 -----------------------------------------------------------
    questions = _questions_of(args)
    if not questions:
        _emit(args, {"fatal": "没有给题目。用 --question 或 --questions-file，"
                              "或用 --dry-run / --program-file（零成本）。"})
        return 2

    try:
        settings = LLMSettings.from_env("text")
    except ValueError as exc:
        _emit(args, {"fatal": "后端配置有错：%s" % exc})
        return 2
    client = LLMClient(settings)
    try:
        client.check_ready()
    except Exception as exc:                # noqa: BLE001  缺 key 秒级失败，不重试
        _emit(args, {"fatal": str(exc), "llm_env": env_report})
        return 2

    try:
        types = _resolve_answer_types(questions, args.answer_type)
    except SystemExit as exc:
        _emit(args, {"fatal": str(exc)})
        return 2

    loop = AgentLoop(client, ctx=ctx, max_synthesis_retries=args.max_retries,
                     exec_timeout_s=args.timeout, render=args.render,
                     planner=args.planner, toolset=toolset)
    runs = []
    for question, atype in zip(questions, types):
        run = loop.run(question, answer_type=atype)
        runs.append(run.to_dict())

    payload = {
        "mode": "llm",
        "arm": "agent-program-synthesis",
        "scene": scene.summary_line(),
        "scene_id": scene.scene_id,
        "scene_hint": hint,
        "tools_version": TOOLS_VERSION,
        # ⚠ 真跑记录里此前**一个提示词字段都没有**（连长度都没有）——
        # 于是"这一轮发的是哪版提示词"只能靠人记。补上（2026-09-22）：
        # 版本号给人读，指纹给机器比；两者缺一，"两次实验的提示词是不是同一份"就答不了。
        "prompt_version": PROMPT_VERSION,
        "prompt_fingerprint": prompt_fingerprint(),
        "toolset": list(toolset),
        # 开关组合进了产物 —— 消融表按它分组，而不是靠人记住跑的是哪一版。
        "switches": loop.switches(),
        "vlm": vlm_report,
        "backend": settings.describe(),
        "llm_env": env_report,
        "usage": client.usage.snapshot(),
        "summary": _summarize(runs),
        "runs": runs,
    }
    _emit(args, payload)
    # 退出码只反映「循环有没有正常交出答案」，**不反映答得对不对**
    # —— 对错要真值，而真值不在这一层（见 agents/verifier.py 的说明）。
    return 0 if all(r["status"] in ("ok", "abstained") for r in runs) else 1


def _questions_of(args: argparse.Namespace) -> list[str]:
    """题目来自 `--questions-file` 或（可重复的）`--question`。两者都给 → 报错。

    不合并是因为合并的**顺序**会成为隐式约定，而隐式约定正是不可复现的起点。
    """
    if args.questions_file:
        if args.question:
            raise SystemExit("--question 与 --questions-file 只能给一个。")
        raw = json.loads(Path(args.questions_file).read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw = raw.get("questions") or []
        out: list[str] = []
        for item in raw:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict) and item.get("question"):
                out.append(str(item["question"]))
            else:
                raise SystemExit("题集里有一项既不是字符串也没有 question 字段：%r" % (item,))
        return out
    return list(args.question or [])


def _summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """跨题的汇总 —— **只统计可数的东西**，不做任何"算对没算对"的判断。

    真值不在这里（那属于 `evaluation/`），所以这里给的是：
    状态分布、证据校验结论分布、重试次数、工具调用、成本。
    其中 `verdict_unsupported` 是最值钱的一列：它是「答案没有证据支持」的题数，
    与正确率**无关** —— 一道题可以答错但证据齐全，也可以答对却纯属巧合。
    """
    def tally(key: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in runs:
            v = str(r.get(key) or "")
            counts[v] = counts.get(v, 0) + 1
        return dict(sorted(counts.items()))

    verdicts: dict[str, int] = {}
    for r in runs:
        lv = str((r.get("verdict") or {}).get("level") or "")
        verdicts[lv] = verdicts.get(lv, 0) + 1

    return {
        "n": len(runs),
        "status": tally("status"),
        "verdict": dict(sorted(verdicts.items())),
        "n_attempts_gt1": sum(1 for r in runs if int(r.get("attempts") or 0) > 1),
        "n_failure": sum(1 for r in runs if r.get("failure")),
        "tool_calls_total": sum(int(r.get("tool_calls") or 0) for r in runs),
        "cost_cny": round(sum(float((r.get("usage") or {}).get("cost_cny") or 0.0)
                              for r in runs), 6),
    }


def _find_image(scene: Any, scene_path: Path, override: str | None) -> tuple[Path | None, str]:
    """找出这次要喂给角色②的图像，返回 `(路径, 来源说明)`。

    四级回退，**来源说明必须跟着走** —— 「图是从哪来的」决定了这次结果能不能
    与别的批次比：`build_meta.image_path`（新场景，正经来源）与
    `build_log.txt`（旧场景的兼容路径）虽然常常指向同一个文件，但前者是契约、
    后者是解析日志，性质不同，产物里必须看得出来。

        ① `--image`                      显式指定
        ② `meta["image_path"]`           新场景：builder 写进 build_meta 的（推荐）
        ③ 场景目录里的常见文件名           手工摆进去的图
        ④ 解析 `build_log.txt` 的 image=  旧场景的兼容路径（会标注）

    **找不到就返回 None**（而不是抛错）：大多数题目不需要看颜色，为了一张图
    让整轮选题跑不起来是不划算的；找不到的后果由 `get_attributes` 如实报 `NOT_FOUND`。
    """
    if override:
        p = Path(override)
        if not p.is_file():
            raise FileNotFoundError("--image 指向的文件不存在：%s" % p)
        return p, "--image（显式指定）"

    base = Path(scene_path)
    if base.is_file():
        base = base.parent

    meta = getattr(scene, "build_meta", None) or {}
    for key in ("image_path", "source_image", "image_file"):
        v = meta.get(key)
        if isinstance(v, str) and v:
            cand = Path(v)
            if not cand.is_absolute():
                cand = base / cand
            if cand.is_file():
                return cand, "build_meta.%s" % key

    for name in _IMAGE_CANDIDATES:
        cand = base / name
        if cand.is_file():
            return cand, "场景目录文件名命中（%s）" % name
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        hits = sorted(base.glob(pattern))
        if hits:
            return hits[0], "场景目录里唯一的图片（%s）" % hits[0].name

    # ④ 兼容旧场景：build_log.txt 的 `image=` 行。
    #    刻意放在最后、且带标注 —— 解析人读日志当接口用是权宜之计，
    #    重建场景（build_scene.py 已把 image_path 写进 build_meta）后就不会再走这条路。
    log = base / "build_log.txt"
    if log.is_file():
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("image="):
                cand = Path(line[len("image="):].strip())
                if cand.is_file():
                    return cand, "build_log.txt（旧场景兼容路径，重建后会变成 build_meta.image_path）"

    return None, "未找到图像"


def _apply_vision_overrides(vision_model: str | None, vision_base_url: str | None) -> None:
    """把 `--vision-*` 写进进程环境。

    写环境变量而不是直接改 `LLMSettings`，是为了让**只有一处**决定视觉端点长什么样
    （`LLMSettings.from_env("vision")`）。两处决定 = 两处会漂移，
    而漂移的表现是"报告里写的模型和实际用的不一致" —— 那种错误查起来最费时间。
    """
    import os

    if vision_model:
        os.environ["SPATIAL_VISION_MODEL"] = vision_model
    if vision_base_url:
        os.environ["SPATIAL_VISION_BASE_URL"] = vision_base_url


def _prompt_preview(args: argparse.Namespace, hint: dict[str, Any],
                    docs: str, toolset: tuple[str, ...]) -> dict[str, Any]:
    from agents.prompts.system import build_user_prompt

    q = _questions_of(args) or ["（没有给题目）"]
    atype = (args.answer_type or [None])[0]
    return {
        "system": build_system_prompt(docs, ALLOWED_MODULES),
        "user": build_user_prompt(q[0], hint, answer_type=atype),
        # 臂 G 开着时，真跑用的 user 里会多一段计划块。这里给一份**示例**，
        # 让「开计划到底改了提示词的哪几行」在零成本下就能看见。
        "user_with_plan": build_user_prompt(
            q[0], hint, answer_type=atype,
            plan_text="需要的工具类别：list_objects、calculate_distance\n步骤：\n  1. 读场景，找到门\n"),
    }


def _plan_prompt_preview(args: argparse.Namespace, hint: dict[str, Any],
                         docs: str, toolset: tuple[str, ...]) -> dict[str, Any]:
    """臂 G 那一次调用的提示词（零成本可看）。"""
    from agents.prompts.system import build_plan_system_prompt, build_plan_user_prompt

    q = _questions_of(args) or ["（没有给题目）"]
    return {
        "system": build_plan_system_prompt(docs, toolset),
        "user": build_plan_user_prompt(
            q[0], hint, answer_type=(args.answer_type or [None])[0]),
    }


def _resolve_answer_types(questions: list[str] | None,
                          answer_types: list[str] | None) -> list[str | None]:
    """把 `--answer-type` 配到每一题上。

    只给一次 = 所有题都用它（最常见）；给多次 = 按顺序逐个配对。
    个数对不上时**报错而不是补齐** —— 悄悄用同一个类型跑完整个题集，
    会让「int 题被当成 float 判」这种口径错误在报告里看不出来。
    """
    questions = questions or []
    types = answer_types or []
    if not types:
        return [None] * len(questions)
    if len(types) == 1:
        return list(types) * len(questions)
    if len(types) != len(questions):
        raise SystemExit("--answer-type 给了 %d 个，但 --question 有 %d 个："
                         "要么只给一个（全部同型），要么一一对应。"
                         % (len(types), len(questions)))
    return list(types)


def _short(value: Any, limit: int = 160) -> str:
    """把 stage 的 detail 压成一行。

    `detail` 有时是字符串、有时是 dict（例如 render 阶段的 `render_report`）——
    直接切片会在 dict 上抛 KeyError，而且**只在这个人读摘要里炸**，
    JSON 产物其实已经写好了。所以这里统一转字符串。
    """
    if isinstance(value, str):
        text = value
    elif value is None:
        text = ""
    else:
        text = json.dumps(value, ensure_ascii=False)
    return text[:limit]


def _slug(name: str) -> str:
    """把 `--scene` 变成安全的文件名片段。

    `--scene` 既可以是 id 也可以是一个路径，直接当文件名会在带路径时
    试图创建多层目录（而且 Windows 上 `D:` 里的冒号还是非法字符）。
    """
    stem = Path(str(name or "scene")).stem or "scene"
    return "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in stem)


def _emit(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    """落盘 JSON + 人读 txt，并往 stdout 打一句摘要。

    stdout 在本机不回传，所以**所有给人看的东西都必须落盘**。
    """
    out = Path(args.out) if args.out else (
        DEFAULT_OUT_DIR / ("%s_%s.json" % (time.strftime("%Y%m%d_%H%M%S"), _slug(args.scene)))
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    text = out.with_suffix(".txt")
    text.write_text(_human(payload), encoding="utf-8")
    print("[agent] %s" % out)
    print("[agent] %s" % text)


def _human(payload: dict[str, Any]) -> str:
    if "fatal" in payload:
        return "失败：%s\n%s" % (payload["fatal"], payload.get("hint", ""))
    out: list[str] = ["=" * 78, "自研 3D Spatial Agent —— %s" % payload.get("mode"), "=" * 78]
    out.append("场景：%s" % payload.get("scene"))
    if "scene_hint" in payload:
        out.append("场景清单（scene_hint，无坐标）：%s"
                   % json.dumps(payload["scene_hint"].get("objects"), ensure_ascii=False))
    if "toolset" in payload:
        out.append("动作空间（%d 个）：%s" % (len(payload["toolset"]), ", ".join(payload["toolset"])))
    if "switches" in payload:
        sw = payload["switches"]
        out.append("开关：action_space=%s  planner=%s  render=%s  vlm=%s(%s)"
                   % (sw.get("action_space"), sw.get("planner"), sw.get("render"),
                      sw.get("vlm"), sw.get("vlm_model")))
    if "vlm" in payload:
        v = payload["vlm"]
        out.append("角色②（视觉语义）：%s   %s"
                   % ("启用" if v.get("enabled") else "关闭", v.get("reason", v.get("model", ""))))
    if "static_check" in payload:
        sc = payload["static_check"]
        out.append("\n-- 静态检查 --")
        out.append("  通过：%s   工具调用 %d 次   有 submit：%s"
                   % (sc["ok"], sc["n_tool_calls"], sc["has_submit"]))
        for e in sc["errors"]:
            out.append("  ! %s" % e)
    if "execution" in payload:
        ex = payload["execution"]
        out.append("\n-- 执行 --")
        out.append("  通过：%s   阶段：%s   工具调用：%d   耗时：%.1f ms"
                   % (ex["ok"], ex["stage"], ex["tool_calls"], ex["duration_ms"]))
        if ex["message"]:
            out.append("  消息：%s" % ex["message"])
        if ex["submission"]:
            s = ex["submission"]
            out.append("  答案：%r（%s）  弃答：%s" % (s["answer"], s["answer_type"], s["abstained"]))
            for ev in s["evidence"]:
                out.append("  证据：%s" % ev)
        if ex["trace_errors"]:
            out.append("  trace 错误码：%s" % json.dumps(ex["trace_errors"], ensure_ascii=False))
    if "prompt_preview" in payload:
        out.append("\n" + "=" * 78 + "\nSYSTEM PROMPT\n" + "=" * 78)
        out.append(payload["prompt_preview"]["system"])
        out.append("\n" + "=" * 78 + "\nUSER PROMPT\n" + "=" * 78)
        out.append(payload["prompt_preview"]["user"])
    for run in payload.get("runs", []):
        out.append("\n" + "=" * 78)
        out.append("问题：%s" % run["question"])
        out.append("状态：%s   答案：%r   渲染：%s"
                   % (run["status"], run["answer"], run["render"]))
        vd = run.get("verdict") or {}
        if vd:
            out.append("证据校验：%s（命中出处：%s）"
                       % (vd.get("level"), vd.get("matched_from") or "-"))
            for c in vd.get("checks", []):
                out.append("   %s %s — %s"
                           % ("OK " if c["ok"] else "!! ", c["name"], c["detail"]))
        out.append("生成次数：%d   工具调用：%d   耗时：%.1f s   成本：¥%.4f"
                   % (run["attempts"], run["tool_calls"], run["elapsed_s"],
                      float(run["usage"].get("cost_cny") or 0.0)))
        pl = run.get("plan") or {}
        if pl.get("steps"):
            out.append("-- 前置计划（臂 G，%d 步）--" % pl.get("n_steps", 0))
            out.append(pl.get("text", ""))
            if pl.get("numbers_in_plan"):
                out.append("   ⚠ 计划里出现了数字：%s（会被记入产物，见 planner 的说明）"
                           % pl["numbers_in_plan"])
            if pl.get("unknown_categories"):
                out.append("   ⚠ 计划里列了不在动作空间的类别：%s" % pl["unknown_categories"])
        if run["failure"]:
            out.append("失败：%s" % json.dumps(run["failure"], ensure_ascii=False))
        for st in run["stages"]:
            out.append("  [%s] attempt=%s ok=%s %s"
                       % (st["stage"], st.get("attempt"), st.get("ok"),
                          _short(st.get("detail", st.get("message", "")))))
        if run["program"]:
            out.append("-- 生成的程序 --")
            out.append(run["program"])
    if "summary" in payload:
        s = payload["summary"]
        out.append("\n" + "=" * 78)
        out.append("汇总：题数 %d ｜ 状态 %s ｜ 证据校验 %s"
                   % (s["n"], json.dumps(s["status"], ensure_ascii=False),
                      json.dumps(s["verdict"], ensure_ascii=False)))
        out.append("      重试过的题 %d ｜ 失败题 %d ｜ 工具调用合计 %d ｜ 成本 ¥%.4f"
                   % (s["n_attempts_gt1"], s["n_failure"], s["tool_calls_total"], s["cost_cny"]))
    if "usage" in payload and payload["mode"] == "llm":
        out.append("\n" + "=" * 78)
        out.append("总用量：%s" % json.dumps(payload["usage"], ensure_ascii=False))
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
