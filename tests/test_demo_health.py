"""`scripts/serve_demo.py::_backend_state()` 的单元测试 —— 零 GPU、零联网、零 LLM。

这组测试盯的是**一个自己跟自己打架的响应**（不报错，但两处判据相反）：

    `/api/health` 的外层 `backend.ready` 恒 `true`（只表示"配置对象构造成功"，不看 key），
    内层 `backend.detail.ready` 却取决于**进程历史** ——
    服务刚起、还没人问过问题时是 `false`，随便问过一次之后变成 `true`。

根因是 `LLMSettings.from_env` 只读 `os.environ`，而 key 是 `build_session`
在跑问答时**注入**进去的。⟹ 同一个只读探针，配置没变，两次调用给出相反答案。

真正危险的是**两个缺陷互相掩盖**：前端 `demo/app.js` 读的正是外层那个恒 `true` 的字段，
而问答确实能用，所以一切"看起来正常"。**只修一条就会把现在能用的功能弄坏** ——
只把判据改成内层 `ready` 而不主动加载 env 文件，服务刚起的那一瞬间前端会显示"未就绪"
并禁用「提问」按钮。所以两条判据必须**同批**改，且这里必须钉住它们**一致**。

用临时 env 文件而不是真配置：测试全程不碰真密钥，也不依赖开发机上的环境变量。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import serve_demo  # noqa: E402
from serve_demo import _backend_state  # noqa: E402

#: 任何一条被设上都会让"缺 key"这个场景测不出来。
_SECRET_ENV = (
    "SPATIAL_API_KEY", "SPATIAL_BASE_URL", "SPATIAL_MODEL", "SPATIAL_ENV_FILE",
)


@pytest.fixture()
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """把进程环境清成"什么都没配" —— 未就绪场景的前提。"""
    for name in _SECRET_ENV:
        monkeypatch.delenv(name, raising=False)


def _use_env_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, body: str) -> str:
    path = tmp_path / "llm_backend.env"
    path.write_text(body, encoding="utf-8")
    monkeypatch.setattr(serve_demo, "ENV_FILE", path)
    return "sk-test-plaintext-must-not-leak"


class TestReadyIsNotProcessHistory:
    """外层与内层必须一致，且不取决于"有没有人问过问题"。"""

    def test_ready_true_without_asking_first(self, monkeypatch, tmp_path, clean_env):
        """**修复前最典型的那个时刻**：干净进程、还没问过任何问题。

        修复前外层 true / 内层 false，自相矛盾；现在两者都该是 true。
        """
        _use_env_file(monkeypatch, tmp_path,
                      "SPATIAL_API_KEY=sk-test-plaintext-must-not-leak\n"
                      "SPATIAL_BASE_URL=https://example.invalid/v1\n"
                      "SPATIAL_MODEL=test-model\n")
        # 前提：进程环境里确实一个都没设（否则测不到"靠自加载"这条路径）
        assert os.environ.get("SPATIAL_API_KEY") is None

        state = _backend_state()

        assert state["ready"] is True
        assert state["detail"]["ready"] is True
        assert state["reason"] is None

    def test_two_consecutive_calls_agree(self, monkeypatch, tmp_path, clean_env):
        """配置没变，两次调用必须完全一致 —— 这条就是"历史依赖"的回归。"""
        _use_env_file(monkeypatch, tmp_path,
                      "SPATIAL_API_KEY=sk-test-plaintext-must-not-leak\n"
                      "SPATIAL_BASE_URL=https://example.invalid/v1\n"
                      "SPATIAL_MODEL=test-model\n")
        first, second = _backend_state(), _backend_state()
        assert first["ready"] == second["ready"] is True

    def test_outer_and_inner_agree_when_not_ready(self, monkeypatch, tmp_path, clean_env):
        """未就绪时两层也必须一致。

        修复前这里正是"外层 true / 内层 false" —— 会把未配置的后端谎报成可用。
        """
        _use_env_file(monkeypatch, tmp_path,
                      "SPATIAL_API_KEY=\n"
                      "SPATIAL_BASE_URL=https://example.invalid/v1\n"
                      "SPATIAL_MODEL=test-model\n")

        state = _backend_state()

        assert state["ready"] is False
        assert state["detail"]["ready"] is False
        # 原因必须是人读得懂的，不能是 None（启动打印曾输出「未就绪：None」）
        assert state["reason"], "未就绪时必须给出原因"
        assert "api_key" in state["reason"]


class TestFailurePathsAreVisible:
    """配置坏了要**看得见**，不能静默当成"没配"。"""

    def test_broken_env_file_reported_not_raised(self, monkeypatch, tmp_path, clean_env):
        """重复键（改配置忘了注释旧行）→ 由 `load_env_file` 抛错，这里必须接住并外显。

        静默把它当成"未配置"，会让人去查环境变量，而真正的问题在文件里。
        """
        _use_env_file(monkeypatch, tmp_path,
                      "SPATIAL_API_KEY=aaa\nSPATIAL_API_KEY=bbb\n")

        state = _backend_state()          # 不该抛

        assert state["ready"] is False
        assert state["error"], "配置错误必须带出来"
        assert "重复" in state["error"] or "SPATIAL_API_KEY" in state["error"]

    def test_no_key_plaintext_in_payload(self, monkeypatch, tmp_path, clean_env):
        """`/api/health` 是 HTTP 响应 —— 密钥只能以掩码出现。

        `_backend_state` 额外投影了 `env_report`，那几个字段必须都是脱敏的。
        """
        key = _use_env_file(monkeypatch, tmp_path,
                            "SPATIAL_API_KEY=sk-test-plaintext-must-not-leak\n"
                            "SPATIAL_BASE_URL=https://example.invalid/v1\n"
                            "SPATIAL_MODEL=test-model\n")

        payload = json.dumps(_backend_state(), ensure_ascii=False)

        assert key not in payload, "密钥明文进了 HTTP 载荷"
        assert "***" in payload, "掩码丢了 —— 前端就看不出到底配没配 key"
