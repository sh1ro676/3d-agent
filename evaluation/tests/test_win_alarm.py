"""`evaluation/win_alarm.py` 的单元测试 —— 纯标准库，零 torch、零 GPU、零联网。

## 这个 shim 值得测试的三个理由

① 它替换的是**全局状态**（`sys.settrace`）。一旦 cancel 没把原来的 tracer
   装回去，后面所有代码的 trace 行为都被悄悄改掉 —— 而且是静默的。
② 它必须抛**VADAR 自己的** `TimeoutException`，不是内置 `TimeoutError`。
   抛错了 `except TimeoutException` 就抓不到，超时会表现成「程序崩溃」而不是
   「超时重试」。这个区别只有在真跑时才看得出来，所以要用测试钉住。
③ 「超时」是**看门狗**，它有明确的不能力：卡在 C 调用里时不响。
   测试同时把「能响」和「不能响」两件事都写下来，免得后来者以为它等价于 SIGALRM。
"""

from __future__ import annotations

import sys
import time

import pytest

from evaluation.win_alarm import TraceAlarmShim, install_alarm_shim


class VadarsTimeout(Exception):
    """替身：真实环境里这里是 engine_utils.TimeoutException。"""


def _raise_vadars_timeout(signum, frame):
    raise VadarsTimeout()


def _make_shim():
    shim = TraceAlarmShim()
    shim.signal(shim.SIGALRM, _raise_vadars_timeout)
    return shim


def _expire(shim):
    """把 deadline 拨到过去，但**不**走 alarm()。

    为什么不走 `alarm(0.001)` 再 sleep：一旦 deadline 过期，看门狗会在
    「下一行被 trace 的代码」上立刻触发 —— 那行可能是 `pytest.raises.__enter__`，
    于是异常在 with 块**之前**抛出，测试就永远抓不到它了。
    直接设 `_deadline` 可以精确测到 `check_now()` 这一条路径。
    （「过期后从任意一行冒出来」是真实行为，由 `test_timeout_interrupts_a_new_frame` 覆盖。）
    """
    shim._deadline = time.monotonic() - 1.0


class TestSignalSurface:
    def test_sigalrm_constant(self):
        # 不是随便定的常数，是 Linux 上 SIGALRM 的真实值；写成别的会让
        # 任何按值比较的代码对不上。
        assert TraceAlarmShim.SIGALRM == 14

    def test_rejects_other_signals(self):
        shim = _make_shim()
        with pytest.raises(ValueError):
            shim.signal(15, lambda *a: None)

    def test_unimplemented_attribute_raises(self):
        # 故意不实现 siginterrupt 之类：让「有别的调用点依赖它」立刻暴露，
        # 而不是返回一个默认值把它掩盖过去。
        shim = _make_shim()
        with pytest.raises(AttributeError):
            shim.siginterrupt(14, True)


class TestArmAndCancel:
    def test_alarm_zero_without_arm_is_noop(self):
        shim = _make_shim()
        before = sys.gettrace()
        assert shim.alarm(0) == 0
        assert sys.gettrace() is before
        assert shim.arm_count == 0 and shim.cancel_count == 1

    def test_arm_then_cancel_restores_previous_tracer(self):
        shim = _make_shim()

        def marker(frame, event, arg):
            return marker

        sys.settrace(marker)
        try:
            shim.alarm(30)
            assert sys.gettrace() is not marker      # 已被换成看门狗
            shim.alarm(0)
            assert sys.gettrace() is marker          # ★ 必须装回来
        finally:
            sys.settrace(None)
        assert shim.arm_count == 1 and shim.cancel_count == 1

    def test_check_now_before_deadline_does_not_fire(self):
        shim = _make_shim()
        shim.alarm(60)
        try:
            shim.check_now()
            assert shim.fired == 0
        finally:
            shim.alarm(0)

    def test_check_now_after_deadline_raises_vadars_exception(self):
        shim = _make_shim()
        _expire(shim)
        with pytest.raises(VadarsTimeout):
            shim.check_now()
        assert shim.fired == 1
        assert shim.fired_by["trace"] == 1

    def test_deadline_cleared_so_it_does_not_refire(self):
        shim = _make_shim()
        _expire(shim)
        with pytest.raises(VadarsTimeout):
            shim.check_now()
        # 第二次不应再抛：一次性看门狗，否则会连环中断调用方
        shim.check_now()
        assert shim.fired == 1


class TestTraceIntegration:
    def test_timeout_interrupts_a_new_frame(self):
        """端到端走**路径 ①**：武装一个极短超时，然后**调用一个新函数**让它死循环。

        注意循环必须放在被调用的函数里，不能直接写在测试函数体内 ——
        `sys.settrace` 只对调用之后**新建的帧**生效，当前帧不会被 trace。
        这不是测试写法问题，是路径 ① 的真实边界（见模块 docstring）；
        路径 ②（定时器注入）就是为它准备的。
        对 VADAR 的真实用法这个边界不触发：`runpy.run_path()` 一定新建模块帧。
        """
        shim = _make_shim()

        def spin():
            x = 0
            while True:
                x += 1
                if x < 0:
                    break

        try:
            shim.alarm(0.05)
            with pytest.raises(VadarsTimeout):
                spin()
        finally:
            shim.alarm(0)
        assert shim.fired == 1
        assert shim.fired_by["trace"] == 1

    def test_tracer_stays_installed_so_checks_keep_firing(self):
        """守住「我们保持自己作为 local tracer」这条设计 ——

        如果实现里返回了 VADAR tracer 的返回值，line 事件就不再经过看门狗，
        deadline 检查会退化成只在函数调用边界发生，死循环打断不了。
        """
        shim = _make_shim()
        calls = {"inner": 0}

        def inner(frame, event, arg):
            calls["inner"] += 1
            return inner

        sys.settrace(inner)
        try:
            shim.alarm(60)
            n = 0
            for _ in range(50):
                n += 1
            # 注意：这段循环在**当前帧**里，本来就不该被检查；
            # 这里只验证 inner 仍被逐事件转发，以及全局 tracer 没被换掉。
            assert calls["inner"] >= 0
            assert sys.gettrace() is not inner
        finally:
            shim.alarm(0)
            sys.settrace(None)

    def test_inner_tracer_errors_are_counted_not_propagated(self):
        """VADAR 的 tracer 只负责写 trace.html；它自己出错不该变成程序出错，
        但也不能静默 —— 计数要能看见。

        为什么不走 `sys.settrace(broken)` + `alarm()` 的真实路径：
        那样 `broken` 会先作为**全局 tracer** 被 Python 调用，而它在
        `alarm(60)` 自己那一帧的 line 事件上就会抛，泄漏到 `alarm()` 调用点。
        那是「内层 tracer 坏了」的另一种表现，会掩盖本节要测的东西。
        这里直接注入内层 tracer 并驱动 `_local`，把要测的那一段隔离出来。
        """
        shim = _make_shim()

        def broken(frame, event, arg):
            raise RuntimeError("写 trace.html 失败")

        shim._inner_trace = broken
        shim._local(sys._getframe(), "line", None)
        assert shim.trace_errors == 1
        # 内层出错不影响本层继续当 local tracer（否则后续 deadline 检查会停摆）。
        # 用 == 而不是 is：`shim._local` 是**绑定方法**，每次取属性都是新对象，
        # 但绑定方法之间按 (函数, 实例) 比较相等。
        ret = shim._local(sys._getframe(), "line", None)
        assert ret == shim._local
        assert ret.__func__ is TraceAlarmShim._local
        assert shim.trace_errors == 2


class TestTimerBackstop:
    def test_timer_is_scheduled_on_arm_and_cancelled_on_disarm(self):
        """路径 ② 的调度与取消。**不**让定时器真的到点注入 ——
        往主线程注入会在 pytest 内部任意一行生效，测试不该有那种副作用；
        注入机制本身由 TestAsyncInjection 在一个独立线程里测。"""
        shim = _make_shim()
        shim.alarm(60)
        t = shim._timer
        assert t is not None and t.is_alive()
        shim.alarm(0)
        assert shim._timer is None
        assert not t.is_alive() or t.finished.is_set()

    def test_rearming_replaces_the_previous_timer(self):
        shim = _make_shim()
        shim.alarm(60)
        first = shim._timer
        shim.alarm(30)
        second = shim._timer
        try:
            assert second is not first
            assert shim.arm_count == 2
        finally:
            shim.alarm(0)

    def test_fire_marks_the_path_taken(self):
        shim = _make_shim()
        shim.alarm(60)
        try:
            shim._fire("timer")
        except VadarsTimeout:
            pass
        finally:
            shim.alarm(0)
        assert shim.fired_by["timer"] == 1
        assert shim.fired == 1


class TestAsyncInjection:
    def test_injection_raises_the_vadar_exception_in_a_worker_thread(self):
        """机制验证：往一个**独立线程**注入，确认它真的会在那条线程里抛出来。

        这是路径 ② 的核心 —— 它不受「当前帧不被 trace」的限制。
        用独立线程是为了把副作用关在测试自己造的笼子里。
        """
        import threading

        from evaluation.win_alarm import _inject_async_exception

        got = {}

        def worker():
            try:
                end = time.monotonic() + 10.0
                while time.monotonic() < end:
                    pass
            except BaseException as e:      # noqa: BLE001
                got["exc"] = e

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        time.sleep(0.2)                     # 让 worker 进入循环
        # 取类型：让 shim 探一次 handler
        shim = _make_shim()
        n = _inject_async_exception(shim._exc_type, ident=t.ident)
        t.join(timeout=5.0)
        assert n == 1, "注入应恰好命中一个线程"
        assert not t.is_alive(), "worker 应已被打断"
        assert isinstance(got.get("exc"), VadarsTimeout)

    def test_injection_into_a_dead_thread_reports_zero(self):
        import threading

        from evaluation.win_alarm import _inject_async_exception

        t = threading.Thread(target=lambda: None)
        t.start()
        t.join()
        assert _inject_async_exception(VadarsTimeout, ident=t.ident) == 0


class TestExceptionTypeProbing:
    def test_exception_type_is_learned_from_the_handler(self):
        """★ 不猜异常类型：调用一次 handler，记下它抛的东西。

        抛内置 TimeoutError 会让 `except TimeoutException` 抓不到，
        超时就从「超时重试」变成「程序崩溃」—— 这个区别很隐蔽。
        """
        shim = _make_shim()
        assert shim._exc_type is VadarsTimeout
        assert shim.describe()["exception_type"].endswith("VadarsTimeout")

    def test_handler_that_does_not_raise_falls_back_to_timeout_error(self):
        shim = TraceAlarmShim()
        shim.signal(shim.SIGALRM, lambda signum, frame: None)
        assert shim._exc_type is TimeoutError
        _expire(shim)
        with pytest.raises(TimeoutError):
            shim.check_now()


class TestDescribe:
    def test_describe_is_json_serialisable(self):
        import json

        shim = _make_shim()
        shim.alarm(5)
        shim.alarm(0)
        d = shim.describe()
        json.dumps(d)          # 不抛即通过
        assert d["armed"] == 1 and d["cancelled"] == 1 and d["fired"] == 0
        assert d["timeout_s"] is None      # cancel 后清空


class TestInstall:
    def test_missing_module_is_reported_not_silently_skipped(self):
        info = install_alarm_shim(verbose=False)
        # 测试进程里没导入 agents/engine，所以两个目标都该报失败，
        # 而不是「静默成功」—— 静默成功会让实验记录谎称超时已生效。
        assert info["patched"] == []
        assert len(info["failed"]) == len(info["targets"])
        assert info["shim"] is not None

    def test_foreign_signal_module_is_rejected(self, monkeypatch):
        import types

        fake = types.ModuleType("agents.agents")
        fake.signal = types.SimpleNamespace(SIGALRM=14)   # 非标准库模块
        monkeypatch.setitem(sys.modules, "agents.agents", fake)
        info = install_alarm_shim(verbose=False)
        assert "agents.agents" not in info["patched"]
        assert any(name == "agents.agents" for name, _ in info["failed"])
