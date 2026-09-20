#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""win_alarm.py —— 在 Windows 上替代 `signal.SIGALRM` 的执行看门狗。

为什么需要它
------------
VADAR 给「生成程序」设了两道执行超时，用来抓无限递归 / 死循环：

    agents/agents.py:572-576   signal.alarm(30)    # APIAgent 测试 API 实现时
    engine/engine.py:594-598   signal.alarm(200)   # Engine 执行解题程序时

Windows 的 `signal` 模块**没有** `SIGALRM`，也没有 `alarm`，
所以这两行在原生 Windows 上会直接 `AttributeError` —— 这是 §17 说
「原版必须在 WSL 里跑」的技术原因之一。本项目全程 Windows 原生，必须补上。

两条互补的打断路径（缺一条都有真实盲区）
----------------------------------------
**路径 ①：顺路搭 VADAR 自己的 `sys.settrace`。**
关键观察：VADAR 在这两处**本来就先装 `sys.settrace` 再开 alarm**：

    sys.settrace(self._trace_execution)
    signal.alarm(200)

`sys.settrace` 的 local tracer 在**每一行字节码**都会被调用一次。
于是「超时」不必由信号驱动 —— 在 tracer 里比一下墙上时钟就够了，
判定精确到行，而且几乎没有额外开销（tracer 本来就在跑）。

**盲区**：`sys.settrace` 只对**调用之后新建的帧**生效。
当前已在执行的那个帧不会被 trace。

**路径 ②：后台定时器 + 异步异常注入，补上面那个盲区。**
对 VADAR 的真实用法来说盲区其实不触发（`runpy.run_path` 一定新建模块帧），
但它是个**看不见的假设**：哪天有人把 `alarm()` 放在循环里调用，
路径 ① 就会一声不响地失效。所以再加一条 `threading.Timer`，
到点了用 `PyThreadState_SetAsyncExc` 往主线程塞异常。
这条路径不需要 tracer，能覆盖当前帧。

它能越过 C 调用吗
------------------
**不能可靠地越过。** CPython 的异步异常要等目标线程**回到字节码边界**才投递；
如果线程卡在一次不返回的长 C 调用里（比如超大图的模型推理），
两条路径都不会及时响。Linux 上真信号能中断可中断的系统调用，**本实现不能**。
这是明确的降级，不是等价替换 —— 写在这里，免得后来者以为 `alarm` 语义完全一致。

异常类型必须与 VADAR 一致
--------------------------
`engine.py:597` / `agents.py:575` 抓的是 `engine_utils.TimeoutException`。
抛内置 `TimeoutError` 会**抓不到**，超时就会表现成「程序崩溃」而不是「超时重试」。
所以本模块不去猜这个类型：在 `signal(signum, handler)` 时**探一次 handler**
（调用它、接住它抛的东西、记下类型），之后两条路径都抛同一个类型。
"""

from __future__ import annotations

import ctypes
import sys
import threading
import time
from typing import Any, Callable, Optional

__all__ = ["TraceAlarmShim", "install_alarm_shim", "ALARM_TARGETS"]


class TraceAlarmShim:
    """鸭子类型替代 `signal` 模块里的 SIGALRM/signal/alarm 三件套。

    只实现 VADAR 用到的那部分，其他属性访问会抛 AttributeError —— 故意的：
    免得有别的调用点悄悄依赖了一个没实现的方法。
    """

    #: `signal.SIGALRM` 在 Linux 上是 14；这里只是个常量，不参与真正投递。
    SIGALRM = 14

    def __init__(self) -> None:
        self._handler: Optional[Callable[..., Any]] = None
        self._exc_type: type = TimeoutError
        self._inner_trace: Optional[Callable[..., Any]] = None
        self._deadline: Optional[float] = None
        self._timeout: Optional[float] = None
        self._timer: Optional[threading.Timer] = None
        self._lock = threading.RLock()
        self.fired = 0            #: 实际触发超时的次数（供报告核对）
        self.arm_count = 0        #: alarm(>0) 次数
        self.cancel_count = 0     #: alarm(0) 次数
        self.trace_errors = 0     #: 转发给 VADAR tracer 时它自己抛错的次数
        self.fired_by = {"trace": 0, "timer": 0}

    # ---------------- signal 模块兼容面 ----------------
    def signal(self, signum: int, handler: Callable[..., Any]) -> None:
        if signum != self.SIGALRM:
            raise ValueError("TraceAlarmShim 只支持 SIGALRM，收到 %r" % (signum,))
        self._handler = handler
        self._exc_type = _probe_exception_type(handler)

    def alarm(self, seconds: float) -> int:
        """`alarm(0)` = 取消；`alarm(n>0)` = 武装 n 秒看门狗。返回值恒为 0。"""
        if seconds and float(seconds) > 0:
            self._cancel_timer()
            self._inner_trace = sys.gettrace()
            self._timeout = float(seconds)
            self._deadline = time.monotonic() + float(seconds)
            self.arm_count += 1
            sys.settrace(self._tracer)
            t = threading.Timer(float(seconds), self._on_timer)
            t.daemon = True
            self._timer = t
            t.start()
            return 0

        self.cancel_count += 1
        self._deadline = None
        self._timeout = None
        self._cancel_timer()
        if self._inner_trace is not None:
            sys.settrace(self._inner_trace)
        else:
            sys.settrace(None)
        return 0

    def _cancel_timer(self) -> None:
        t, self._timer = self._timer, None
        if t is not None:
            t.cancel()

    # ---------------- 看门狗本体 ----------------
    def check_now(self) -> None:
        """供单测直接调用的判定入口（不依赖 sys.settrace 与定时器）。"""
        if self._deadline is not None and time.monotonic() > self._deadline:
            self._fire("trace")

    def _on_timer(self) -> None:
        """路径 ②：定时器到点，往主线程注入异步异常。"""
        with self._lock:
            if self._deadline is None or time.monotonic() < self._deadline:
                return                      # 已被 alarm(0) 取消
        self._fire("timer", async_inject=True)

    def _fire(self, via: str = "trace", async_inject: bool = False) -> None:
        """触发超时。默认复用 VADAR 的 handler 抛异常（保证 catch 的类型一致）。"""
        self.fired += 1
        self.fired_by[via] = self.fired_by.get(via, 0) + 1
        self._deadline = None
        self._cancel_timer()

        if async_inject:
            # 路径 ②：注入到主线程。若返回 >1 说明注入到了多个线程，按
            # CPython 文档要求立即置 0 回滚，否则会留下不可预期的状态。
            n = _inject_async_exception(self._exc_type)
            if n == 1:
                return
            # 注入失败/异常（例如非主线程调用）→ 退回主线程抛。
            # 这条兜底只在测试或怪异调用方式下走到。
        h = self._handler
        if h is not None:
            # `timeout_handler(signum, frame)` 的正常行为就是抛 TimeoutException，
            # 这里故意不 catch：让那个类原样冒出去，
            # engine.py / agents.py 里的 `except TimeoutException` 才认得。
            h(self.SIGALRM, None)
        raise self._exc_type("VADAR 程序执行超时（TraceAlarmShim 兜底路径）")

    def _tracer(self, frame: Any, event: str, arg: Any) -> Any:
        """global trace：新帧建立时被调用，返回该帧的 local tracer。"""
        return self._local

    def _local(self, frame: Any, event: str, arg: Any) -> Any:
        if self._deadline is not None and time.monotonic() > self._deadline:
            self._fire("trace")
        if self._inner_trace is not None:
            try:
                self._inner_trace(frame, event, arg)
            except Exception:
                # VADAR 的 tracer 只是往 trace.html 里写一行；它自己出错不该
                # 变成「程序出错」。但也不能静默 —— 计数让异常可见。
                self.trace_errors += 1
        # 保持自己是 local tracer：返回 inner 的返回值会让逐行检查退化成逐帧
        return self._local

    def describe(self) -> dict:
        return {
            "armed": self.arm_count,
            "cancelled": self.cancel_count,
            "fired": self.fired,
            "fired_by": dict(self.fired_by),
            "trace_errors": self.trace_errors,
            "timeout_s": self._timeout,
            "exception_type": "%s.%s" % (self._exc_type.__module__,
                                         self._exc_type.__name__),
        }


def _probe_exception_type(handler: Callable[..., Any]) -> type:
    """调用一次 handler，记下它抛的类型 —— 不猜，直接问。"""
    try:
        handler(TraceAlarmShim.SIGALRM, None)
    except BaseException as e:          # noqa: BLE001  故意接所有，包括 KeyboardInterrupt
        return type(e)
    return TimeoutError


def _inject_async_exception(exc_type: type, ident: Optional[int] = None) -> int:
    """把 `exc_type` 注入目标线程。返回 1 表示成功注入恰好一个线程。

    `ident=None` 表示主线程。之所以把目标线程做成参数，是为了让单测
    能把它指向一个**自己起的 worker 线程**去验证机制 —— 往主线程注入
    会在 pytest 内部任意一行生效，可能污染整个会话；测试不该有这种副作用。
    """
    try:
        if ident is None:
            ident = threading.main_thread().ident
        if ident is None:
            return 0
        ctypes.pythonapi.PyThreadState_SetAsyncExc.argtypes = [
            ctypes.c_ulong, ctypes.py_object]
        ctypes.pythonapi.PyThreadState_SetAsyncExc.restype = ctypes.c_int
        n = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(ident), ctypes.py_object(exc_type))
        if n > 1:
            # 文档要求：注入到多个线程时必须立刻用 NULL 回滚
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(ident), None)
            return 0
        return int(n)
    except Exception:
        return 0


#: 需要被替换 `signal` 属性的模块（VADAR 里只有这两处用 alarm）
ALARM_TARGETS = ("agents.agents", "engine.engine")


def install_alarm_shim(verbose: bool = True) -> dict:
    """把已导入的 VADAR 模块命名空间里的 `signal` 换成 TraceAlarmShim。

    必须在 `import agents.agents` / `import engine.engine` **之后**调用。
    返回一份可写进实验记录的报告（`shim` 是对象，落盘前要调 `.describe()`）。
    """
    shim = TraceAlarmShim()
    info = {"patched": [], "failed": [], "targets": list(ALARM_TARGETS), "shim": shim}
    for name in ALARM_TARGETS:
        mod = sys.modules.get(name)
        if mod is None:
            info["failed"].append((name, "模块未导入"))
            continue
        orig = getattr(mod, "signal", None)
        if orig is None:
            info["failed"].append((name, "命名空间里没有 signal"))
            continue
        if isinstance(orig, TraceAlarmShim):
            info["patched"].append(name + "(已装)")
        elif getattr(orig, "__name__", "") == "signal":
            setattr(mod, "signal", shim)
            info["patched"].append(name)
        else:
            info["failed"].append((name, "signal 不是标准库模块: %r" % (orig,)))

    if verbose:
        print("[win_alarm] 已替换 alarm 的模块: %s" % (", ".join(info["patched"]) or "(无)"))
        for name, err in info["failed"]:
            print("[win_alarm] !! %s 未替换: %s" % (name, err))
    return info
