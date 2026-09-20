"""evaluation —— 实验臂运行与评估。

模块职责（与文档 §17 的目录约定一致）：

    metrics.py        VADAR 的四个子指标 + Total 聚合（口径要能被论文数字反推）
    win_alarm.py      Windows 上替代 signal.SIGALRM 的执行看门狗
    vadar_compat.py   ★ 让 vendor/VADAR 一字不改跑起来（stub 包 + 视觉适配 + 桥接）
    runner.py         跑一个实验臂，产出 results/<arm>/*

设计约束（写在这里，免得后来者以为可以随手改）：
* **vendor/VADAR 不改一字节**，所有差异都在这层适配器里显式可见。
* 每个结果文件都必须自带模型指纹与环境指纹 —— §16.2 明说了
  「换模型后的对比无法归因」是这类实验最常见的失败方式。
"""

__all__ = ["metrics", "win_alarm", "vadar_compat"]
