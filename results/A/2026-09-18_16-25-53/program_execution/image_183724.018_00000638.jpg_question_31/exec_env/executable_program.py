import math

# PROGRAM STARTS HERE

sofas = loc(image, "sofa")
rightmost_sofa = sofas[0]
for s in sofas:
    if s[2] > rightmost_sofa[2]:
        rightmost_sofa = s

tables = loc(image, "table")
circular_table = tables[0]

ss = get_2D_object_size(image, rightmost_sofa)
sd = depth(image, rightmost_sofa)
sofa_volume = ss[0] * ss[1] * sd

ts = get_2D_object_size(image, circular_table)
td = depth(image, circular_table)
table_volume = ts[0] * ts[1] * td

final_result = sofa_volume / table_volume


# WRITE NAMESPACE
import json

def is_serializable(obj):
    try:
        json.dumps(obj)
    except (TypeError, OverflowError):
        return False
    return True

def _unbox_scalar(v):
    """0 维 numpy/torch 标量 → Python 标量。其余原样返回。"""
    if getattr(v, "shape", None) == ():
        item = getattr(v, "item", None)
        if callable(item):
            try:
                return item()
            except Exception:
                return v
    return v


def is_serializable(obj):
    """VADAR 原本只捕 TypeError/OverflowError —— 漏了 `ValueError`。

    `json.dumps` 撞上**循环引用**时抛的是 `ValueError("Circular reference
    detected")`，没被捕获 → 整段程序崩掉、VADAR 记
    `Error in executing <方法>: Circular reference detected`（2026-09-18 实测）。
    一个循环引用的中间变量不该能废掉整道题：本意只是「这个值不写进结果」。
    """
    try:
        json.dumps(obj)
    except (TypeError, OverflowError, ValueError):
        return False
    return True


serializable_globals = {}
for _k, _v in list(globals().items()):
    _v = _unbox_scalar(_v)
    if is_serializable(_v):
        serializable_globals[_k] = _v

with open(r"D:\3D_Spatial_Agent\results\A\2026-09-18_16-25-53\program_execution\image_183724.018_00000638.jpg_question_31/exec_env/result.json", "w+") as result_file:
    json.dump(serializable_globals, result_file)
        