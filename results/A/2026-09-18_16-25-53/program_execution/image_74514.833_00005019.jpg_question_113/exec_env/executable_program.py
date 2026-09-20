import math

# PROGRAM STARTS HERE

table_list = loc(image, "table")
dresser_list = loc(image, "brown dresser")

table_bb = table_list[0]
dresser_bb = dresser_list[0]

table_wh = get_2D_object_size(image, table_bb)
dresser_wh = get_2D_object_size(image, dresser_bb)

table_h_px = table_wh[1]
dresser_h_px = dresser_wh[1]

table_z = depth(image, table_bb)
dresser_z = depth(image, dresser_bb)

table_h_m = table_h_px * table_z
dresser_h_m = dresser_h_px * dresser_z

final_result = (dresser_h_m / table_h_m) * 1.8


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

with open(r"D:\3D_Spatial_Agent\results\A\2026-09-18_16-25-53\program_execution\image_74514.833_00005019.jpg_question_113/exec_env/result.json", "w+") as result_file:
    json.dump(serializable_globals, result_file)
        