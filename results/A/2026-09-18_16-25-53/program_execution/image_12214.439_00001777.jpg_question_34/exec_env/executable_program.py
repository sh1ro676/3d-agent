import math

# PROGRAM STARTS HERE

sofa_boxes = loc(image, "sofa")
sofa_size = get_2D_object_size(image, sofa_boxes[0])
scale_factor = 0.4 / sofa_size[1]

stool_boxes = loc(image, "stool")

stool_width_pixels = 0
for sb in stool_boxes:
    check = vqa(image, "Is this stool in front of the piano?", sb)
    if check == "yes":
        stool_width_pixels = get_2D_object_size(image, sb)[0]
        break

final_result = stool_width_pixels * scale_factor


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

with open(r"D:\3D_Spatial_Agent\results\A\2026-09-18_16-25-53\program_execution\image_12214.439_00001777.jpg_question_34/exec_env/result.json", "w+") as result_file:
    json.dump(serializable_globals, result_file)
        