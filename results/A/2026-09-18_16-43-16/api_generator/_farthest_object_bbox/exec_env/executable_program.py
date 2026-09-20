import math
def loc(image, object_prompt):

	return [[25, 25, 50, 50]]

import math
def vqa(image, question, bbox):

	return ""

import math
def depth(image, bbox):

	return 1.0

import math
def same_object(image, bbox1, bbox2):

	return False

import math
def get_2D_object_size(image, bbox):

	return (50, 50)


def _closest_object_bbox(image, object_prompt):

    
    bboxes = loc(image, object_prompt)
    if not bboxes:
        return None
    closest_bbox = bboxes[0]
    closest_depth = depth(image, closest_bbox)
    for bbox in bboxes[1:]:
        d = depth(image, bbox)
        if d < closest_depth:
            closest_depth = d
            closest_bbox = bbox
    return closest_bbox
    


# PROGRAM STARTS HERE

bboxes = loc(image, object_prompt)
if not bboxes:
    final_result = None

farthest_bbox = bboxes[0]
max_depth = depth(image, bboxes[0])

for bbox in bboxes[1:]:
    d = depth(image, bbox)
    if d > max_depth:
        max_depth = d
        farthest_bbox = bbox

final_result = farthest_bbox


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


serializable_globals = {_k: _unbox_scalar(_v) for _k, _v in list(globals().items())
                        if is_serializable(_unbox_scalar(_v))}

with open(r"D:\3D_Spatial_Agent\results\A\2026-09-18_16-43-16\api_generator\_farthest_object_bbox\exec_env/result.json", "w+") as result_file:
    json.dump(serializable_globals, result_file)
        