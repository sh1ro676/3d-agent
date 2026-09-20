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


# PROGRAM STARTS HERE

xmin, ymin, xmax, ymax = bbox
width = xmax - xmin
height = ymax - ymin
final_result = (width, height)


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


serializable_globals = {}
for _k, _v in list(globals().items()):
    _v = _unbox_scalar(_v)
    if is_serializable(_v):
        serializable_globals[_k] = _v

with open(r"D:\3D_Spatial_Agent\results\A\2026-09-18_16-15-03\api_generator\_get_2D_object_size\exec_env/result.json", "w+") as result_file:
    json.dump(serializable_globals, result_file)
        