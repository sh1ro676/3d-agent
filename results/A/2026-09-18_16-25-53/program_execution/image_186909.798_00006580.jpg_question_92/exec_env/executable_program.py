import math

# PROGRAM STARTS HERE

tv_objects = loc(image, "TV")
stand_objects = loc(image, "TV stand")

tv_bbox_1 = tv_objects[0]
stand_bbox_1 = stand_objects[0]

tv_size_tuple = get_2D_object_size(image, tv_bbox_1)
stand_size_tuple = get_2D_object_size(image, stand_bbox_1)

tv_height_pixels = tv_size_tuple[1]
stand_height_pixels = stand_size_tuple[1]

tv_depth_meters = depth(image, tv_bbox_1)
stand_depth_meters = depth(image, stand_bbox_1)

tv_height_3d_units = tv_height_pixels * tv_depth_meters
stand_height_3d_units = stand_height_pixels * stand_depth_meters

combined_height_3d_units = tv_height_3d_units + stand_height_3d_units

conversion_factor = 2.0 / combined_height_3d_units

coffee_table_objects = loc(image, "coffee table")

coffee_table_bbox_1 = coffee_table_objects[0]
brown_answer = vqa(image, "Is this coffee table brown?", coffee_table_bbox_1)
silver_answer = vqa(image, "Is this coffee table silver?", coffee_table_bbox_1)

coffee_table_size_tuple = get_2D_object_size(image, coffee_table_bbox_1)
coffee_table_width_pixels = coffee_table_size_tuple[0]
coffee_table_depth_meters = depth(image, coffee_table_bbox_1)
coffee_table_width_3d_units = coffee_table_width_pixels * coffee_table_depth_meters

final_result = coffee_table_width_3d_units * conversion_factor


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

with open(r"D:\3D_Spatial_Agent\results\A\2026-09-18_16-25-53\program_execution\image_186909.798_00006580.jpg_question_92/exec_env/result.json", "w+") as result_file:
    json.dump(serializable_globals, result_file)
        