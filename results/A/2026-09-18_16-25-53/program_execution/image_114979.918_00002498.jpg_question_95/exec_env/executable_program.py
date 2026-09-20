import math

# PROGRAM STARTS HERE

bedside_tables = loc(image, "bedside table")
bedside_table_bbox = bedside_tables[0]

cabinets = loc(image, "cabinet")
left_most_cabinet_bbox = cabinets[0]
for cab in cabinets:
    if cab[0] < left_most_cabinet_bbox[0]:
        left_most_cabinet_bbox = cab

bedside_table_size = get_2D_object_size(image, bedside_table_bbox)
bedside_table_height_2d = bedside_table_size[1]
bedside_table_depth = depth(image, bedside_table_bbox)

left_cabinet_size = get_2D_object_size(image, left_most_cabinet_bbox)
left_cabinet_height_2d = left_cabinet_size[1]
left_cabinet_depth = depth(image, left_most_cabinet_bbox)

bedside_table_height_3d = bedside_table_height_2d * bedside_table_depth
left_cabinet_height_3d = left_cabinet_height_2d * left_cabinet_depth

combined_height_3d = bedside_table_height_3d + left_cabinet_height_3d
scale_factor = 3.0 / combined_height_3d

beds = loc(image, "bed")
bed_bbox = beds[0]

bed_size = get_2D_object_size(image, bed_bbox)
bed_width_2d = bed_size[0]
bed_depth = depth(image, bed_bbox)

bed_width_3d = bed_width_2d * bed_depth * scale_factor

final_result = bed_width_3d


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

with open(r"D:\3D_Spatial_Agent\results\A\2026-09-18_16-25-53\program_execution\image_114979.918_00002498.jpg_question_95/exec_env/result.json", "w+") as result_file:
    json.dump(serializable_globals, result_file)
        