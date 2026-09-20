import math

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
    

def _farthest_object_bbox(image, object_prompt):

    
    bboxes = loc(image, object_prompt)
    if not bboxes:
        return None
    
    farthest_bbox = bboxes[0]
    max_depth = depth(image, bboxes[0])
    
    for bbox in bboxes[1:]:
        d = depth(image, bbox)
        if d > max_depth:
            max_depth = d
            farthest_bbox = bbox
    
    return farthest_bbox
    

def _count_objects(image, object_prompt):

    
    bboxes = loc(image=image, object_prompt=object_prompt)
    return len(bboxes)
    

def _get_color(image, bbox):

    
    color = vqa(image=image, question='What color is this object?', bbox=bbox)
    return color
    

def _get_real_height(image, bbox, ref_bbox, ref_height):

    
    # Get 2D sizes of both objects
    target_width, target_height = get_2D_object_size(image, bbox)
    ref_width, ref_height = get_2D_object_size(image, ref_bbox)
    
    # Get depths of both objects
    target_depth = depth(image, bbox)
    ref_depth = depth(image, ref_bbox)
    
    # Convert 2D heights to 3D heights (2D size * depth)
    target_3D_height = target_height * target_depth
    ref_3D_height = ref_height * ref_depth
    
    # Scale using the known real-world height of the reference object
    real_height = (target_3D_height / ref_3D_height) * ref_height
    
    return real_height
    

def _get_real_width(image, bbox, ref_bbox, ref_width):

    
    target_width_2d, target_height_2d = get_2D_object_size(image, bbox)
    target_depth = depth(image, bbox)
    target_3d_width = target_width_2d * target_depth
    
    ref_width_2d, ref_height_2d = get_2D_object_size(image, ref_bbox)
    ref_depth = depth(image, ref_bbox)
    ref_3d_width = ref_width_2d * ref_depth
    
    scale = ref_width / ref_3d_width
    return target_3d_width * scale
    

# PROGRAM STARTS HERE

# Find the bedside table
bedside_table_bboxes = loc(image, "bedside table")
bedside_table_bbox = bedside_table_bboxes[0] if bedside_table_bboxes else None

# Find all cabinets and get the left-most one
cabinet_bboxes = loc(image, "cabinet")
if cabinet_bboxes:
    # Left-most means smallest xmin
    left_most_cabinet_bbox = min(cabinet_bboxes, key=lambda bbox: bbox[0])
else:
    left_most_cabinet_bbox = None

# Find the bed
bed_bboxes = loc(image, "bed")
bed_bbox = bed_bboxes[0] if bed_bboxes else None

# Get 2D heights of bedside table and left-most cabinet
_, bedside_table_2d_height = get_2D_object_size(image, bedside_table_bbox)
_, cabinet_2d_height = get_2D_object_size(image, left_most_cabinet_bbox)

# Get depths
bedside_table_depth = depth(image, bedside_table_bbox)
cabinet_depth = depth(image, left_most_cabinet_bbox)

# Calculate 3D heights (2D height * depth)
bedside_table_3d_height = bedside_table_2d_height * bedside_table_depth
cabinet_3d_height = cabinet_2d_height * cabinet_depth

# Combined height is 3m
combined_3d_height = bedside_table_3d_height + cabinet_3d_height

# Scale factor: real_world / pixel_world
# We need to find the scale such that combined height = 3m
scale_factor = 3.0 / combined_3d_height

# Get bed's 2D width and depth
bed_2d_width, _ = get_2D_object_size(image, bed_bbox)
bed_depth = depth(image, bed_bbox)

# Calculate bed's 3D width
bed_3d_width = bed_2d_width * bed_depth

# Apply scale factor to get real-world width
bed_real_width = bed_3d_width * scale_factor

final_result = bed_real_width


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

with open(r"D:\3D_Spatial_Agent\results\A\2026-09-18_16-43-16\program_execution\image_114979.918_00002498.jpg_question_95/exec_env/result.json", "w+") as result_file:
    json.dump(serializable_globals, result_file)
        