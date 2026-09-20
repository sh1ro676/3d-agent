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

# Find the TV and TV stand to establish reference height
tv_bboxes = loc(image, "TV")
tv_stand_bboxes = loc(image, "TV stand")

# Get the first TV and TV stand (assuming there's one of each)
tv_bbox = tv_bboxes[0]
tv_stand_bbox = tv_stand_bboxes[0]

# Get 2D heights of TV and TV stand
tv_width_2d, tv_height_2d = get_2D_object_size(image, tv_bbox)
tv_stand_width_2d, tv_stand_height_2d = get_2D_object_size(image, tv_stand_bbox)

# Get depths to convert to 3D heights
tv_depth = depth(image, tv_bbox)
tv_stand_depth = depth(image, tv_stand_bbox)

# Calculate 3D heights (2D height * depth)
tv_height_3d = tv_height_2d * tv_depth
tv_stand_height_3d = tv_stand_height_2d * tv_stand_depth

# Combined height is 2m, so we can find the scale factor
combined_height_3d = tv_height_3d + tv_stand_height_3d
scale_factor = 2.0 / combined_height_3d

# Now find the brown and silver coffee table
coffee_table_bboxes = loc(image, "brown and silver coffee table")

# Get the first coffee table
coffee_table_bbox = coffee_table_bboxes[0]

# Get 2D width of coffee table
coffee_table_width_2d, coffee_table_height_2d = get_2D_object_size(image, coffee_table_bbox)

# Get depth of coffee table
coffee_table_depth = depth(image, coffee_table_bbox)

# Calculate 3D width (2D width * depth)
coffee_table_width_3d = coffee_table_width_2d * coffee_table_depth

# Apply scale factor to get real-world width
final_result = coffee_table_width_3d * scale_factor


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

with open(r"D:\3D_Spatial_Agent\results\A\2026-09-18_16-43-16\program_execution\image_186909.798_00006580.jpg_question_92/exec_env/result.json", "w+") as result_file:
    json.dump(serializable_globals, result_file)
        