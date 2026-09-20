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

# Locate the washing machine
washing_machine_bboxes = loc(image, "washing machine")

# We need to find the washing machine that has a cabinet above it
# Let's get all cabinets
cabinet_bboxes = loc(image, "cabinet")

# Find the washing machine and the cabinet above it
# We'll check each washing machine and each cabinet to find the pair where cabinet is above washing machine
washing_machine_bbox = None
cabinet_above_bbox = None

for wm_bbox in washing_machine_bboxes:
    wm_xmin, wm_ymin, wm_xmax, wm_ymax = wm_bbox
    wm_center_x = (wm_xmin + wm_xmax) / 2
    
    for cab_bbox in cabinet_bboxes:
        cab_xmin, cab_ymin, cab_xmax, cab_ymax = cab_bbox
        cab_center_x = (cab_xmin + cab_xmax) / 2
        
        # Check if cabinet is above washing machine (cabinet ymin < washing machine ymin)
        # and they are roughly aligned horizontally
        if cab_ymax < wm_ymin and abs(cab_center_x - wm_center_x) < (wm_xmax - wm_xmin):
            washing_machine_bbox = wm_bbox
            cabinet_above_bbox = cab_bbox
            break
    
    if washing_machine_bbox is not None:
        break

# Get the 2D heights and depths of both objects
wm_width_2d, wm_height_2d = get_2D_object_size(image, washing_machine_bbox)
cab_width_2d, cab_height_2d = get_2D_object_size(image, cabinet_above_bbox)

wm_depth = depth(image, washing_machine_bbox)
cab_depth = depth(image, cabinet_above_bbox)

# Calculate 3D heights (2D height * depth)
wm_height_3d = wm_height_2d * wm_depth
cab_height_3d = cab_height_2d * cab_depth

# The new washing machine height is 1 foot
new_wm_height = 1.0

# Calculate the scale factor
scale_factor = new_wm_height / wm_height_3d

# Calculate the new cabinet height
new_cab_height = cab_height_3d * scale_factor

final_result = new_cab_height


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

with open(r"D:\3D_Spatial_Agent\results\A\2026-09-18_16-43-16\program_execution\image_164039.451_00000637.jpg_question_235/exec_env/result.json", "w+") as result_file:
    json.dump(serializable_globals, result_file)
        