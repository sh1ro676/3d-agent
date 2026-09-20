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

# Find all sofas
sofas = loc(image, "sofa")

# Find the rightmost sofa (highest xmax value)
rightmost_sofa = None
max_xmax = -1
for sofa in sofas:
    if sofa[2] > max_xmax:  # sofa[2] is xmax
        max_xmax = sofa[2]
        rightmost_sofa = sofa

# Find the circular table under the TV
# First find the TV
tv = loc(image, "TV")
if tv:
    tv_bbox = tv[0]  # Take the first TV found
else:
    tv_bbox = None

# Find circular tables
circular_tables = loc(image, "circular table")

# Find the table under the TV
table_under_tv = None
if tv_bbox and circular_tables:
    for table in circular_tables:
        # Check if table is under the TV (table's ymin should be greater than TV's ymax, and x ranges should overlap)
        if table[1] > tv_bbox[3]:  # table ymin > TV ymax (table is below TV)
            # Check horizontal overlap
            if not (table[2] < tv_bbox[0] or table[0] > tv_bbox[2]):
                table_under_tv = table
                break

# Calculate volume of rightmost sofa
if rightmost_sofa:
    sofa_width, sofa_height = get_2D_object_size(image, rightmost_sofa)
    sofa_depth = depth(image, rightmost_sofa)
    sofa_volume = sofa_width * sofa_height * sofa_depth
else:
    sofa_volume = 0

# Calculate volume of circular table under TV
if table_under_tv:
    table_width, table_height = get_2D_object_size(image, table_under_tv)
    table_depth = depth(image, table_under_tv)
    table_volume = table_width * table_height * table_depth
else:
    table_volume = 0

# Calculate ratio
if table_volume != 0:
    final_result = sofa_volume / table_volume
else:
    final_result = 0


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

with open(r"D:\3D_Spatial_Agent\results\A\2026-09-18_16-43-16\program_execution\image_183724.018_00000638.jpg_question_31/exec_env/result.json", "w+") as result_file:
    json.dump(serializable_globals, result_file)
        