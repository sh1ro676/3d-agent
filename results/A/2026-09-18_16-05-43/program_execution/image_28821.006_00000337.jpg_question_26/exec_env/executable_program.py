import math
def _count_objects(image, object_prompt):
    
    bboxes = loc(image, object_prompt)
    return len(bboxes)
    
def _get_closest_object(image, object_prompt):
    
    bboxes = loc(image, object_prompt)
    if not bboxes:
        return None
    
    closest_bbox = None
    min_depth = float('inf')
    
    for bbox in bboxes:
        d = depth(image, bbox)
        if d < min_depth:
            min_depth = d
            closest_bbox = bbox
    
    return closest_bbox
    
def _get_color(image, bbox):
    
    color = vqa(image=image, question='What color is this object?', bbox=bbox)
    return color
    
def _get_real_world_width(image, bbox, height_m):
    
    width_px, height_px = get_2D_object_size(image, bbox)
    return (width_px / height_px) * height_m
    

# PROGRAM STARTS HERE

# Find all cabinets
cabinets = loc(image, "cabinet")

# Find the leftmost cabinet (smallest xmin)
leftmost_cabinet = min(cabinets, key=lambda bbox: bbox[0])

# Get 2D size of leftmost cabinet
cabinet_width_2d, cabinet_height_2d = get_2D_object_size(image, leftmost_cabinet)

# Get depth of leftmost cabinet
cabinet_depth = depth(image, leftmost_cabinet)

# Calculate real-world width of cabinet in 3D
# 3D width = 2D width * depth
cabinet_width_3d = cabinet_width_2d * cabinet_depth

# Given: cabinet is 2m wide in real world
# Scale factor: real_world_width / 3D_width
scale_factor = 2.0 / cabinet_width_3d

# Find the black chair
chairs = loc(image, "chair")

# Find the black chair
black_chair = None
for chair in chairs:
    color = _get_color(image, chair)
    if color.lower() == "black":
        black_chair = chair
        break

# Get 2D height of black chair
chair_width_2d, chair_height_2d = get_2D_object_size(image, black_chair)

# Get depth of black chair
chair_depth = depth(image, black_chair)

# Calculate real-world height of chair in 3D
# 3D height = 2D height * depth
chair_height_3d = chair_height_2d * chair_depth

# Convert to real-world meters using scale factor
final_result = chair_height_3d * scale_factor


# WRITE NAMESPACE
import json

def is_serializable(obj):
    try:
        json.dumps(obj)
    except (TypeError, OverflowError):
        return False
    return True

serializable_globals = {k: v for k, v in globals().items() if is_serializable(v)}

with open(r"D:\3D_Spatial_Agent\results\A\2026-09-18_16-05-43\program_execution\image_28821.006_00000337.jpg_question_26/exec_env/result.json", "w+") as result_file:
    json.dump(serializable_globals, result_file)
        