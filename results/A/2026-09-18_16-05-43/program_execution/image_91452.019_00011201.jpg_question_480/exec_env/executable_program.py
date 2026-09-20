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

# Find all chairs
chairs = loc(image, "chair")

# Find the chair closest to the camera (smallest depth)
closest_chair_bbox = None
min_depth = float('inf')

for chair_bbox in chairs:
    chair_depth = depth(image, chair_bbox)
    if chair_depth < min_depth:
        min_depth = chair_depth
        closest_chair_bbox = chair_bbox

# Get 2D height of closest chair
closest_2d_width, closest_2d_height = get_2D_object_size(image, closest_chair_bbox)

# Calculate 3D height of closest chair (in pixel*meters)
closest_3d_height = closest_2d_height * min_depth

# The 3D height of closest chair is 5 feet
scale_factor = 5.0 / closest_3d_height

# Find the chair farthest from the camera (largest depth)
farthest_chair_bbox = None
max_depth = -1

for chair_bbox in chairs:
    chair_depth = depth(image, chair_bbox)
    if chair_depth > max_depth:
        max_depth = chair_depth
        farthest_chair_bbox = chair_bbox

# Get 2D height of farthest chair
farthest_2d_width, farthest_2d_height = get_2D_object_size(image, farthest_chair_bbox)

# Calculate 3D height of farthest chair (in pixel*meters)
farthest_3d_height = farthest_2d_height * max_depth

# Convert to feet using scale factor
final_result = farthest_3d_height * scale_factor


# WRITE NAMESPACE
import json

def is_serializable(obj):
    try:
        json.dumps(obj)
    except (TypeError, OverflowError):
        return False
    return True

serializable_globals = {k: v for k, v in globals().items() if is_serializable(v)}

with open(r"D:\3D_Spatial_Agent\results\A\2026-09-18_16-05-43\program_execution\image_91452.019_00011201.jpg_question_480/exec_env/result.json", "w+") as result_file:
    json.dump(serializable_globals, result_file)
        