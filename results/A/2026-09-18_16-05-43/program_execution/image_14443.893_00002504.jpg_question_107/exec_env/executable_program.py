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

# Get the ceiling color
ceiling_bboxes = loc(image, "ceiling")
ceiling_color = None
if ceiling_bboxes:
    ceiling_color = _get_color(image, ceiling_bboxes[0])

# Get all remote controls
remote_bboxes = loc(image, "remote control")

# Find the closest remote control
closest_remote = _get_closest_object(image, "remote control")
closest_remote_color = _get_color(image, closest_remote)

# Find the furthest remote control by checking depths
furthest_remote = None
max_depth = -1
for remote_bbox in remote_bboxes:
    remote_depth = depth(image, remote_bbox)
    if remote_depth > max_depth:
        max_depth = remote_depth
        furthest_remote = remote_bbox

furthest_remote_color = _get_color(image, furthest_remote)

# Compare colors
if ceiling_color == closest_remote_color:
    final_result = "closest"
elif ceiling_color == furthest_remote_color:
    final_result = "furthest"
else:
    final_result = "closest"  # default if neither matches (shouldn't happen with binary options)


# WRITE NAMESPACE
import json

def is_serializable(obj):
    try:
        json.dumps(obj)
    except (TypeError, OverflowError):
        return False
    return True

serializable_globals = {k: v for k, v in globals().items() if is_serializable(v)}

with open(r"D:\3D_Spatial_Agent\results\A\2026-09-18_16-05-43\program_execution\image_14443.893_00002504.jpg_question_107/exec_env/result.json", "w+") as result_file:
    json.dump(serializable_globals, result_file)
        