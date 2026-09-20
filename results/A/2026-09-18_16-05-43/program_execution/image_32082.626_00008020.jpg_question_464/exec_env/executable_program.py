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

paintings = loc(image, "painting")
plants = loc(image, "plant")

num_paintings = len(paintings)
num_plants = len(plants)

if num_paintings > num_plants:
    final_result = "yes"
else:
    final_result = "no"


# WRITE NAMESPACE
import json

def is_serializable(obj):
    try:
        json.dumps(obj)
    except (TypeError, OverflowError):
        return False
    return True

serializable_globals = {k: v for k, v in globals().items() if is_serializable(v)}

with open(r"D:\3D_Spatial_Agent\results\A\2026-09-18_16-05-43\program_execution\image_32082.626_00008020.jpg_question_464/exec_env/result.json", "w+") as result_file:
    json.dump(serializable_globals, result_file)
        