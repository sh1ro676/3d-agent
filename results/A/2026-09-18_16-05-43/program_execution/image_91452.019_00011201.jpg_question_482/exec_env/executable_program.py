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

# Locate the yellow flower vase
vase_bboxes = loc(image, "yellow flower vase")

# Locate the brown door
door_bboxes = loc(image, "brown door")

# Check if both objects were found
if len(vase_bboxes) > 0 and len(door_bboxes) > 0:
    # Get the first vase and door (assuming there's one of each)
    vase_bbox = vase_bboxes[0]
    door_bbox = door_bboxes[0]
    
    # Get the depth of the vase
    vase_depth = depth(image, vase_bbox)
    
    # Get the depth of the door
    door_depth = depth(image, door_bbox)
    
    # Check if the vase is closer to the camera than the door
    # Closer means smaller depth value
    if vase_depth < door_depth:
        final_result = "yes"
    else:
        final_result = "no"
else:
    # If either object is not found, we cannot make the comparison
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

with open(r"D:\3D_Spatial_Agent\results\A\2026-09-18_16-05-43\program_execution\image_91452.019_00011201.jpg_question_482/exec_env/result.json", "w+") as result_file:
    json.dump(serializable_globals, result_file)
        