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

# Locate the washing machine
washing_machines = loc(image, "washing machine")
washing_machine_bbox = washing_machines[0]

# Locate the cabinet
cabinets = loc(image, "cabinet")

# Find the cabinet above the washing machine
# The cabinet above should have a smaller y-coordinate (higher in image) than the washing machine
washing_machine_ymin = washing_machine_bbox[1]
cabinet_bbox = None
for cab in cabinets:
    if cab[1] < washing_machine_ymin:  # cabinet is above washing machine
        cabinet_bbox = cab
        break

# Get 2D heights and depths
wm_width, wm_height_2d = get_2D_object_size(image, washing_machine_bbox)
wm_depth = depth(image, washing_machine_bbox)

cab_width, cab_height_2d = get_2D_object_size(image, cabinet_bbox)
cab_depth = depth(image, cabinet_bbox)

# Calculate 3D heights (2D height * depth)
wm_height_3d = wm_height_2d * wm_depth
cab_height_3d = cab_height_2d * cab_depth

# The washing machine's new 3D height is 1 foot
# Scale factor = new_height / original_height
scale_factor = 1.0 / wm_height_3d

# Apply scale factor to cabinet's 3D height
final_result = cab_height_3d * scale_factor


# WRITE NAMESPACE
import json

def is_serializable(obj):
    try:
        json.dumps(obj)
    except (TypeError, OverflowError):
        return False
    return True

serializable_globals = {k: v for k, v in globals().items() if is_serializable(v)}

with open(r"D:\3D_Spatial_Agent\results\A\2026-09-18_16-05-43\program_execution\image_164039.451_00000637.jpg_question_235/exec_env/result.json", "w+") as result_file:
    json.dump(serializable_globals, result_file)
        