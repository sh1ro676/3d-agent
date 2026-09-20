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

# Find the TV
tv_bboxes = loc(image, "TV")
tv_bbox = tv_bboxes[0]  # Assume first TV

# Find the TV stand
tv_stand_bboxes = loc(image, "TV stand")
tv_stand_bbox = tv_stand_bboxes[0]  # Assume first TV stand

# Find the brick-fireplace
fireplace_bboxes = loc(image, "brick-fireplace")
fireplace_bbox = fireplace_bboxes[0]  # Assume first brick-fireplace

# Get 2D heights and depths for TV and TV stand
tv_width_2d, tv_height_2d = get_2D_object_size(image, tv_bbox)
tv_depth = depth(image, tv_bbox)
tv_height_3d = tv_height_2d * tv_depth

tv_stand_width_2d, tv_stand_height_2d = get_2D_object_size(image, tv_stand_bbox)
tv_stand_depth = depth(image, tv_stand_bbox)
tv_stand_height_3d = tv_stand_height_2d * tv_stand_depth

# Get 2D width and depth for fireplace
fireplace_width_2d, fireplace_height_2d = get_2D_object_size(image, fireplace_bbox)
fireplace_depth = depth(image, fireplace_bbox)
fireplace_width_3d = fireplace_width_2d * fireplace_depth

# Calculate combined height of TV and TV stand
combined_height = tv_height_3d + tv_stand_height_3d

# Compare combined height with fireplace width
final_result = combined_height > fireplace_width_3d


# WRITE NAMESPACE
import json

def is_serializable(obj):
    try:
        json.dumps(obj)
    except (TypeError, OverflowError):
        return False
    return True

serializable_globals = {k: v for k, v in globals().items() if is_serializable(v)}

with open(r"D:\3D_Spatial_Agent\results\A\2026-09-18_16-05-43\program_execution\image_45899.622_00000877.jpg_question_175/exec_env/result.json", "w+") as result_file:
    json.dump(serializable_globals, result_file)
        