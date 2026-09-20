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

# Find TV and TV stand
tv_bboxes = loc(image, "TV")
tv_stand_bboxes = loc(image, "TV stand")

# Get the first TV and TV stand
tv_bbox = tv_bboxes[0]
tv_stand_bbox = tv_stand_bboxes[0]

# Get 2D heights
_, tv_height_2d = get_2D_object_size(image, tv_bbox)
_, tv_stand_height_2d = get_2D_object_size(image, tv_stand_bbox)

# Get depths
tv_depth = depth(image, tv_bbox)
tv_stand_depth = depth(image, tv_stand_bbox)

# Calculate 3D heights
tv_height_3d = tv_height_2d * tv_depth
tv_stand_height_3d = tv_stand_height_2d * tv_stand_depth

# Combined 3D height corresponds to 2m
combined_3d_height = tv_height_3d + tv_stand_height_3d

# Find the brown and silver coffee table
coffee_table_bboxes = loc(image, "brown and silver coffee table")
coffee_table_bbox = coffee_table_bboxes[0]

# Get 2D width and depth of coffee table
ct_width_2d, _ = get_2D_object_size(image, coffee_table_bbox)
ct_depth = depth(image, coffee_table_bbox)

# Calculate 3D width of coffee table
ct_width_3d = ct_width_2d * ct_depth

# Scale to real world: combined_3d_height corresponds to 2m
final_result = (ct_width_3d / combined_3d_height) * 2.0


# WRITE NAMESPACE
import json

def is_serializable(obj):
    try:
        json.dumps(obj)
    except (TypeError, OverflowError):
        return False
    return True

serializable_globals = {k: v for k, v in globals().items() if is_serializable(v)}

with open(r"D:\3D_Spatial_Agent\results\A\2026-09-18_16-05-43\program_execution\image_186909.798_00006580.jpg_question_92/exec_env/result.json", "w+") as result_file:
    json.dump(serializable_globals, result_file)
        