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

# Find the table
table_bboxes = loc(image, "table")
table_bbox = table_bboxes[0]

# Find the brown dresser
dresser_bboxes = loc(image, "brown dresser")
dresser_bbox = dresser_bboxes[0]

# Get 2D heights
table_width, table_height_2d = get_2D_object_size(image, table_bbox)
dresser_width, dresser_height_2d = get_2D_object_size(image, dresser_bbox)

# Get depths
table_depth = depth(image, table_bbox)
dresser_depth = depth(image, dresser_bbox)

# Calculate 3D heights (2D height * depth)
table_height_3d = table_height_2d * table_depth
dresser_height_3d = dresser_height_2d * dresser_depth

# The table is 1.8m tall, so scale the dresser's height accordingly
# dresser_real_height / table_real_height = dresser_height_3d / table_height_3d
dresser_real_height = (dresser_height_3d / table_height_3d) * 1.8

final_result = dresser_real_height


# WRITE NAMESPACE
import json

def is_serializable(obj):
    try:
        json.dumps(obj)
    except (TypeError, OverflowError):
        return False
    return True

serializable_globals = {k: v for k, v in globals().items() if is_serializable(v)}

with open(r"D:\3D_Spatial_Agent\results\A\2026-09-18_16-05-43\program_execution\image_74514.833_00005019.jpg_question_113/exec_env/result.json", "w+") as result_file:
    json.dump(serializable_globals, result_file)
        