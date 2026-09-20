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

# Find the bedside table
bedside_tables = loc(image, "bedside table")
bedside_table_bbox = bedside_tables[0]  # assume first one

# Find all cabinets and get the left-most one
cabinets = loc(image, "cabinet")
# Left-most means smallest xmin
left_most_cabinet_bbox = min(cabinets, key=lambda bbox: bbox[0])

# Get 2D heights and depths for both objects
_, bedside_table_2d_height = get_2D_object_size(image, bedside_table_bbox)
bedside_table_depth = depth(image, bedside_table_bbox)

_, cabinet_2d_height = get_2D_object_size(image, left_most_cabinet_bbox)
cabinet_depth = depth(image, left_most_cabinet_bbox)

# Calculate 3D heights (2D height * depth)
bedside_table_3d_height = bedside_table_2d_height * bedside_table_depth
cabinet_3d_height = cabinet_2d_height * cabinet_depth

# Combined 3D height is 3m, so we can find the scale factor
combined_3d_height = bedside_table_3d_height + cabinet_3d_height
scale_factor = 3.0 / combined_3d_height

# Find the bed
beds = loc(image, "bed")
bed_bbox = beds[0]  # assume first bed

# Get 2D width and depth of the bed
bed_2d_width, _ = get_2D_object_size(image, bed_bbox)
bed_depth = depth(image, bed_bbox)

# Calculate 3D width of the bed
bed_3d_width = bed_2d_width * bed_depth * scale_factor

final_result = bed_3d_width


# WRITE NAMESPACE
import json

def is_serializable(obj):
    try:
        json.dumps(obj)
    except (TypeError, OverflowError):
        return False
    return True

serializable_globals = {k: v for k, v in globals().items() if is_serializable(v)}

with open(r"D:\3D_Spatial_Agent\results\A\2026-09-18_16-05-43\program_execution\image_114979.918_00002498.jpg_question_95/exec_env/result.json", "w+") as result_file:
    json.dump(serializable_globals, result_file)
        