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

# Find the red armchair
armchairs = loc(image, "armchair")
red_armchair_bbox = None

for armchair_bbox in armchairs:
    color = _get_color(image, armchair_bbox)
    if color.lower() == "red":
        red_armchair_bbox = armchair_bbox
        break

# If no red armchair found, try just "red armchair" directly
if red_armchair_bbox is None:
    red_armchairs = loc(image, "red armchair")
    if red_armchairs:
        red_armchair_bbox = red_armchairs[0]

# Get the 3D height of the red armchair
armchair_2d_size = get_2D_object_size(image, red_armchair_bbox)
armchair_2d_height = armchair_2d_size[1]
armchair_depth = depth(image, red_armchair_bbox)
armchair_3d_height = armchair_2d_height * armchair_depth

# Find the table below the TV
tables = loc(image, "table")
tv_list = loc(image, "tv")

table_below_tv_bbox = None

if tv_list and tables:
    tv_bbox = tv_list[0]
    tv_bottom = tv_bbox[3]
    
    for table_bbox in tables:
        table_top = table_bbox[1]
        if table_top > tv_bottom:
            table_below_tv_bbox = table_bbox
            break

# If no table found below TV, just use the first table
if table_below_tv_bbox is None and tables:
    table_below_tv_bbox = tables[0]

# Get the 3D length of the table below the TV
# Length is along the z-axis (depth direction)
table_depth = depth(image, table_below_tv_bbox)
table_2d_size = get_2D_object_size(image, table_below_tv_bbox)
table_2d_width = table_2d_size[0]

# The length of the table in 3D space (along z-axis)
table_3d_length = table_2d_width * table_depth

# Calculate how many armchairs needed
final_result = table_3d_length / armchair_3d_height


# WRITE NAMESPACE
import json

def is_serializable(obj):
    try:
        json.dumps(obj)
    except (TypeError, OverflowError):
        return False
    return True

serializable_globals = {k: v for k, v in globals().items() if is_serializable(v)}

with open(r"D:\3D_Spatial_Agent\results\A\2026-09-18_16-05-43\program_execution\image_100592.766_00001489.jpg_question_387/exec_env/result.json", "w+") as result_file:
    json.dump(serializable_globals, result_file)
        