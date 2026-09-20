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

# Find all sofas
sofas = loc(image, "sofa")

# Find the rightmost sofa (largest xmax)
rightmost_sofa = None
max_xmax = -1
for sofa in sofas:
    if sofa[2] > max_xmax:
        max_xmax = sofa[2]
        rightmost_sofa = sofa

# Find the circular table under the TV
# First find the TV
tv = loc(image, "TV")
tables = loc(image, "circular table")

# Find the table under the TV
circular_table = None
if tv and tables:
    tv_bbox = tv[0]
    # Find table that is below the TV (larger ymin than TV's ymax or overlapping horizontally)
    for table in tables:
        # Check if table is under the TV (table's ymin is greater than TV's ymin)
        if table[1] > tv_bbox[1]:
            circular_table = table
            break
    # If no table found under TV, just take the first one
    if circular_table is None and tables:
        circular_table = tables[0]

# Calculate volume of rightmost sofa
# Get 2D size and depth
sofa_width_2d, sofa_height_2d = get_2D_object_size(image, rightmost_sofa)
sofa_depth = depth(image, rightmost_sofa)

# 3D dimensions = 2D size * depth
sofa_width_3d = sofa_width_2d * sofa_depth
sofa_height_3d = sofa_height_2d * sofa_depth
sofa_length_3d = sofa_depth  # depth is the length in 3D

sofa_volume = sofa_width_3d * sofa_height_3d * sofa_length_3d

# Calculate volume of circular table
table_width_2d, table_height_2d = get_2D_object_size(image, circular_table)
table_depth = depth(image, circular_table)

# For a circular table, width = length (diameter), height is the height
table_diameter_3d = table_width_2d * table_depth
table_height_3d = table_height_2d * table_depth

# Volume of cylinder = π * r² * h = π * (d/2)² * h
import math
table_radius_3d = table_diameter_3d / 2
table_volume = math.pi * (table_radius_3d ** 2) * table_height_3d

# Calculate ratio
final_result = sofa_volume / table_volume


# WRITE NAMESPACE
import json

def is_serializable(obj):
    try:
        json.dumps(obj)
    except (TypeError, OverflowError):
        return False
    return True

serializable_globals = {k: v for k, v in globals().items() if is_serializable(v)}

with open(r"D:\3D_Spatial_Agent\results\A\2026-09-18_16-05-43\program_execution\image_183724.018_00000638.jpg_question_31/exec_env/result.json", "w+") as result_file:
    json.dump(serializable_globals, result_file)
        