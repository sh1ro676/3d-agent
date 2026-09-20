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

# Find all cabinets
cabinets = loc(image, "cabinet")

# Find the leftmost cabinet (smallest xmin)
leftmost_cabinet = min(cabinets, key=lambda bbox: bbox[0])

# Find the cabinet directly adjacent to the leftmost cabinet
# This should be the cabinet with the next smallest xmin that is not the leftmost
other_cabinets = [cab for cab in cabinets if not same_object(image, leftmost_cabinet, cab)]
adjacent_cabinet = min(other_cabinets, key=lambda bbox: bbox[0])

# Get the 2D dimensions of both cabinets
leftmost_width_2d, leftmost_height_2d = get_2D_object_size(image, leftmost_cabinet)
adjacent_width_2d, adjacent_height_2d = get_2D_object_size(image, adjacent_cabinet)

# The leftmost cabinet is 1.9m wide
# Use this to find the scale factor
# Real world width = 2D width * depth * scale_factor
# We need to find the scale factor using the known width

# Get depths
leftmost_depth = depth(image, leftmost_cabinet)
adjacent_depth = depth(image, adjacent_cabinet)

# Calculate the real-world width of the leftmost cabinet using the formula
# real_world_width = 2D_width * depth * scale_factor
# We know real_world_width = 1.9m, so we can find scale_factor
# But we need to be careful about how the API works

# Actually, let's use the _get_real_world_width function
# We know the leftmost cabinet is 1.9m wide
# We can find its height using the aspect ratio

# The real-world width is 1.9m
# The 2D width is leftmost_width_2d
# The 2D height is leftmost_height_2d
# The real-world height should be proportional: real_height / real_width = 2D_height / 2D_width
# So real_height = real_width * (2D_height / 2D_width)

# But we need to account for depth differences
# Actually, for objects at the same depth, the ratio holds
# Let's assume they're at similar depths or use the proper formula

# Using the API function _get_real_world_width to find the scale
# We know the leftmost cabinet's real width is 1.9m
# We can find its real height by using the aspect ratio in 3D space

# Real world dimensions: width_3d = width_2d * depth * k, height_3d = height_2d * depth * k
# where k is some constant scale factor

# For leftmost cabinet: 1.9 = leftmost_width_2d * leftmost_depth * k
# So k = 1.9 / (leftmost_width_2d * leftmost_depth)

# For adjacent cabinet: height_3d = adjacent_height_2d * adjacent_depth * k
# height_3d = adjacent_height_2d * adjacent_depth * 1.9 / (leftmost_width_2d * leftmost_depth)

k = 1.9 / (leftmost_width_2d * leftmost_depth)
adjacent_height_3d = adjacent_height_2d * adjacent_depth * k

final_result = adjacent_height_3d


# WRITE NAMESPACE
import json

def is_serializable(obj):
    try:
        json.dumps(obj)
    except (TypeError, OverflowError):
        return False
    return True

serializable_globals = {k: v for k, v in globals().items() if is_serializable(v)}

with open(r"D:\3D_Spatial_Agent\results\A\2026-09-18_16-05-43\program_execution\image_65662.203_00000817.jpg_question_153/exec_env/result.json", "w+") as result_file:
    json.dump(serializable_globals, result_file)
        