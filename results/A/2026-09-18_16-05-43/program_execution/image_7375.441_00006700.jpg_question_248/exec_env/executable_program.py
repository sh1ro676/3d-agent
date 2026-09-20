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

# Find the stovetop
stovetop_bboxes = loc(image, "stovetop")

# Find the oven
oven_bboxes = loc(image, "oven")

# Get the stovetop (assuming there's one)
stovetop_bbox = stovetop_bboxes[0]

# Get the oven (assuming there's one)
oven_bbox = oven_bboxes[0]

# Get 2D dimensions of stovetop
stovetop_width_2d, stovetop_height_2d = get_2D_object_size(image, stovetop_bbox)

# Get depth of stovetop
stovetop_depth = depth(image, stovetop_bbox)

# Calculate 3D length of stovetop (length is along z-axis, which is depth)
# Wait - length is along z-axis, but we need to think about this differently
# The stovetop's length in 3D would be its depth (z-axis dimension)
# But we only have 2D width and height from the image
# Actually, for a stovetop, the "length" might refer to its horizontal extent
# Let me reconsider: width = x-axis, height = y-axis, length = z-axis
# The 3D length would be the depth of the object

# Actually, looking at this more carefully:
# 3D size = 2D size * depth
# For length (z-axis), we need the depth itself as the 3D length
# But wait - the depth function gives us distance from camera, not the object's z-dimension

# Let me reconsider - for the stovetop, its length in 3D space
# The stovetop's 2D width in the image corresponds to its actual width
# The stovetop's 2D height in the image corresponds to its actual length (depth into the scene)
# So 3D length = 2D height * depth

stovetop_3d_length = stovetop_height_2d * stovetop_depth

# Get 2D dimensions of oven
oven_width_2d, oven_height_2d = get_2D_object_size(image, oven_bbox)

# Get depth of oven
oven_depth = depth(image, oven_bbox)

# Calculate 3D height of oven
oven_3d_height = oven_height_2d * oven_depth

# Calculate the ratio
final_result = stovetop_3d_length / oven_3d_height


# WRITE NAMESPACE
import json

def is_serializable(obj):
    try:
        json.dumps(obj)
    except (TypeError, OverflowError):
        return False
    return True

serializable_globals = {k: v for k, v in globals().items() if is_serializable(v)}

with open(r"D:\3D_Spatial_Agent\results\A\2026-09-18_16-05-43\program_execution\image_7375.441_00006700.jpg_question_248/exec_env/result.json", "w+") as result_file:
    json.dump(serializable_globals, result_file)
        