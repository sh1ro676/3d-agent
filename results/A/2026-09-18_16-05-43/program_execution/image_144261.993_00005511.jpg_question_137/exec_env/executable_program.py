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

# Locate the central gray couch
couches = loc(image, "couch")

# Find the gray couch (central one)
gray_couch_bbox = None
for couch_bbox in couches:
    # Check if this couch is gray
    color = _get_color(image, couch_bbox)
    if color.lower() == "gray":
        # Check if it's central by asking VQA
        is_central = vqa(image, "Is this couch in the center of the image?", couch_bbox)
        if is_central.lower() == "yes":
            gray_couch_bbox = couch_bbox
            break

# If we didn't find a central gray couch, try just finding any gray couch
if gray_couch_bbox is None:
    for couch_bbox in couches:
        color = _get_color(image, couch_bbox)
        if color.lower() == "gray":
            gray_couch_bbox = couch_bbox
            break

# Now find all pillows
pillows = loc(image, "pillow")

# Count pillows on the gray couch
pillow_count = 0
for pillow_bbox in pillows:
    # Check if this pillow is on the gray couch
    is_on_couch = vqa(image, "Is this pillow on the gray couch?", pillow_bbox)
    if is_on_couch.lower() == "yes":
        pillow_count += 1

final_result = pillow_count


# WRITE NAMESPACE
import json

def is_serializable(obj):
    try:
        json.dumps(obj)
    except (TypeError, OverflowError):
        return False
    return True

serializable_globals = {k: v for k, v in globals().items() if is_serializable(v)}

with open(r"D:\3D_Spatial_Agent\results\A\2026-09-18_16-05-43\program_execution\image_144261.993_00005511.jpg_question_137/exec_env/result.json", "w+") as result_file:
    json.dump(serializable_globals, result_file)
        