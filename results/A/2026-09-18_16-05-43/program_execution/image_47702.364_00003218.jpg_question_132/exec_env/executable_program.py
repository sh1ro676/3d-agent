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

# Locate the fruit basket
fruit_baskets = loc(image, "fruit basket")

# Initialize counter
fruit_count = 0

# If we found a fruit basket, check each fruit
if len(fruit_baskets) > 0:
    basket_bbox = fruit_baskets[0]
    
    # Locate all fruits
    fruits = loc(image, "fruit")
    
    # For each fruit, check if it's in the fruit basket
    for fruit_bbox in fruits:
        answer = vqa(image, "Is this fruit in the fruit basket?", fruit_bbox)
        if answer.lower() == "yes":
            fruit_count += 1

final_result = fruit_count


# WRITE NAMESPACE
import json

def is_serializable(obj):
    try:
        json.dumps(obj)
    except (TypeError, OverflowError):
        return False
    return True

serializable_globals = {k: v for k, v in globals().items() if is_serializable(v)}

with open(r"D:\3D_Spatial_Agent\results\A\2026-09-18_16-05-43\program_execution\image_47702.364_00003218.jpg_question_132/exec_env/result.json", "w+") as result_file:
    json.dump(serializable_globals, result_file)
        