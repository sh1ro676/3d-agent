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

# Step 1: Locate the sofa to use as a scale reference
sofa_bboxes = loc(image, "sofa")

# We need the sofa's 2D height and its real-world height (0.4m) to establish scale
if len(sofa_bboxes) > 0:
    sofa_bbox = sofa_bboxes[0]
    sofa_2d_width, sofa_2d_height = get_2D_object_size(image, sofa_bbox)
    sofa_real_height = 0.4  # meters
    
    # Step 2: Locate the piano
    piano_bboxes = loc(image, "piano")
    
    # Step 3: Find the stool in front of the piano
    # First, locate all stools
    stool_bboxes = loc(image, "stool")
    
    # Find the stool that is in front of the piano
    # We'll check each stool to see if it's in front of the piano
    stool_in_front = None
    
    if len(piano_bboxes) > 0 and len(stool_bboxes) > 0:
        piano_bbox = piano_bboxes[0]
        
        for stool_bbox in stool_bboxes:
            # Ask if this stool is in front of the piano
            answer = vqa(image, "Is this stool in front of the piano?", stool_bbox)
            if answer.lower() == "yes":
                stool_in_front = stool_bbox
                break
        
        # Step 4: Calculate the real-world width of the stool
        if stool_in_front is not None:
            # Get the stool's 2D dimensions
            stool_2d_width, stool_2d_height = get_2D_object_size(image, stool_in_front)
            
            # Calculate the scale factor using the sofa
            # scale = real_height / 2d_height (this gives meters per pixel at the sofa's depth)
            # But we need to account for depth differences
            
            # Get depths
            sofa_depth = depth(image, sofa_bbox)
            stool_depth = depth(image, stool_in_front)
            
            # The real-world height of the sofa is 0.4m
            # 2D height * depth = 3D height (in some units)
            # So: 0.4 = sofa_2d_height * sofa_depth * k (where k is a constant)
            # Therefore: k = 0.4 / (sofa_2d_height * sofa_depth)
            
            k = sofa_real_height / (sofa_2d_height * sofa_depth)
            
            # Now for the stool:
            # stool_real_width = stool_2d_width * stool_depth * k
            stool_real_width = stool_2d_width * stool_depth * k
            
            final_result = stool_real_width
        else:
            final_result = 0
    else:
        final_result = 0
else:
    final_result = 0


# WRITE NAMESPACE
import json

def is_serializable(obj):
    try:
        json.dumps(obj)
    except (TypeError, OverflowError):
        return False
    return True

serializable_globals = {k: v for k, v in globals().items() if is_serializable(v)}

with open(r"D:\3D_Spatial_Agent\results\A\2026-09-18_16-05-43\program_execution\image_12214.439_00001777.jpg_question_34/exec_env/result.json", "w+") as result_file:
    json.dump(serializable_globals, result_file)
        