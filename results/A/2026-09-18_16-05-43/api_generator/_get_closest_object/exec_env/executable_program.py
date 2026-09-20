import math
def loc(image, object_prompt):

	return [[25, 25, 50, 50]]

import math
def vqa(image, question, bbox):

	return ""

import math
def depth(image, bbox):

	return 1.0

import math
def same_object(image, bbox1, bbox2):

	return False

import math
def get_2D_object_size(image, bbox):

	return (50, 50)

def _count_objects(image, object_prompt):
    
    bboxes = loc(image, object_prompt)
    return len(bboxes)
    


# PROGRAM STARTS HERE

bboxes = loc(image, object_prompt)
if not bboxes:
    final_result = None

closest_bbox = None
min_depth = float('inf')

for bbox in bboxes:
    d = depth(image, bbox)
    if d < min_depth:
        min_depth = d
        closest_bbox = bbox

final_result = closest_bbox


# WRITE NAMESPACE
import json
def is_serializable(obj):
    try:
        json.dumps(obj)
    except (TypeError, OverflowError):
        return False
    return True

serializable_globals = {k: v for k, v in globals().items() if is_serializable(v)}

with open(r"D:\3D_Spatial_Agent\results\A\2026-09-18_16-05-43\api_generator\_get_closest_object\exec_env/result.json", "w+") as result_file:
    json.dump(serializable_globals, result_file)
        