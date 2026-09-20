import math

# PROGRAM STARTS HERE

fireplace_bboxes = loc(image, "fireplace")
fireplace_bbox = fireplace_bboxes[0]

coffee_table_bboxes = loc(image, "coffee table")
coffee_table_bbox = coffee_table_bboxes[0]

sofa_bboxes = loc(image, "sofa")

coffee_table_center_x = (coffee_table_bbox[0] + coffee_table_bbox[2]) / 2

sofa_to_right_bbox = None
for sofa_bbox in sofa_bboxes:
    sofa_center_x = (sofa_bbox[0] + sofa_bbox[2]) / 2
    if sofa_center_x > coffee_table_center_x:
        sofa_to_right_bbox = sofa_bbox
        break

fireplace_size = get_2D_object_size(image, fireplace_bbox)
coffee_table_size = get_2D_object_size(image, coffee_table_bbox)
sofa_size = get_2D_object_size(image, sofa_to_right_bbox)

fireplace_depth = depth(image, fireplace_bbox)
coffee_table_depth = depth(image, coffee_table_bbox)
sofa_depth = depth(image, sofa_to_right_bbox)

fireplace_height_3d = fireplace_size[1] * fireplace_depth
coffee_table_height_3d = coffee_table_size[1] * coffee_table_depth
sofa_height_3d = sofa_size[1] * sofa_depth

combined_height = coffee_table_height_3d + sofa_height_3d
final_result = fireplace_height_3d / combined_height


# WRITE NAMESPACE
import json

def is_serializable(obj):
    try:
        json.dumps(obj)
    except (TypeError, OverflowError):
        return False
    return True

serializable_globals = {k: v for k, v in globals().items() if is_serializable(v)}

with open("D:\3D_Spatial_Agent\results\A\2026-09-18_15-54-15\program_execution\image_91339.246_00000463.jpg_question_0/exec_env/result.json", "w+") as result_file:
    json.dump(serializable_globals, result_file)
        