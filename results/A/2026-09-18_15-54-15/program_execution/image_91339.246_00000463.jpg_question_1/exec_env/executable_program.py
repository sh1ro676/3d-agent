import math

# PROGRAM STARTS HERE

coffee_tables = loc(image, "coffee table")
coffee_table_bbox = coffee_tables[0]
ct_width_2d, ct_height_2d = get_2D_object_size(image, coffee_table_bbox)
scale = 2.0 / ct_width_2d
sofas = loc(image, "sofa")
ct_center_x = (coffee_table_bbox[0] + coffee_table_bbox[2]) / 2.0
sofa_to_right = None
for sofa_bbox in sofas:
    sofa_center_x = (sofa_bbox[0] + sofa_bbox[2]) / 2.0
    if sofa_center_x > ct_center_x:
        sofa_to_right = sofa_bbox
        break
sofa_width_2d, sofa_height_2d = get_2D_object_size(image, sofa_to_right)
sofa_length_m = sofa_width_2d * scale
final_result = sofa_length_m


# WRITE NAMESPACE
import json

def is_serializable(obj):
    try:
        json.dumps(obj)
    except (TypeError, OverflowError):
        return False
    return True

serializable_globals = {k: v for k, v in globals().items() if is_serializable(v)}

with open("D:\3D_Spatial_Agent\results\A\2026-09-18_15-54-15\program_execution\image_91339.246_00000463.jpg_question_1/exec_env/result.json", "w+") as result_file:
    json.dump(serializable_globals, result_file)
        