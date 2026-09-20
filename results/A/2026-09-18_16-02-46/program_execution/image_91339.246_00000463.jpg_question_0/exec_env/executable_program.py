import math
def _get_object_height(image, bbox):
    
    width, height = get_2D_object_size(image, bbox)
    return height
    
def _get_object_width(image, bbox):
    
    width, height = get_2D_object_size(image, bbox)
    return width
    
    
    depth_value = depth(image, bbox)
    width, height = get_2D_object_size(image, bbox)
    # Use the ratio of pixel_length to the object's 2D size, scaled by depth
    # Since 3D size = 2D size * depth, a pixel length converts to meters as:
    # meters = pixel_length * depth / focal_length_equivalent
    # We can estimate using the object's known 2D size and depth:
    # 3D_size = 2D_size * depth, so 1 pixel = depth (in meters per pixel at that depth)
    # Actually, meters_per_pixel = depth / focal_length, but we can approximate using object size
    # A common approach: meters = pixel_length * depth / (2D_size_in_pixels) * (3D_size_in_meters / 3D_size_in_meters)
    # Simpler: use the object's 2D size and depth to get scale
    # scale = depth (meters) per pixel at that depth is depth / focal_length
    # We don't have focal length, but we can use: 3D_size = 2D_size * depth, so meters_per_pixel = depth
    # Wait, that's not right dimensionally. Let's think:
    # If object is W pixels wide and D meters deep, its 3D width = W * D (per the definition given)
    # So 1 pixel = D meters (in 3D space at that depth)
    # Therefore pixel_length meters = pixel_length * depth
    return pixel_length * depth_value
    

# PROGRAM STARTS HERE

# Locate the fireplace
fireplaces = loc(image, "fireplace")
fireplace_bbox = fireplaces[0]
fireplace_height = _get_object_height(image, fireplace_bbox)

# Locate the coffee table
coffee_tables = loc(image, "coffee table")
coffee_table_bbox = coffee_tables[0]
coffee_table_height = _get_object_height(image, coffee_table_bbox)

# Locate all sofas
sofas = loc(image, "sofa")

# Find the sofa to the right of the coffee table
# Get coffee table center x-coordinate
coffee_table_center_x = (coffee_table_bbox[0] + coffee_table_bbox[2]) / 2

sofa_to_right_bbox = None
for sofa_bbox in sofas:
    sofa_center_x = (sofa_bbox[0] + sofa_bbox[2]) / 2
    if sofa_center_x > coffee_table_center_x:
        sofa_to_right_bbox = sofa_bbox
        break

sofa_height = _get_object_height(image, sofa_to_right_bbox)

# Calculate the ratio
combined_height = coffee_table_height + sofa_height
final_result = fireplace_height / combined_height


# WRITE NAMESPACE
import json

def is_serializable(obj):
    try:
        json.dumps(obj)
    except (TypeError, OverflowError):
        return False
    return True

serializable_globals = {k: v for k, v in globals().items() if is_serializable(v)}

with open(r"D:\3D_Spatial_Agent\results\A\2026-09-18_16-02-46\program_execution\image_91339.246_00000463.jpg_question_0/exec_env/result.json", "w+") as result_file:
    json.dump(serializable_globals, result_file)
        