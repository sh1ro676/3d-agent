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

# Locate the coffee table
coffee_tables = loc(image, "coffee table")
coffee_table_bbox = coffee_tables[0]

# Get the 2D width (length along x-axis) of the coffee table in pixels
coffee_table_width_pixels = _get_object_width(image, coffee_table_bbox)

# The coffee table is 2m long, so calculate pixels per meter
pixels_per_meter = coffee_table_width_pixels / 2.0

# Locate all sofas
sofas = loc(image, "sofa")

# Find the sofa to the right of the coffee table
# The sofa to the right should have its xmin greater than the coffee table's xmax
sofa_to_right_bbox = None
for sofa_bbox in sofas:
    if sofa_bbox[0] > coffee_table_bbox[2]:  # sofa's xmin > coffee table's xmax
        sofa_to_right_bbox = sofa_bbox
        break

# Get the 2D width (length along x-axis) of the sofa in pixels
sofa_width_pixels = _get_object_width(image, sofa_to_right_bbox)

# Convert sofa width from pixels to meters using the scale
sofa_length_meters = sofa_width_pixels / pixels_per_meter

final_result = sofa_length_meters


# WRITE NAMESPACE
import json

def is_serializable(obj):
    try:
        json.dumps(obj)
    except (TypeError, OverflowError):
        return False
    return True

serializable_globals = {k: v for k, v in globals().items() if is_serializable(v)}

with open(r"D:\3D_Spatial_Agent\results\A\2026-09-18_16-02-46\program_execution\image_91339.246_00000463.jpg_question_1/exec_env/result.json", "w+") as result_file:
    json.dump(serializable_globals, result_file)
        