import math
import pandas as pd
import folium
import os
import webbrowser
import time
import random
import keyboard

# ------------------------
# CV-Based Buoy Heading Correction Functions
# ------------------------

# Estimate the Z-distance from the camera to a buoy based on its pixel height and camera vertical FOV
def buoy_distance_cm(observed_height_px, real_height_cm, frame_height_px, vertical_fov_deg):
    fov_rad = math.radians(vertical_fov_deg)
    focal_length_px = (frame_height_px / 2) / math.tan(fov_rad / 2)
    return (real_height_cm * focal_length_px) / observed_height_px

# Calculate the lateral offset of a buoy from the center of the frame in cm
def lateral_offset_cm(bbox_x, bbox_w, frame_width_px, horizontal_fov_deg, distance_cm):
    fov_rad = math.radians(horizontal_fov_deg)
    frame_width_cm = 2 * distance_cm * math.tan(fov_rad / 2)
    cm_per_pixel = frame_width_cm / frame_width_px
    buoy_center_x = bbox_x + (bbox_w / 2)
    frame_center_x = frame_width_px / 2
    offset_px = buoy_center_x - frame_center_x
    return offset_px * cm_per_pixel

# Compute the missing FOV (vertical or horizontal) based on known FOV and image resolution
def calculate_missing_fov(known_fov_deg, resolution_width, resolution_height, known_axis='horizontal'):
    aspect_ratio = resolution_width / resolution_height
    known_fov_rad = math.radians(known_fov_deg)

    if known_axis == 'horizontal':
        missing_fov_rad = 2 * math.atan(math.tan(known_fov_rad / 2) / aspect_ratio)
        missing_axis = 'vertical'
    elif known_axis == 'vertical':
        missing_fov_rad = 2 * math.atan(math.tan(known_fov_rad / 2) * aspect_ratio)
        missing_axis = 'horizontal'
    else:
        raise ValueError("known_axis must be either 'horizontal' or 'vertical'")

    missing_fov_deg = math.degrees(missing_fov_rad)
    return missing_axis, missing_fov_deg

# Decide the heading adjustment angle to avoid a buoy depending on its type and position
def heading_adjustment_from_buoy(offset_cm, distance_cm, buoy_type, safe_margin_cm=100):
    if buoy_type in ['red', 'west'] and offset_cm >= 0:
        corrected_offset = offset_cm + safe_margin_cm
        return math.degrees(math.atan2(corrected_offset, distance_cm))
    elif buoy_type in ['green', 'east'] and offset_cm <= 0:
        corrected_offset = offset_cm - safe_margin_cm
        return math.degrees(math.atan2(corrected_offset, distance_cm))
    else:
        return 0.0

# Use buoy detections to compute heading adjustments and update vessel state
def apply_cv_navigation(vessel, buoy_detections, frame_dims, fov, buoy_real_height_cm=60, safe_margin_cm=100):
    if not buoy_detections:
        return

    elif len(buoy_detections) == 1:
        b = buoy_detections[0]
        distance_cm = buoy_distance_cm(b['h'], buoy_real_height_cm, frame_dims['height'], fov['vertical'])
        offset_cm = lateral_offset_cm(b['x'], b['w'], frame_dims['width'], fov['horizontal'], distance_cm)
        heading_adjustment = heading_adjustment_from_buoy(offset_cm, distance_cm, b['type'], safe_margin_cm)

    elif len(buoy_detections) == 2:
        b1, b2 = sorted(buoy_detections, key=lambda b: b['x'])
        left_type = b1['type']
        right_type = b2['type']
        mid_x = (b1['x'] + b1['w'] / 2 + b2['x'] + b2['w'] / 2) / 2
        avg_height_px = (b1['h'] + b2['h']) / 2
        distance_cm = buoy_distance_cm(avg_height_px, buoy_real_height_cm, frame_dims['height'], fov['vertical'])
        midpoint_offset_cm = lateral_offset_cm(mid_x, 0, frame_dims['width'], fov['horizontal'], distance_cm)
        valid_corridors = [('red', 'green'), ('west', 'east')]
        if (left_type, right_type) in valid_corridors:
            heading_adjustment = math.degrees(math.atan2(midpoint_offset_cm, distance_cm))
        else:
            heading_adjustment = heading_adjustment_from_buoy(midpoint_offset_cm, distance_cm, left_type, safe_margin_cm)

    else:
        raise ValueError("Only 1 or 2 buoy detections are supported.")

    vessel.heading = (vessel.heading + heading_adjustment) % 360
    vessel.speed = 0.5 if heading_adjustment != 0 else 1.0

# Simulate buoy detections for testing (returns 0–2 detections per frame)
def read_dummy_cv_buoy_detections():
    detections = []
    if random.random() < 0.3:
        num_buoys = random.choice([1, 2])
        for _ in range(num_buoys):
            detections.append({
                'type': random.choice(['red', 'green', 'west', 'east']),
                'x': random.randint(200, 1000),
                'y': random.randint(200, 700),
                'w': random.randint(30, 80),
                'h': random.randint(100, 200)
            })
    return detections

# ------------------------
# Navigation System
# ------------------------

# Stores the vessel's current position, heading, and speed
class VesselState:
    def __init__(self, position, heading, speed):
        self.position = position
        self.heading = heading
        self.speed = speed

# Load waypoints from a CSV file into a list of (lat, lon) tuples
def load_waypoints_from_csv(filepath):
    df = pd.read_csv(filepath)
    return [(row['latitude'], row['longitude']) for index, row in df.iterrows()]

# Calculate the bearing (direction) between two GPS points
def calculate_bearing(lat1, lon1, lat2, lon2):
    phi1, lambda1, phi2, lambda2 = map(math.radians, [lat1, lon1, lat2, lon2])
    d_lambda = lambda2 - lambda1
    x = math.sin(d_lambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(d_lambda)
    theta = math.atan2(x, y)
    return (math.degrees(theta) + 360) % 360

# Calculate straight-line (Euclidean) distance between two 2D points
def distance(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])

# Adjust vessel heading toward the current target waypoint
def correct_heading_toward_waypoint(vessel, target_wp):
    vessel.heading = calculate_bearing(vessel.position[0], vessel.position[1], target_wp[0], target_wp[1])

# Generate and open an HTML map showing the waypoints and paths
def display_map(waypoints, filename='map.html'):
    fmap = folium.Map(location=waypoints[0], zoom_start=18)
    for i, point in enumerate(waypoints):
        folium.Marker(point, tooltip=f"Waypoint {i}").add_to(fmap)
        if i > 0:
            folium.PolyLine([waypoints[i-1], point], color='blue').add_to(fmap)
    fmap.save(filename)
    webbrowser.open('file://' + os.path.realpath(filename))

# ------------------------
# Main Simulation Loop
# ------------------------

waypoints = load_waypoints_from_csv('waypoints_task2_trondheim.csv')
display_map(waypoints)

vessel = VesselState(position=waypoints[0], heading=90, speed=1.0)
current_wp_index = 1

print("--- Starting NJORD 2025 Task 2 Simulation ---")

frame_dims = {'width': 1920, 'height': 1080}
fov = {'horizontal': 90, 'vertical': 58}

while current_wp_index < len(waypoints):
    if keyboard.is_pressed('q'):
        print("[INFO] Hotkey 'q' pressed. Exiting simulation.")
        break

    target_wp = waypoints[current_wp_index]
    cv_detections = read_dummy_cv_buoy_detections()

    if cv_detections:
        apply_cv_navigation(vessel, cv_detections, frame_dims, fov, buoy_real_height_cm=60)
    else:
        correct_heading_toward_waypoint(vessel, target_wp)
        vessel.speed = 1.0

    if distance(vessel.position, target_wp) < 0.0002:
        print(f"[STATUS] Reached waypoint {current_wp_index}: {target_wp}")
        vessel.position = target_wp
        vessel.speed = 1.0
        correct_heading_toward_waypoint(vessel, target_wp)
        current_wp_index += 1

    print(f"[INFO] Heading: {vessel.heading:.2f}, Speed: {vessel.speed:.2f}")
    time.sleep(0.5)

print("--- Simulation Complete. Vessel reached final waypoint. ---")
