import random
import math
import pandas as pd
import folium
import os
import keyboard
import webbrowser
import time

# ------------------------
# Vessel State Definition
# ------------------------

class VesselState:
    def __init__(self, position, heading, speed):
        self.position = position
        self.heading = heading
        self.speed = speed

# ------------------------
# Waypoint Management
# ------------------------

def load_waypoints_from_csv(filepath):
    df = pd.read_csv(filepath)
    return [(row['latitude'], row['longitude']) for index, row in df.iterrows()]

# ------------------------
# Helper Functions
# ------------------------

def calculate_bearing(lat1, lon1, lat2, lon2):
    phi1, lambda1, phi2, lambda2 = map(math.radians, [lat1, lon1, lat2, lon2])
    d_lambda = lambda2 - lambda1
    x = math.sin(d_lambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(d_lambda)
    theta = math.atan2(x, y)
    return (math.degrees(theta) + 360) % 360

def distance(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])

def correct_heading(vessel, target_wp):
    vessel.heading = calculate_bearing(vessel.position[0], vessel.position[1], target_wp[0], target_wp[1])

def display_map(waypoints, filename='map.html'):
    fmap = folium.Map(location=waypoints[0], zoom_start=18)
    for i, point in enumerate(waypoints):
        folium.Marker(point, tooltip=f"Waypoint {i}").add_to(fmap)
        if i > 0:
            folium.PolyLine([waypoints[i-1], point], color='blue').add_to(fmap)
    fmap.save(filename)
    webbrowser.open('file://' + os.path.realpath(filename))

# ------------------------
# Real-time Sensor Input Reading
# ------------------------

def read_real_sensor_inputs():
    sensor_inputs = []
    if random.random() < 0.2:
        sensor_inputs.append({'type': 'lidar', 'distance': random.uniform(1, 5), 'angle': random.choice([-45, 45])})
    if random.random() < 0.2:
        sensor_inputs.append({'type': 'danger_mark', 'angle': random.choice([-30, 30])})
    if random.random() < 0.2:
        sensor_inputs.append({'type': 'cardinal_mark', 'direction': random.choice(['north', 'south', 'east', 'west']), 'angle': random.choice([-20, 20]), 'distance': random.uniform(3, 10)})
    if random.random() < 0.2:
        sensor_inputs.append({'type': 'buoy', 'color': random.choice(['red', 'green'])})
    return sensor_inputs

# ------------------------
# Decision Functions
# ------------------------

def avoid_obstacle(vessel, obstacle_info):
    angle = obstacle_info['angle']
    if angle > 0:
        print("[ACTION] Obstacle detected on right. Turning left and slowing down.")
        vessel.heading = (vessel.heading - 30) % 360
    else:
        print("[ACTION] Obstacle detected on left. Turning right and slowing down.")
        vessel.heading = (vessel.heading + 30) % 360
    vessel.speed = 0.5

def avoid_danger_mark(vessel, danger_info):
    relative_angle = danger_info.get('angle', 0)
    if relative_angle > 0:
        print("[ACTION] Danger mark on right. Turning left 45 degrees and slowing down.")
        vessel.heading = (vessel.heading - 45) % 360
    else:
        print("[ACTION] Danger mark on left. Turning right 45 degrees and slowing down.")
        vessel.heading = (vessel.heading + 45) % 360
    vessel.speed = 0.5

def navigate_cardinal_mark(vessel, mark_info):
    direction = mark_info['direction']
    relative_angle = mark_info.get('angle', 0)
    distance_to_mark = mark_info.get('distance', 0)
    print(f"[ACTION] Cardinal Mark ({direction}) detected at {relative_angle} degrees and {distance_to_mark:.2f} meters away.")
    if direction == "north":
        if relative_angle > 0:
            print("[ACTION] Move slightly left to pass north of the buoy.")
            vessel.heading = (vessel.heading - 15) % 360
        else:
            print("[ACTION] Move slightly right to pass north of the buoy.")
            vessel.heading = (vessel.heading + 15) % 360
    elif direction == "south":
        if relative_angle > 0:
            print("[ACTION] Move slightly left to pass south of the buoy.")
            vessel.heading = (vessel.heading - 15) % 360
        else:
            print("[ACTION] Move slightly right to pass south of the buoy.")
            vessel.heading = (vessel.heading + 15) % 360
    elif direction == "east":
        if relative_angle > 0:
            print("[ACTION] Move slightly left to pass east of the buoy.")
            vessel.heading = (vessel.heading - 15) % 360
        else:
            print("[ACTION] Move slightly right to pass east of the buoy.")
            vessel.heading = (vessel.heading + 15) % 360
    elif direction == "west":
        if relative_angle > 0:
            print("[ACTION] Move slightly left to pass west of the buoy.")
            vessel.heading = (vessel.heading - 15) % 360
        else:
            print("[ACTION] Move slightly right to pass west of the buoy.")
            vessel.heading = (vessel.heading + 15) % 360
    vessel.speed = 0.8

def navigate_buoy(vessel, buoy_info):
    color = buoy_info['color']
    if color == "green":
        print("[ACTION] Green buoy detected. Pass keeping it on the right. Slowing down.")
        vessel.heading = (vessel.heading + 15) % 360
    elif color == "red":
        print("[ACTION] Red buoy detected. Pass keeping it on the left. Slowing down.")
        vessel.heading = (vessel.heading - 15) % 360
    vessel.speed = 0.5

def move_to_waypoint(vessel, target_wp):
    bearing_to_wp = calculate_bearing(vessel.position[0], vessel.position[1], target_wp[0], target_wp[1])
    heading_error = (bearing_to_wp - vessel.heading + 360) % 360
    if heading_error < 15 or heading_error > 345:
        action = "Move Forward"
    elif heading_error < 180:
        action = "Turn Right"
    else:
        action = "Turn Left"
    print(f"[ACTION] {action} toward waypoint at {target_wp} (Heading error: {heading_error:.2f})")

# ------------------------
# Main Simulation Loop
# ------------------------

waypoints = load_waypoints_from_csv('waypoints_task2_trondheim.csv')

display_map(waypoints)

vessel = VesselState(position=waypoints[0], heading=90, speed=1.0)

current_wp_index = 1

print("--- Starting NJORD 2025 Task 2 Simulation ---")

while current_wp_index < len(waypoints):
    if keyboard.is_pressed('q'):
        print("[INFO] Hotkey 'q' pressed. Exiting simulation.")
        break

    target_wp = waypoints[current_wp_index]

    sensor_inputs = read_real_sensor_inputs()

    acted = False

    for input_data in sensor_inputs:
        if input_data['type'] == "lidar":
            avoid_obstacle(vessel, input_data)
            acted = True
            break
        elif input_data['type'] == "danger_mark":
            avoid_danger_mark(vessel, input_data)
            acted = True
            break
        elif input_data['type'] == "cardinal_mark":
            navigate_cardinal_mark(vessel, input_data)
            acted = True
            break
        elif input_data['type'] == "buoy":
            navigate_buoy(vessel, input_data)
            acted = True
            break

    if not acted:
        move_to_waypoint(vessel, target_wp)

    if distance(vessel.position, target_wp) < 0.0002:
        print(f"[STATUS] Reached waypoint {current_wp_index}: {target_wp}")
        vessel.position = target_wp
        vessel.speed = 1.0  # Reset speed to normal after reaching waypoint
        correct_heading(vessel, target_wp)
        current_wp_index += 1

    time.sleep(0.5)

print("--- Simulation Complete. Vessel reached final waypoint. ---")
