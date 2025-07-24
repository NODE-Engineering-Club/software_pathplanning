import numpy as np
import matplotlib.pyplot as plt
from pymavlink import mavutil
from rplidar import RPLidar
import time
from collections import deque
import csv
import json

# ----- EKF Class -----
class PositionEKF:
    def __init__(self):
        self.x = np.zeros(6)  # [px, py, pz, vx, vy, vz]
        self.P = np.eye(6)
        self.Q = np.eye(6) * 0.01  # smaller process noise to reduce drift
        self.R_gps = np.eye(3) * 3.0
        self.R_lidar = np.array([[0.3]])
        self.R_vel = np.eye(3) * 0.1  # velocity zero-update noise

    def predict(self, accel, dt):
        A = np.eye(6)
        A[0, 3] = A[1, 4] = A[2, 5] = dt

        B = np.zeros((6, 3))
        B[3, 0] = B[4, 1] = B[5, 2] = dt

        self.x = A @ self.x + B @ accel
        self.P = A @ self.P @ A.T + self.Q

    def update_gps(self, gps_pos):
        H = np.zeros((3, 6))
        H[0, 0] = H[1, 1] = H[2, 2] = 1
        y = gps_pos - H @ self.x
        S = H @ self.P @ H.T + self.R_gps
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = (np.eye(6) - K @ H) @ self.P

    def update_lidar(self, z_measured):
        H = np.zeros((1, 6))
        H[0, 2] = 1
        y = np.array([[z_measured]]) - H @ self.x
        S = H @ self.P @ H.T + self.R_lidar
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x += (K @ y).flatten()
        self.P = (np.eye(6) - K @ H) @ self.P

    def zero_velocity_update(self):
        H_vel = np.zeros((3, 6))
        H_vel[0, 3] = 1
        H_vel[1, 4] = 1
        H_vel[2, 5] = 1
        z_vel = np.zeros(3)
        y = z_vel - H_vel @ self.x
        S = H_vel @ self.P @ H_vel.T + self.R_vel
        K = self.P @ H_vel.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = (np.eye(6) - K @ H_vel) @ self.P

# ----- LiDAR Reader -----
def read_lidar(scan_iter):
    try:
        scan = next(scan_iter)
        distances = [m[2] for m in scan if 50 < m[2] < 3000]
        if distances:
            return True, np.mean(distances) / 1000.0  # mm → m
    except:
        pass
    return False, 0.0

# ----- Yaw Wrapping Helper -----
def wrap_angle_deg(angle):
    return (angle + 180) % 360 - 180

# ----- Serial Connections -----
master = mavutil.mavlink_connection('COM4', baud=57600)
master.wait_heartbeat()
print("✅ MAVLink connected (COM4)")

lidar = RPLidar('COM3')
scan_iter = lidar.iter_scans()
print("✅ RPLidar connected (COM3)")

# ----- EKF Initialization -----
ekf = PositionEKF()
last_time = time.time()

# ----- CSV Logging -----
csv_file = open("fusion_log.csv", mode="w", newline="")
csv_writer = csv.DictWriter(csv_file, fieldnames=[
    "timestamp", "x", "y", "z", "vx", "vy", "vz", "yaw", "speed", "lidar_z"
])
csv_writer.writeheader()

# ----- JSON Logging -----
json_file = open("fusion_log.json", mode="w")
json_file.write("[\n")  # start JSON array
first_json_entry = True

# ----- Plot Setup -----
z_history = deque(maxlen=100)
speed_history = deque(maxlen=100)
lidar_history = deque(maxlen=100)
yaw_history = deque(maxlen=100)

plt.ion()
fig, axs = plt.subplots(4, 1, figsize=(10, 12))
fig.suptitle("Sensor Fusion Live Data")

line_alt, = axs[0].plot([], [], label="EKF Altitude (m)", color='blue')
line_lidar, = axs[0].plot([], [], label="LiDAR Distance (m)", color='orange')
axs[0].set_ylabel("Altitude (m)")
axs[0].legend()
axs[0].grid()

line_speed, = axs[1].plot([], [], label="Speed (m/s)", color='green')
axs[1].set_ylabel("Speed (m/s)")
axs[1].legend()
axs[1].grid()

line_yaw, = axs[2].plot([], [], label="Yaw (deg)", color='purple')
axs[2].set_ylabel("Yaw (°)")
axs[2].legend()
axs[2].grid()

line_lidar2, = axs[3].plot([], [], label="LiDAR Distance (m)", color='red')
axs[3].set_ylabel("LiDAR Distance (m)")
axs[3].legend()
axs[3].grid()
axs[3].set_xlabel("Frame")

# ----- Main Loop -----
try:
    while True:
        msg = master.recv_match(type=['GPS_RAW_INT', 'HIGHRES_IMU', 'ATTITUDE'], blocking=True)
        now = time.time()
        dt = np.clip(now - last_time, 0.01, 0.2)
        last_time = now

        # Default yaw to None in case ATTITUDE message not received this loop
        yaw_deg = None

        if msg.get_type() == 'HIGHRES_IMU':
            accel = np.array([msg.xacc, msg.yacc, msg.zacc])
            ekf.predict(accel, dt)

        elif msg.get_type() == 'GPS_RAW_INT':
            lat = msg.lat / 1e7
            lon = msg.lon / 1e7
            alt = msg.alt / 1000.0
            x = lon * 111320
            y = lat * 110540
            z = alt
            gps_pos = np.array([x, y, z])
            ekf.update_gps(gps_pos)

        elif msg.get_type() == 'ATTITUDE':
            # Yaw is in radians, convert to degrees and wrap
            yaw_deg = np.degrees(msg.yaw)
            yaw_deg = wrap_angle_deg(yaw_deg)

        # LiDAR Update
        detected, lidar_z = read_lidar(scan_iter)
        if detected:
            ekf.update_lidar(lidar_z)
            lidar_history.append(lidar_z)
        else:
            lidar_history.append(0)

        # Zero velocity update every 1 second (adjust as needed)
        if int(now) % 1 == 0:
            ekf.zero_velocity_update()

        # Calculate speed and wrap yaw for logging and plotting
        vx, vy, vz = ekf.x[3], ekf.x[4], ekf.x[5]
        speed = np.linalg.norm([vx, vy, vz])
        if speed > 1000:  # sanity check
            ekf.x[3:6] = 0
            speed = 0

        z_history.append(ekf.x[2])
        speed_history.append(speed)
        yaw_history.append(yaw_deg if yaw_deg is not None else 0)

        # Plot update
        frames = range(len(z_history))
        line_alt.set_data(frames, z_history)
        line_lidar.set_data(frames, lidar_history)
        axs[0].relim()
        axs[0].autoscale_view()

        line_speed.set_data(frames, speed_history)
        axs[1].relim()
        axs[1].autoscale_view()

        line_yaw.set_data(frames, yaw_history)
        axs[2].set_ylim(-180, 180)
        axs[2].relim()
        axs[2].autoscale_view()

        line_lidar2.set_data(frames, lidar_history)
        axs[3].relim()
        axs[3].autoscale_view()

        plt.pause(0.05)

        # CSV Logging
        csv_writer.writerow({
            "timestamp": now,
            "x": ekf.x[0],
            "y": ekf.x[1],
            "z": ekf.x[2],
            "vx": ekf.x[3],
            "vy": ekf.x[4],
            "vz": ekf.x[5],
            "yaw": yaw_deg if yaw_deg is not None else 0,
            "speed": speed,
            "lidar_z": lidar_z
        })
        csv_file.flush()

        # JSON Logging (append new line with comma except first entry)
        json_entry = {
            "timestamp": now,
            "x": ekf.x[0],
            "y": ekf.x[1],
            "z": ekf.x[2],
            "vx": ekf.x[3],
            "vy": ekf.x[4],
            "vz": ekf.x[5],
            "yaw": yaw_deg if yaw_deg is not None else 0,
            "speed": speed,
            "lidar_z": lidar_z
        }
        if not first_json_entry:
            json_file.write(",\n")
        else:
            first_json_entry = False
        json_file.write(json.dumps(json_entry))
        json_file.flush()

        # Print summary to console
        print(f"[EKF] Z={ekf.x[2]:.2f} m | Speed={speed:.2f} m/s | Yaw={yaw_deg if yaw_deg is not None else 0:.1f}° | LiDAR={lidar_z:.2f} m")

        time.sleep(0.05)

except KeyboardInterrupt:
    print("🛑 Interrupted by user")

finally:
    print("🔻 Cleaning up...")
    csv_file.close()
    json_file.write("\n]\n")  # close JSON array
    json_file.close()
    lidar.stop()
    lidar.disconnect()
