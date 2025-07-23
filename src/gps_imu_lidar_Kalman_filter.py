import numpy as np
import matplotlib.pyplot as plt
from pymavlink import mavutil
import time
import csv
import serial
from collections import deque
from random import random

# ---- Kalman & EKF Filters ----
class KalmanFilter1D:
    def __init__(self, q=1e-4, r=1e-2):
        self.x = 0.0
        self.P = 1.0
        self.Q = q
        self.R = r
        self.F = 1.0
        self.H = 1.0

    def predict(self):
        self.x = self.F * self.x
        self.P = self.F * self.P * self.F + self.Q

    def update(self, measurement):
        K = self.P * self.H / (self.H * self.P * self.H + self.R)
        self.x = self.x + K * (measurement - self.H * self.x)
        self.P = (1 - K * self.H) * self.P

    def filter(self, measurement):
        self.predict()
        self.update(measurement)
        return self.x

class GPS_EKF:
    def __init__(self, dt=0.1):
        self.x = np.zeros(6)
        self.P = np.eye(6) * 100
        self.Q = np.eye(6) * 0.01
        self.R = np.eye(3) * 0.5
        self.dt = dt

    def predict(self):
        F = np.array([
            [1, 0, 0, self.dt, 0, 0],
            [0, 1, 0, 0, self.dt, 0],
            [0, 0, 1, 0, 0, self.dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update(self, measurement):
        H = np.eye(3, 6)
        y = measurement - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P

    def filter(self, measurement):
        self.predict()
        self.update(measurement)
        return self.x[:3], self.x[3:]

# ---- LiDAR Reader ----
def read_lidar(serial_lidar):
    try:
        line = serial_lidar.readline().decode().strip()
        if line:
            distance = float(line)  # assume the LiDAR sends distance in meters
            return True, distance
    except:
        pass
    return False, 0.0

# ---- Plot Setup ----
window_size = 100
z_history = deque(maxlen=window_size)
lidar_history = deque(maxlen=window_size)
speed_history = deque(maxlen=window_size)

plt.ion()
fig, ax = plt.subplots()
line1, = ax.plot([], [], label="Fused Z (Altitude)", color='blue')
line2, = ax.plot([], [], label="LiDAR Distance", color='orange')
line3, = ax.plot([], [], label="Speed (m/s)", color='green')
ax.set_ylim(0, 30)
ax.set_xlabel("Time (frames)")
ax.set_ylabel("Distance / Speed (m)")
ax.legend()

# ---- MAVLink Init ----
master = mavutil.mavlink_connection('COM3', baud=57600)
master.wait_heartbeat()
print("Connected to MAVLink")

# ---- LiDAR Serial Init ----
serial_lidar = serial.Serial('COM4', 115200, timeout=1)

# ---- Filters ----
gps_filter = GPS_EKF()
roll_f = KalmanFilter1D()
pitch_f = KalmanFilter1D()
yaw_f = KalmanFilter1D()
gps_initialized = False

# ---- CSV Setup ----
csv_file = open("output_log.csv", mode="w", newline="")
csv_writer = csv.DictWriter(csv_file, fieldnames=[
    "timestamp", "fused_x", "fused_y", "fused_z",
    "speed_mps", "roll", "pitch", "yaw",
    "object_detected", "object_distance_m"
])
csv_writer.writeheader()

last_plot_save_time = time.time()

# ---- Main Loop ----
try:
    while True:
        msg = master.recv_match(type=['GPS_RAW_INT', 'ATTITUDE'], blocking=True)
        now = time.time()

        output = {
            "timestamp": now,
            "fused_x": None,
            "fused_y": None,
            "fused_z": None,
            "speed_mps": None,
            "roll": None,
            "pitch": None,
            "yaw": None,
            "object_detected": False,
            "object_distance_m": 0.0
        }

        if msg.get_type() == 'GPS_RAW_INT':
            lat = msg.lat / 1e7
            lon = msg.lon / 1e7
            alt = msg.alt / 1000.0
            x = (lon - 0) * 111320
            y = (lat - 0) * 110540
            z = alt

            if not gps_initialized:
                gps_filter.x[:3] = np.array([x, y, z])
                gps_initialized = True
                fused_xyz, vel = [x, y, z], [0, 0, 0]
            else:
                fused_xyz, vel = gps_filter.filter(np.array([x, y, z]))

            output["fused_x"], output["fused_y"], output["fused_z"] = fused_xyz
            output["speed_mps"] = np.linalg.norm(vel)
            z_history.append(fused_xyz[2])
            speed_history.append(output["speed_mps"])

        elif msg.get_type() == 'ATTITUDE':
            output["roll"] = roll_f.filter(msg.roll)
            output["pitch"] = pitch_f.filter(msg.pitch)
            output["yaw"] = yaw_f.filter(msg.yaw)

        # ---- LiDAR Input ----
        obj_detected, obj_distance = read_lidar(serial_lidar)
        output["object_detected"] = obj_detected
        output["object_distance_m"] = obj_distance
        lidar_history.append(obj_distance)

        # ---- Plot update ----
        line1.set_ydata(z_history)
        line1.set_xdata(range(len(z_history)))
        line2.set_ydata(lidar_history)
        line2.set_xdata(range(len(lidar_history)))
        line3.set_ydata(speed_history)
        line3.set_xdata(range(len(speed_history)))

        ax.relim()
        ax.autoscale_view()
        plt.pause(0.01)

        # ---- Save Plot Periodically ----
        if time.time() - last_plot_save_time >= 10:
            filename = f"plot_{int(time.time())}.png"
            fig.savefig(filename)
            print(f"Saved plot: {filename}")
            last_plot_save_time = time.time()

        print("FUSED OUTPUT:", output)
        csv_writer.writerow(output)
        csv_file.flush()

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    csv_file.close()
    serial_lidar.close()
