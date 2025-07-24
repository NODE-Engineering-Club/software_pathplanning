import numpy as np
import matplotlib.pyplot as plt
from pymavlink import mavutil
from rplidar import RPLidar, RPLidarException
import time
import csv
from collections import deque

# ---------------- Kalman Filter ----------------
class KalmanFilter1D:
    def __init__(self, q=1e-4, r=1e-2):
        self.x = 0.0
        self.P = 1.0
        self.Q = q
        self.R = r

    def predict(self):
        self.P += self.Q

    def update(self, measurement):
        K = self.P / (self.P + self.R)
        self.x += K * (measurement - self.x)
        self.P *= (1 - K)

    def filter(self, measurement):
        self.predict()
        self.update(measurement)
        return self.x

# ---------------- GPS EKF ----------------
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

# ---------------- LiDAR Reader ----------------
def read_lidar(scan_iter):
    try:
        scan = next(scan_iter)
        distances = [m[2] for m in scan if m[2] > 0]
        if distances:
            min_distance = min(distances) / 1000.0  # mm to m
            return True, min_distance
    except (StopIteration, RPLidarException):
        pass
    return False, 0.0

# ---------------- Plot Setup ----------------
window_size = 100
z_history = deque(maxlen=window_size)
lidar_history = deque(maxlen=window_size)
speed_history = deque(maxlen=window_size)

plt.ion()
fig, ax = plt.subplots()
line1, = ax.plot([], [], label="Fused Altitude (Z)", color='blue')
line2, = ax.plot([], [], label="LiDAR Distance", color='orange')
line3, = ax.plot([], [], label="Speed (m/s)", color='green')
ax.set_ylim(0, 30)
ax.set_xlabel("Time (frames)")
ax.set_ylabel("Distance / Speed (m)")
ax.legend()

# ---------------- Connections ----------------
master = mavutil.mavlink_connection('COM4', baud=57600)
master.wait_heartbeat()
print("✅ Connected to MAVLink on COM4")

lidar = RPLidar('COM3')
lidar_scan_iter = lidar.iter_scans()
print("✅ Connected to RPLidar on COM3")

# ---------------- Filters ----------------
gps_filter = GPS_EKF()
roll_f = KalmanFilter1D()
pitch_f = KalmanFilter1D()
yaw_f = KalmanFilter1D()
gps_initialized = False

# ---------------- CSV Logger ----------------
csv_file = open("output_log.csv", mode="w", newline="")
csv_writer = csv.DictWriter(csv_file, fieldnames=[
    "timestamp", "fused_x", "fused_y", "fused_z",
    "speed_mps", "roll", "pitch", "yaw",
    "object_detected", "object_distance_m"
])
csv_writer.writeheader()

# ---------------- Main Loop ----------------
last_plot_save_time = time.time()
last_lidar_read_time = time.time()
lidar_read_interval = 0.2  # seconds

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

        # ---------------- GPS ----------------
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

        # ---------------- Attitude ----------------
        elif msg.get_type() == 'ATTITUDE':
            output["roll"] = roll_f.filter(msg.roll)
            output["pitch"] = pitch_f.filter(msg.pitch)
            output["yaw"] = yaw_f.filter(msg.yaw)

        # ---------------- LiDAR ----------------
        if now - last_lidar_read_time >= lidar_read_interval:
            obj_detected, obj_distance = read_lidar(lidar_scan_iter)
            output["object_detected"] = obj_detected
            output["object_distance_m"] = obj_distance
            lidar_history.append(obj_distance)
            last_lidar_read_time = now

        # ---------------- Plot Update ----------------
        line1.set_ydata(z_history)
        line1.set_xdata(range(len(z_history)))
        line2.set_ydata(lidar_history)
        line2.set_xdata(range(len(lidar_history)))
        line3.set_ydata(speed_history)
        line3.set_xdata(range(len(speed_history)))
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.01)

        # ---------------- Save Plot ----------------
        if now - last_plot_save_time >= 10:
            filename = f"plot_{int(now)}.png"
            fig.savefig(filename)
            print(f"📸 Saved plot: {filename}")
            last_plot_save_time = now

        # ---------------- Log Output ----------------
        print("📡 FUSED OUTPUT:", output)
        csv_writer.writerow(output)
        csv_file.flush()

except KeyboardInterrupt:
    print("🛑 Stopped by user.")

finally:
    print("⚠️ Cleaning up...")
    csv_file.close()
    lidar.stop()
    lidar.disconnect()
