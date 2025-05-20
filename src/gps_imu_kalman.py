import serial
import pynmea2
import utm
import time
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pymavlink import mavutil
import os
os.environ["PYTHON"] = "python"
# === Kalman Filter Setup ===
class KalmanFilter:
    def __init__(self):
        dt = 1  # Time step (s)

        # State: [x, y, vx, vy]
        self.x = np.zeros((4, 1))

        # State transition matrix
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ])

        # Measurement matrix
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Measurement noise covariance
        self.R = np.eye(2) * 5

        # Process noise covariance
        self.Q = np.eye(4) * 0.01

        # Estimate covariance
        self.P = np.eye(4)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def get_state(self):
        return self.x.flatten()


# === GPS Reading ===
def get_gps_coordinates(port="/dev/ttyACM0", baudrate=9600):
    gps = serial.Serial(port, baudrate=baudrate, timeout=1)
    while True:
        line = gps.readline().decode("ascii", errors="replace").strip()
        if line.startswith("$GNGGA") or line.startswith("$GPGGA"):
            try:
                msg = pynmea2.parse(line)
                if msg.latitude and msg.longitude:
                    lat = msg.latitude
                    lon = msg.longitude
                    return lat, lon
            except pynmea2.ParseError:
                continue


# === IMU Heading Reading ===
def get_heading(port="/dev/ttyACM0", baud=57600):
    try:
        master = mavutil.mavlink_connection(port, baud=baud)
        master.wait_heartbeat()
        msg = master.recv_match(type='ATTITUDE', blocking=True)
        yaw_rad = msg.yaw
        heading_deg = (math.degrees(yaw_rad) + 360) % 360
        return heading_deg
    except Exception as e:
        print("❌ Heading error:", e)
        return None


# === CSV Logging Setup ===
csv_file = open('kalman_log.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Timestamp', 'X (m)', 'Y (m)', 'Vx (m/s)', 'Vy (m/s)', 'Heading (deg)'])

# === Plot Setup ===
xs, ys = [], []
fig, ax = plt.subplots()
line, = ax.plot([], [], 'b-', label='Kalman Path')
ax.set_title("Real-Time Kalman Filter Position")
ax.set_xlabel("X (meters)")
ax.set_ylabel("Y (meters)")
ax.legend()

kf = KalmanFilter()
first_reading = True
ref_utm = (0, 0)

# === Update Loop ===
def update(frame):
    global ref_utm, first_reading

    lat, lon = get_gps_coordinates()
    heading = get_heading()

    utm_x, utm_y, _, _ = utm.from_latlon(lat, lon)

    if first_reading:
        ref_utm = (utm_x, utm_y)
        first_reading = False

    x_local = utm_x - ref_utm[0]
    y_local = utm_y - ref_utm[1]

    # Kalman predict + update
    kf.predict()
    kf.update(np.array([[x_local], [y_local]]))
    x, y, vx, vy = kf.get_state()

    # Print state
    print(f"Position: X = {x:.2f} m, Y = {y:.2f} m")
    print(f"Velocity: Vx = {vx:.2f} m/s, Vy = {vy:.2f} m/s")
    print(f"Heading: {heading:.2f}°")
    print("------------")

    # Save to CSV
    csv_writer.writerow([time.time(), x, y, vx, vy, heading])

    # Update plot
    xs.append(x)
    ys.append(y)
    line.set_data(xs, ys)
    ax.relim()
    ax.autoscale_view()

    return line,

ani = FuncAnimation(fig, update, interval=1000)
plt.show()
csv_file.close()
