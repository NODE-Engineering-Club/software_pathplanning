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
def get_gps_coordinates(port="COM3", baudrate=9600, timeout=5):
    """
    Read GPS coordinates from COM3 port
    """
    try:
        gps = serial.Serial(port, baudrate=baudrate, timeout=1)
        print(f"✅ GPS connected to {port}")
        
        start_time = time.time()
        while True:
            # Timeout check
            if time.time() - start_time > timeout:
                print("⚠️ GPS timeout - no valid data received")
                return None, None
            
            try:
                line = gps.readline().decode("ascii", errors="replace").strip()
                
                # Check for GPS fix messages
                if line.startswith("$GNGGA") or line.startswith("$GPGGA"):
                    msg = pynmea2.parse(line)
                    if msg.latitude and msg.longitude and msg.gps_qual > 0:  # Check GPS quality
                        lat = msg.latitude
                        lon = msg.longitude
                        print(f"📍 GPS Fix: {lat:.6f}, {lon:.6f} (Quality: {msg.gps_qual})")
                        gps.close()
                        return lat, lon
                        
            except pynmea2.ParseError as e:
                print(f"⚠️ NMEA Parse Error: {e}")
                continue
            except UnicodeDecodeError:
                print("⚠️ Unicode decode error - skipping line")
                continue
                
    except serial.SerialException as e:
        print(f"❌ Serial Error: {e}")
        print("Check if COM3 is available and not used by another application")
        return None, None
    except Exception as e:
        print(f"❌ GPS Error: {e}")
        return None, None

# === IMU Heading Reading ===
def get_heading(port="COM3", baud=57600):
    """
    Get heading from MAVLink connection - adjust port if different from GPS
    """
    try:
        # Use different port if IMU is on different COM port
        master = mavutil.mavlink_connection(f'com:{port}', baud=baud)
        master.wait_heartbeat(timeout=3)
        
        msg = master.recv_match(type='ATTITUDE', blocking=True, timeout=3)
        if msg:
            yaw_rad = msg.yaw
            heading_deg = (math.degrees(yaw_rad) + 360) % 360
            return heading_deg
        else:
            print("⚠️ No attitude message received")
            return None
            
    except Exception as e:
        print(f"❌ Heading error: {e}")
        return None

# === Main Loop ===
def main():
    # === CSV Logging Setup ===
    csv_file = open('kalman_log.csv', mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Timestamp', 'X (m)', 'Y (m)', 'Vx (m/s)', 'Vy (m/s)', 'Heading (deg)', 'Lat', 'Lon'])

    # === Plot Setup ===
    xs, ys = [], []
    fig, ax = plt.subplots(figsize=(10, 8))
    line, = ax.plot([], [], 'b-', linewidth=2, label='Kalman Path')
    current_point, = ax.plot([], [], 'ro', markersize=8, label='Current Position')
    ax.set_title("Real-Time Kalman Filter Position (COM3 GPS)")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.legend()
    ax.grid(True)

    kf = KalmanFilter()
    first_reading = True
    ref_utm = (0, 0)
    
    print("🚀 Starting GPS Kalman Filter on COM3...")
    print("Press Ctrl+C to stop")

    # === Update Loop ===
    def update(frame):
        nonlocal ref_utm, first_reading

        try:
            # Get GPS data
            lat, lon = get_gps_coordinates(port="COM3", baudrate=9600, timeout=2)
            
            if lat is None or lon is None:
                print("⚠️ No GPS data - skipping frame")
                return line, current_point

            # Get heading (adjust port if IMU is on different COM port)
            heading = get_heading(port="COM4", baud=57600)  # Change if needed
            if heading is None:
                heading = 0.0  # Default heading if unavailable

            # Convert to UTM
            utm_x, utm_y, zone_num, zone_letter = utm.from_latlon(lat, lon)

            # Set reference point on first reading
            if first_reading:
                ref_utm = (utm_x, utm_y)
                first_reading = False
                print(f"📍 Reference point set: UTM Zone {zone_num}{zone_letter}")

            # Calculate local coordinates
            x_local = utm_x - ref_utm[0]
            y_local = utm_y - ref_utm[1]

            # Kalman predict + update
            kf.predict()
            kf.update(np.array([[x_local], [y_local]]))
            x, y, vx, vy = kf.get_state()

            # Print state
            print(f"📍 Position: X = {x:.2f} m, Y = {y:.2f} m")
            print(f"🏃 Velocity: Vx = {vx:.2f} m/s, Vy = {vy:.2f} m/s")
            print(f"🧭 Heading: {heading:.2f}°")
            print(f"🌍 GPS: {lat:.6f}, {lon:.6f}")
            print("=" * 50)

            # Save to CSV
            csv_writer.writerow([time.time(), x, y, vx, vy, heading, lat, lon])
            csv_file.flush()  # Ensure data is written

            # Update plot
            xs.append(x)
            ys.append(y)
            
            # Keep only last 100 points for performance
            if len(xs) > 100:
                xs.pop(0)
                ys.pop(0)
            
            line.set_data(xs, ys)
            current_point.set_data([x], [y])
            
            # Auto-scale plot
            if len(xs) > 1:
                ax.relim()
                ax.autoscale_view()

        except Exception as e:
            print(f"❌ Update error: {e}")

        return line, current_point

    # Start animation
    try:
        ani = FuncAnimation(fig, update, interval=1000, blit=False)
        plt.show()
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    finally:
        csv_file.close()
        plt.close()

if __name__ == "__main__":
    main()