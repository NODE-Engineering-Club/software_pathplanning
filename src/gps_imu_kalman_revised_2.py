from pymavlink import mavutil
import time
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

class KalmanFilter:
    def __init__(self, process_variance=1e-3, measurement_variance=1e-1):
        # State estimate
        self.x = 0.0
        # Error covariance
        self.P = 1.0
        # Process noise covariance
        self.Q = process_variance
        # Measurement noise covariance
        self.R = measurement_variance
        # State transition model (assuming constant value)
        self.F = 1.0
        # Observation model
        self.H = 1.0
        
    def predict(self):
        # Predict the state
        self.x = self.F * self.x
        # Predict the error covariance
        self.P = self.F * self.P * self.F + self.Q
        
    def update(self, measurement):
        # Calculate Kalman gain
        K = self.P * self.H / (self.H * self.P * self.H + self.R)
        # Update state estimate
        self.x = self.x + K * (measurement - self.H * self.x)
        # Update error covariance
        self.P = (1 - K * self.H) * self.P
        
    def filter(self, measurement):
        self.predict()
        self.update(measurement)
        return self.x

class ExtendedKalmanFilter:
    """Extended Kalman Filter for GPS position estimation"""
    def __init__(self):
        # State vector: [lat, lon, alt, lat_vel, lon_vel, alt_vel]
        self.x = np.zeros(6)
        # Covariance matrix
        self.P = np.eye(6) * 100
        # Process noise
        self.Q = np.eye(6) * 0.01
        # Measurement noise
        self.R = np.eye(3) * 0.1
        # Time step
        self.dt = 0.1
        
    def predict(self):
        # State transition matrix (constant velocity model)
        F = np.array([
            [1, 0, 0, self.dt, 0, 0],
            [0, 1, 0, 0, self.dt, 0],
            [0, 0, 1, 0, 0, self.dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # Predict state
        self.x = F @ self.x
        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q
        
    def update(self, measurement):
        # Measurement matrix (we observe position only)
        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        
        # Innovation
        y = measurement - H @ self.x
        # Innovation covariance
        S = H @ self.P @ H.T + self.R
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ y
        # Update covariance
        I = np.eye(6)
        self.P = (I - K @ H) @ self.P
        
    def filter(self, measurement):
        self.predict()
        self.update(measurement)
        return self.x[:3]  # Return position only

# Change this to your connection string:
udp = 'udp:127.0.0.1:14551'
connection_string = 'COM3' #'/dev/ttyACM0'  # or 'udp:127.0.0.1:14551'
baud = 57600

# Connect to the flight controller
master = mavutil.mavlink_connection(connection_string, baud=baud)

# Wait for a heartbeat to confirm connection
master.wait_heartbeat()
print(f"Connected to system {master.target_system}, component {master.target_component}")

# Initialize data lists
lat_list = []
lon_list = []
alt_list = []
lat_filtered_list = []
lon_filtered_list = []
alt_filtered_list = []

imu_roll_list = []
imu_pitch_list = []
imu_yaw_list = []
imu_roll_filtered_list = []
imu_pitch_filtered_list = []
imu_yaw_filtered_list = []

# Initialize Kalman filters
gps_filter = ExtendedKalmanFilter()
roll_filter = KalmanFilter(process_variance=1e-4, measurement_variance=1e-2)
pitch_filter = KalmanFilter(process_variance=1e-4, measurement_variance=1e-2)
yaw_filter = KalmanFilter(process_variance=1e-4, measurement_variance=1e-2)

# Initialize GPS filter with first measurement
gps_initialized = False

plt.figure(figsize=(15, 10))
plt.ion()

# Loop to read GPS and IMU data
while True:
    msg = master.recv_match(type=['GPS_RAW_INT', 'ATTITUDE'], blocking=True)
    if not msg:
        continue

    if msg.get_type() == 'GPS_RAW_INT':
        lat = msg.lat / 1e7
        lon = msg.lon / 1e7
        alt = msg.alt / 1000

        lat_list.append(lat)
        lon_list.append(lon)
        alt_list.append(alt)
        
        # Apply Kalman filter to GPS data
        if not gps_initialized:
            gps_filter.x[:3] = np.array([lat, lon, alt])
            gps_initialized = True
            lat_filtered, lon_filtered, alt_filtered = lat, lon, alt
        else:
            measurement = np.array([lat, lon, alt])
            filtered_pos = gps_filter.filter(measurement)
            lat_filtered, lon_filtered, alt_filtered = filtered_pos
        
        lat_filtered_list.append(lat_filtered)
        lon_filtered_list.append(lon_filtered)
        alt_filtered_list.append(alt_filtered)
        
        print(f"GPS Raw | Lat: {lat:.6f}, Lon: {lon:.6f}, Alt: {alt:.2f} m")
        print(f"GPS Filtered | Lat: {lat_filtered:.6f}, Lon: {lon_filtered:.6f}, Alt: {alt_filtered:.2f} m")

    if msg.get_type() == 'ATTITUDE':
        roll = msg.roll
        pitch = msg.pitch
        yaw = msg.yaw

        imu_roll_list.append(roll)
        imu_pitch_list.append(pitch)
        imu_yaw_list.append(yaw)
        
        # Apply Kalman filter to IMU data
        roll_filtered = roll_filter.filter(roll)
        pitch_filtered = pitch_filter.filter(pitch)
        yaw_filtered = yaw_filter.filter(yaw)
        
        imu_roll_filtered_list.append(roll_filtered)
        imu_pitch_filtered_list.append(pitch_filtered)
        imu_yaw_filtered_list.append(yaw_filtered)
        
        print(f"IMU Raw | Roll: {roll:.3f}, Pitch: {pitch:.3f}, Yaw: {yaw:.3f}")
        print(f"IMU Filtered | Roll: {roll_filtered:.3f}, Pitch: {pitch_filtered:.3f}, Yaw: {yaw_filtered:.3f}")

    # Plotting the live data
    plt.clf()
    
    # GPS Position Plot
    if lat_list and lon_list:
        plt.subplot(2, 2, 1)
        plt.plot(lon_list, lat_list, marker='o', linestyle='-', color='b', alpha=0.5, label='Raw GPS')
        if lat_filtered_list and lon_filtered_list:
            plt.plot(lon_filtered_list, lat_filtered_list, marker='s', linestyle='-', color='r', label='Filtered GPS')
        plt.title('GPS Position')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        plt.grid()
    
    # GPS Altitude Plot
    if alt_list:
        plt.subplot(2, 2, 2)
        plt.plot(alt_list, color='b', alpha=0.5, label='Raw Altitude')
        if alt_filtered_list:
            plt.plot(alt_filtered_list, color='r', label='Filtered Altitude')
        plt.title('GPS Altitude')
        plt.xlabel('Sample Number')
        plt.ylabel('Altitude (m)')
        plt.legend()
        plt.grid()
    
    # IMU Attitude Plot
    if imu_roll_list and imu_pitch_list and imu_yaw_list:
        plt.subplot(2, 2, 3)
        plt.plot(imu_roll_list, label='Raw Roll', color='r', alpha=0.5)
        plt.plot(imu_pitch_list, label='Raw Pitch', color='g', alpha=0.5)
        plt.plot(imu_yaw_list, label='Raw Yaw', color='b', alpha=0.5)
        plt.title('IMU Attitude (Raw)')
        plt.xlabel('Sample Number')
        plt.ylabel('Angle (radians)')
        plt.legend()
        plt.grid()
        
    # Filtered IMU Attitude Plot
    if imu_roll_filtered_list and imu_pitch_filtered_list and imu_yaw_filtered_list:
        plt.subplot(2, 2, 4)
        plt.plot(imu_roll_filtered_list, label='Filtered Roll', color='r')
        plt.plot(imu_pitch_filtered_list, label='Filtered Pitch', color='g')
        plt.plot(imu_yaw_filtered_list, label='Filtered Yaw', color='b')
        plt.title('IMU Attitude (Filtered)')
        plt.xlabel('Sample Number')
        plt.ylabel('Angle (radians)')
        plt.legend()
        plt.grid()

    plt.tight_layout()
    plt.pause(0.1)