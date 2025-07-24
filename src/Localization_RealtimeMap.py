from rplidar import RPLidar, RPLidarException
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import csv
import threading
import queue
from datetime import datetime
from collections import deque
from sklearn.cluster import DBSCAN
import serial
import os

# Optional imports (will work without them if not available)
try:
    from pymavlink import mavutil
    MAVLINK_AVAILABLE = True
except ImportError:
    MAVLINK_AVAILABLE = False
    print("⚠️ pymavlink not available - using simulated GPS/IMU data")

try:
    from rplidar import RPLidar, RPLidarException
    RPLIDAR_AVAILABLE = True
except ImportError:
    RPLIDAR_AVAILABLE = False
    print("⚠️ rplidar library not available - using simulated LiDAR data")

# ============== FILTER CLASSES (STANDALONE) ==============
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
        self.x = np.zeros(6)  # [x, y, z, vx, vy, vz]
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

# ============== SIMULATION CLASSES ==============
class SimulatedGPS:
    """Simulated GPS data for testing when MAVLink is not available"""
    def __init__(self):
        self.time_start = time.time()
        self.base_lat = 40.7128  # New York coordinates
        self.base_lon = -74.0060
        self.base_alt = 10.0
        
    def get_gps_data(self):
        """Generate simulated GPS data with circular movement"""
        t = time.time() - self.time_start
        radius = 0.001  # Small radius in degrees
        
        lat = self.base_lat + radius * np.sin(t * 0.1)
        lon = self.base_lon + radius * np.cos(t * 0.1)
        alt = self.base_alt + 2 * np.sin(t * 0.05)
        
        return lat, lon, alt
    
    def get_attitude_data(self):
        """Generate simulated attitude data"""
        t = time.time() - self.time_start
        
        roll = 0.1 * np.sin(t * 0.3)
        pitch = 0.05 * np.sin(t * 0.2)
        yaw = t * 0.1  # Slowly rotating
        
        return roll, pitch, yaw

class SimulatedLiDAR:
    """Simulated LiDAR data for testing"""
    def __init__(self):
        self.time_start = time.time()
        
    def get_clusters(self):
        """Generate simulated LiDAR clusters"""
        t = time.time() - self.time_start
        clusters = []
        
        # Simulate 2-4 objects at different distances
        num_objects = 3
        for i in range(num_objects):
            angle = (i * 120 + t * 10) % 360  # Objects moving around
            distance = 50 + 30 * np.sin(t * 0.1 + i)  # Varying distance
            
            cluster = {
                'distance': distance,
                'angle': angle,
                'intensity': 0.7 + 0.3 * np.sin(t + i),
                'point_count': 5 + int(3 * np.sin(t + i)),
                'x': distance * np.cos(np.radians(angle)) / 1000,  # Convert to meters
                'y': distance * np.sin(np.radians(angle)) / 1000
            }
            clusters.append(cluster)
        
        return clusters

# ============== MAIN INTEGRATED SYSTEM ==============
class IntegratedShipNavigation:
    def __init__(self, mavlink_port='COM4', lidar_port='COM3', map_size_m=2000, simulation_mode=False):
        """
        Integrated ship navigation system - STANDALONE VERSION
        
        Args:
            mavlink_port: MAVLink serial port (if available)
            lidar_port: LiDAR serial port (if available)  
            map_size_m: Map size in meters
            simulation_mode: Use simulated data instead of real sensors
        """
        print("=== Integrated Ship Navigation System ===")
        
        self.simulation_mode = simulation_mode or not (MAVLINK_AVAILABLE and RPLIDAR_AVAILABLE)
        self.map_size_m = map_size_m
        
        # Initialize filters
        self.gps_filter = GPS_EKF()
        self.roll_filter = KalmanFilter1D()
        self.pitch_filter = KalmanFilter1D()
        self.yaw_filter = KalmanFilter1D()
        self.gps_initialized = False
        
        # Ship state
        self.ship_position = np.array([0.0, 0.0, 0.0])  # X, Y, Z in local coordinates
        self.ship_velocity = np.array([0.0, 0.0, 0.0])
        self.ship_attitude = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
        self.gps_reference = None
        
        # Object tracking
        self.detected_objects = []
        self.object_history = deque(maxlen=1000)
        self.position_history = deque(maxlen=500)
        
        # Initialize data sources
        self.setup_data_sources(mavlink_port, lidar_port)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize real-time plot
        self.setup_realtime_plot()
        
        # Threading control
        self.running = True
        
    def setup_data_sources(self, mavlink_port, lidar_port):
        """Initialize real or simulated data sources"""
        if self.simulation_mode:
            print("🎮 Using simulation mode")
            self.gps_sim = SimulatedGPS()
            self.lidar_sim = SimulatedLiDAR()
            self.mavlink_master = None
            self.lidar_connection = None
        else:
            print("📡 Attempting to connect to real sensors...")
            try:
                # MAVLink connection
                if MAVLINK_AVAILABLE:
                    self.mavlink_master = mavutil.mavlink_connection(mavlink_port, baud=57600)
                    self.mavlink_master.wait_heartbeat()
                    print(f"✓ Connected to MAVLink on {mavlink_port}")
                else:
                    self.mavlink_master = None
                
                # LiDAR connection - try different approaches
                self.lidar_connection = None
                if RPLIDAR_AVAILABLE:
                    try:
                        self.lidar_connection = RPLidar(lidar_port)
                        print(f"✓ Connected to RPLiDAR on {lidar_port}")
                    except:
                        try:
                            self.lidar_connection = serial.Serial(lidar_port, 115200, timeout=1)
                            print(f"✓ Connected to LiDAR via serial on {lidar_port}")
                        except:
                            print(f"⚠️ Could not connect to LiDAR on {lidar_port}")
                            self.lidar_connection = None
                
                # Fall back to simulation if connections failed
                if not self.mavlink_master and not self.lidar_connection:
                    print("⚠️ No sensors connected, falling back to simulation mode")
                    self.simulation_mode = True
                    self.gps_sim = SimulatedGPS()
                    self.lidar_sim = SimulatedLiDAR()
                    
            except Exception as e:
                print(f"⚠️ Sensor connection failed: {e}")
                print("⚠️ Falling back to simulation mode")
                self.simulation_mode = True
                self.gps_sim = SimulatedGPS()
                self.lidar_sim = SimulatedLiDAR()
                self.mavlink_master = None
                self.lidar_connection = None
    
    def setup_logging(self):
        """Setup CSV logging"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"navigation_log_{timestamp}.csv"
        
        self.csv_file = open(filename, mode="w", newline="")
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=[
            "timestamp", "ship_x", "ship_y", "ship_z", "ship_speed",
            "roll", "pitch", "yaw", "objects_detected", "nearest_object_distance",
            "simulation_mode"
        ])
        self.csv_writer.writeheader()
        print(f"📝 Logging to: {filename}")
    
    def setup_realtime_plot(self):
        """Initialize real-time plotting"""
        plt.ion()
        self.fig, (self.ax_map, self.ax_sensors) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Map plot
        half_size = self.map_size_m / 2
        self.ax_map.set_xlim(-half_size, half_size)
        self.ax_map.set_ylim(-half_size, half_size)
        self.ax_map.set_aspect('equal')
        self.ax_map.grid(True, alpha=0.3)
        self.ax_map.set_xlabel('East (m)')
        self.ax_map.set_ylabel('North (m)')
        self.ax_map.set_title('Ship Navigation Map')
        
        # Initialize map elements
        self.ship_marker, = self.ax_map.plot([], [], 'ro', markersize=12, label='Ship')
        self.ship_heading_line, = self.ax_map.plot([], [], 'r-', linewidth=3, alpha=0.8)
        self.path_line, = self.ax_map.plot([], [], 'b-', alpha=0.6, linewidth=2, label='Path')
        self.objects_scatter = self.ax_map.scatter([], [], c=[], s=60, cmap='plasma', 
                                                 alpha=0.8, label='Objects')
        self.ax_map.legend()
        
        # Add range circles
        for radius in [200, 500, 1000]:
            circle = plt.Circle((0, 0), radius, fill=False, alpha=0.2, linestyle='--', color='gray')
            self.ax_map.add_patch(circle)
        
        # Sensor data plot
        self.sensor_history = {
            'altitude': deque(maxlen=100), 
            'speed': deque(maxlen=100), 
            'object_distance': deque(maxlen=100)
        }
        self.time_history = deque(maxlen=100)
        
        self.line_alt, = self.ax_sensors.plot([], [], 'b-', label='Altitude (m)', linewidth=2)
        self.line_speed, = self.ax_sensors.plot([], [], 'g-', label='Speed (m/s)', linewidth=2)
        self.line_obj_dist, = self.ax_sensors.plot([], [], 'r-', label='Nearest Object (m)', linewidth=2)
        
        self.ax_sensors.set_xlabel('Time (s)')
        self.ax_sensors.set_ylabel('Value')
        self.ax_sensors.set_title('Sensor Data')
        self.ax_sensors.legend()
        self.ax_sensors.grid(True, alpha=0.3)
        
        # Info text
        self.info_text = self.ax_map.text(0.02, 0.98, '', transform=self.ax_map.transAxes,
                                         verticalalignment='top', fontfamily='monospace',
                                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        plt.show()
    
    def gps_to_local(self, lat, lon, alt):
        """Convert GPS coordinates to local ENU coordinates"""
        if self.gps_reference is None:
            self.gps_reference = np.array([lat, lon, alt])
            print(f"📍 GPS reference set: {lat:.6f}, {lon:.6f}, {alt:.1f}m")
            return np.array([0.0, 0.0, 0.0])
        
        # ENU conversion
        lat_diff = lat - self.gps_reference[0]
        lon_diff = lon - self.gps_reference[1]
        alt_diff = alt - self.gps_reference[2]
        
        # Approximate conversion for small areas
        x = lon_diff * 111320 * np.cos(np.radians(self.gps_reference[0]))  # East
        y = lat_diff * 110540  # North
        z = alt_diff  # Up
        
        return np.array([x, y, z])
    
    def get_gps_imu_data(self):
        """Get GPS and IMU data from real sensors or simulation"""
        if self.simulation_mode:
            lat, lon, alt = self.gps_sim.get_gps_data()
            roll, pitch, yaw = self.gps_sim.get_attitude_data()
            return lat, lon, alt, roll, pitch, yaw
        else:
            # Real MAVLink data
            if self.mavlink_master:
                try:
                    msg = self.mavlink_master.recv_match(type=['GPS_RAW_INT', 'ATTITUDE'], timeout=0.1)
                    if msg:
                        if msg.get_type() == 'GPS_RAW_INT':
                            lat = msg.lat / 1e7
                            lon = msg.lon / 1e7
                            alt = msg.alt / 1000.0
                            return lat, lon, alt, None, None, None
                        elif msg.get_type() == 'ATTITUDE':
                            return None, None, None, msg.roll, msg.pitch, msg.yaw
                except:
                    pass
            
            # Return None if no data available
            return None, None, None, None, None, None
    
    def get_lidar_data(self):
        """Get LiDAR data from real sensor or simulation"""
        if self.simulation_mode:
            return self.lidar_sim.get_clusters()
        else:
            # Try to read from real LiDAR
            clusters = []
            
            # Method 1: Try to load from your existing LiDAR JSON file
            if os.path.exists('lidar_data.json'):
                try:
                    with open('lidar_data.json', 'r') as f:
                        lidar_data = json.load(f)
                    
                    if lidar_data and isinstance(lidar_data, list) and len(lidar_data) > 0:
                        latest_scan = lidar_data[-1]
                        if 'Point_Data' in latest_scan:
                            for point in latest_scan['Point_Data']:
                                x_m, y_m = point['x'], point['y']
                                distance = np.sqrt(x_m*x_m + y_m*y_m)
                                angle = np.degrees(np.arctan2(y_m, x_m))
                                
                                cluster = {
                                    'distance': distance,
                                    'angle': angle,
                                    'intensity': point.get('intensity', 0.5),
                                    'point_count': 1,
                                    'x': x_m,
                                    'y': y_m
                                }
                                clusters.append(cluster)
                            
                            return clusters
                except:
                    pass
            
            # Method 2: Try direct serial reading
            if self.lidar_connection and hasattr(self.lidar_connection, 'readline'):
                try:
                    line = self.lidar_connection.readline().decode().strip()
                    if line:
                        distance = float(line)
                        # Create a single cluster directly ahead
                        cluster = {
                            'distance': distance,
                            'angle': 0.0,
                            'intensity': 0.5,
                            'point_count': 1,
                            'x': distance,
                            'y': 0.0
                        }
                        return [cluster]
                except:
                    pass
            
            return []
    
    def polar_to_cartesian_global(self, angle_deg, distance_m):
        """Convert LiDAR polar coordinates to global map coordinates"""
        # Account for ship heading
        global_angle = np.radians(angle_deg) + self.ship_attitude['yaw']
        
        # Calculate object position relative to ship
        obj_x = self.ship_position[0] + distance_m * np.cos(global_angle)
        obj_y = self.ship_position[1] + distance_m * np.sin(global_angle)
        
        return np.array([obj_x, obj_y])
    
    def update_objects_from_lidar(self, clusters):
        """Update detected objects from LiDAR clusters"""
        current_objects = []
        current_time = time.time()
        
        for cluster in clusters:
            # Convert to global coordinates
            if 'x' in cluster and 'y' in cluster:
                # Direct cartesian coordinates - rotate by ship heading
                cos_yaw = np.cos(self.ship_attitude['yaw'])
                sin_yaw = np.sin(self.ship_attitude['yaw'])
                
                # Rotate LiDAR coordinates by ship heading
                rotated_x = cluster['x'] * cos_yaw - cluster['y'] * sin_yaw
                rotated_y = cluster['x'] * sin_yaw + cluster['y'] * cos_yaw
                
                global_pos = np.array([
                    self.ship_position[0] + rotated_x,
                    self.ship_position[1] + rotated_y
                ])
            else:
                # Polar coordinates
                global_pos = self.polar_to_cartesian_global(cluster['angle'], cluster['distance'])
            
            obj_data = {
                'position': global_pos,
                'distance': cluster['distance'],
                'angle': cluster.get('angle', 0),
                'intensity': cluster.get('intensity', 0.5),
                'timestamp': current_time
            }
            
            current_objects.append(obj_data)
            self.object_history.append(obj_data)
        
        # Remove old objects (older than 5 seconds)
        self.object_history = deque([obj for obj in self.object_history 
                                   if current_time - obj['timestamp'] < 5.0], maxlen=1000)
        
        self.detected_objects = current_objects
    
    def get_persistent_objects(self):
        """Get spatially clustered persistent objects"""
        if not self.object_history:
            return []
        
        positions = np.array([obj['position'] for obj in self.object_history])
        
        # Spatial clustering
        clustering = DBSCAN(eps=5.0, min_samples=2).fit(positions)
        labels = clustering.labels_
        
        # Group by cluster
        clusters = {}
        for i, label in enumerate(labels):
            if label != -1:
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(list(self.object_history)[i])
        
        # Calculate cluster centers
        persistent_objects = []
        for cluster_objects in clusters.values():
            if len(cluster_objects) >= 2:
                avg_pos = np.mean([obj['position'] for obj in cluster_objects], axis=0)
                avg_intensity = np.mean([obj['intensity'] for obj in cluster_objects])
                
                persistent_objects.append({
                    'position': avg_pos,
                    'intensity': avg_intensity,
                    'confidence': min(len(cluster_objects) / 5.0, 1.0)
                })
        
        return persistent_objects
    
    def update_display(self):
        """Update the real-time display"""
        current_time = time.time()
        
        # Update ship position and heading
        self.ship_marker.set_data([self.ship_position[0]], [self.ship_position[1]])
        
        # Update ship heading indicator
        heading_length = 50  # meters
        heading_end_x = self.ship_position[0] + heading_length * np.cos(self.ship_attitude['yaw'])
        heading_end_y = self.ship_position[1] + heading_length * np.sin(self.ship_attitude['yaw'])
        self.ship_heading_line.set_data([self.ship_position[0], heading_end_x], 
                                       [self.ship_position[1], heading_end_y])
        
        # Update ship path
        if len(self.position_history) > 1:
            path_x = [pos[0] for pos in self.position_history]
            path_y = [pos[1] for pos in self.position_history]
            self.path_line.set_data(path_x, path_y)
        
        # Update detected objects
        persistent_objects = self.get_persistent_objects()
        if persistent_objects:
            obj_x = [obj['position'][0] for obj in persistent_objects]
            obj_y = [obj['position'][1] for obj in persistent_objects]
            obj_colors = [obj['confidence'] for obj in persistent_objects]
            obj_sizes = [60 + 40 * obj['confidence'] for obj in persistent_objects]
            
            # Clear previous scatter and create new one
            self.objects_scatter.remove()
            self.objects_scatter = self.ax_map.scatter(obj_x, obj_y, c=obj_colors, s=obj_sizes,
                                                     cmap='plasma', alpha=0.8, 
                                                     label='Objects', edgecolors='black')
        
        # Update sensor history plots
        self.time_history.append(current_time)
        self.sensor_history['altitude'].append(self.ship_position[2])
        self.sensor_history['speed'].append(np.linalg.norm(self.ship_velocity))
        
        # Find nearest object distance
        nearest_dist = 1000  # Default far distance
        if self.detected_objects:
            distances = [obj['distance'] for obj in self.detected_objects]
            nearest_dist = min(distances)
        self.sensor_history['object_distance'].append(nearest_dist)
        
        # Update sensor plots
        if len(self.time_history) > 1:
            time_array = np.array(self.time_history) - self.time_history[0]  # Relative time
            
            self.line_alt.set_data(time_array, list(self.sensor_history['altitude']))
            self.line_speed.set_data(time_array, list(self.sensor_history['speed']))
            self.line_obj_dist.set_data(time_array, list(self.sensor_history['object_distance']))
            
            # Auto-scale sensor plot
            self.ax_sensors.relim()
            self.ax_sensors.autoscale_view()
        
        # Update info text
        mode_str = "🎮 SIMULATION" if self.simulation_mode else "📡 LIVE SENSORS"
        info_str = f"{mode_str}\n"
        info_str += f"Position: ({self.ship_position[0]:.1f}, {self.ship_position[1]:.1f}, {self.ship_position[2]:.1f}) m\n"
        info_str += f"Speed: {np.linalg.norm(self.ship_velocity):.1f} m/s\n"
        info_str += f"Heading: {np.degrees(self.ship_attitude['yaw']):.1f}°\n"
        info_str += f"Objects: {len(self.detected_objects)} current, {len(persistent_objects)} persistent\n"
        info_str += f"Nearest: {nearest_dist:.1f} m\n"
        info_str += f"Time: {datetime.now().strftime('%H:%M:%S')}"
        
        self.info_text.set_text(info_str)
        
        # Refresh display
        plt.draw()
        plt.pause(0.01)
    
    def run(self):
        """Main execution loop"""
        print(f"🚀 Starting navigation system ({'simulation' if self.simulation_mode else 'live sensors'})")
        print("Press Ctrl+C to stop")
        
        last_gps_time = 0
        last_save_time = time.time()
        
        try:
            while self.running:
                current_time = time.time()
                
                # Get GPS/IMU data
                gps_data = self.get_gps_imu_data()
                lat, lon, alt, roll, pitch, yaw = gps_data
                
                # Process GPS data
                if lat is not None and lon is not None and alt is not None:
                    local_pos = self.gps_to_local(lat, lon, alt)
                    
                    # Apply EKF filtering
                    if not self.gps_initialized:
                        self.gps_filter.x[:3] = local_pos
                        self.gps_initialized = True
                        self.ship_position, self.ship_velocity = local_pos, np.array([0, 0, 0])
                    else:
                        self.ship_position, self.ship_velocity = self.gps_filter.filter(local_pos)
                    
                    # Add to position history
                    self.position_history.append(self.ship_position[:2])  # Only X, Y for path
                    last_gps_time = current_time
                
                # Process attitude data
                if roll is not None:
                    self.ship_attitude['roll'] = self.roll_filter.filter(roll)
                if pitch is not None:
                    self.ship_attitude['pitch'] = self.pitch_filter.filter(pitch)
                if yaw is not None:
                    self.ship_attitude['yaw'] = self.yaw_filter.filter(yaw)
                
                # Get LiDAR data
                lidar_clusters = self.get_lidar_data()
                if lidar_clusters:
                    self.update_objects_from_lidar(lidar_clusters)
                
                # Update display
                self.update_display()
                
                # Log data every second
                if current_time - last_save_time >= 1.0:
                    nearest_distance = 1000
                    if self.detected_objects:
                        nearest_distance = min([obj['distance'] for obj in self.detected_objects])
                    
                    log_data = {
                        "timestamp": current_time,
                        "ship_x": float(self.ship_position[0]),
                        "ship_y": float(self.ship_position[1]),
                        "ship_z": float(self.ship_position[2]),
                        "ship_speed": float(np.linalg.norm(self.ship_velocity)),
                        "roll": float(self.ship_attitude['roll']),
                        "pitch": float(self.ship_attitude['pitch']),
                        "yaw": float(self.ship_attitude['yaw']),
                        "objects_detected": len(self.detected_objects),
                        "nearest_object_distance": float(nearest_distance),
                        "simulation_mode": self.simulation_mode
                    }
                    
                    self.csv_writer.writerow(log_data)
                    self.csv_file.flush()
                    last_save_time = current_time
                
                # Control loop rate
                time.sleep(0.05)  # 20 Hz
                
        except KeyboardInterrupt:
            print("\n🛑 Navigation system stopped by user")
        except Exception as e:
            print(f"\n❌ Error in navigation system: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        
        if hasattr(self, 'csv_file'):
            self.csv_file.close()
            print("📝 Log file saved")
        
        if hasattr(self, 'lidar_connection') and self.lidar_connection:
            try:
                if hasattr(self.lidar_connection, 'stop'):
                    self.lidar_connection.stop()
                if hasattr(self.lidar_connection, 'disconnect'):
                    self.lidar_connection.disconnect()
                elif hasattr(self.lidar_connection, 'close'):
                    self.lidar_connection.close()
            except:
                pass
        
        plt.ioff()
        print("✅ Cleanup complete")

def main():
    """Main function to run the integrated navigation system"""
    print("🚢 Integrated Ship Navigation System")
    print("=" * 50)
    
    # Configuration - modify these for your setup
    config = {
        'mavlink_port': 'COM4',      # Change to your MAVLink port
        'lidar_port': 'COM3',        # Change to your LiDAR port
        'map_size_m': 2000,          # Map size in meters
        'simulation_mode': False     # Set True to force simulation mode
    }
    
    # Display configuration
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("")
    
    # Create and run navigation system
    try:
        nav_system = IntegratedShipNavigation(**config)
        nav_system.run()
    except Exception as e:
        print(f"❌ Failed to start navigation system: {e}")
        print("\n💡 Try running in simulation mode by setting simulation_mode=True")

if __name__ == "__main__":
    main()