from rplidar import RPLidar, RPLidarException
import matplotlib.pyplot as plt
import numpy as np
import time
import json
import csv
from datetime import datetime
from sklearn.cluster import DBSCAN
import os

# ------------------ Config ------------------
PORT = 'COM3'          # Change if needed
MAX_DISTANCE = 6000    # in mm
PLOT_SIZE = 6          # inches
SHOW_PLOT = True       # Set False to disable plotting
OUTPUT_FILE = 'lidar_data.json'  # Output JSON file
OUTPUT_CSV = 'lidar_data.csv'    # Output CSV file
SAVE_TO_JSON = True              # Enable JSON saving
SAVE_TO_CSV = True               # Enable CSV saving
CLUSTERING_EPS = 200   # DBSCAN epsilon parameter (mm)
MIN_SAMPLES = 3        # Minimum points per cluster
CONTINUOUS_RUN = True  # Run continuously without scan limit
SAVE_INTERVAL = 1      # Save every N scans
# --------------------------------------------

def polar_to_cartesian(angle_deg, distance_mm):
    """Convert polar coordinates to cartesian"""
    angle_rad = np.radians(angle_deg)
    x = distance_mm * np.cos(angle_rad)
    y = distance_mm * np.sin(angle_rad)
    z = 0  # LiDAR is 2D, so z is always 0
    return x, y, z

def cluster_points(points, eps=200, min_samples=3):
    """
    Cluster LiDAR points using DBSCAN algorithm
    
    Args:
        points: List of (x, y, z, distance, angle) tuples
        eps: Maximum distance between points in same cluster (mm)
        min_samples: Minimum points required to form a cluster
    
    Returns:
        List of cluster dictionaries with mean values
    """
    if len(points) < min_samples:
        return []
    
    # Extract x, y coordinates for clustering
    coordinates = np.array([(p[0], p[1]) for p in points])
    
    # Perform DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coordinates)
    labels = clustering.labels_
    
    # Group points by cluster
    clusters = {}
    for i, label in enumerate(labels):
        if label != -1:  # Ignore noise points (label -1)
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(points[i])
    
    # Calculate mean values for each cluster
    cluster_data = []
    for cluster_id, cluster_points in clusters.items():
        if len(cluster_points) >= min_samples:
            # Calculate means
            x_values = [p[0] for p in cluster_points]
            y_values = [p[1] for p in cluster_points]
            z_values = [p[2] for p in cluster_points]
            distances = [p[3] for p in cluster_points]
            angles = [p[4] for p in cluster_points]
            
            mean_x = np.mean(x_values)
            mean_y = np.mean(y_values)
            mean_z = np.mean(z_values)
            mean_distance = np.mean(distances)
            mean_angle = np.mean(angles)
            
            cluster_data.append({
                'cluster_id': f'P{len(cluster_data) + 1}',
                'x_m': mean_x / 1000.0,  # Convert to meters
                'y_m': mean_y / 1000.0,  # Convert to meters
                'z_m': mean_z / 1000.0,  # Convert to meters
                'distance_m': mean_distance / 1000.0,  # Convert to meters
                'angle_deg': mean_angle,
                'point_count': len(cluster_points)
            })
    
    return cluster_data

def save_scan_data(scan_data, json_filename, csv_filename):
    """Save scan data to both JSON and CSV files"""
    success = True
    
    # Save to JSON
    if SAVE_TO_JSON:
        try:
            # Load existing data if file exists
            if os.path.exists(json_filename):
                with open(json_filename, 'r') as f:
                    existing_data = json.load(f)
            else:
                existing_data = []
            
            # Append new data
            existing_data.append(scan_data)
            
            # Save back to file
            with open(json_filename, 'w') as f:
                json.dump(existing_data, f, indent=2)
        except Exception as e:
            print(f"[!] Error saving JSON data: {e}")
            success = False
    
    # Save to CSV
    if SAVE_TO_CSV:
        try:
            # Check if CSV file exists to determine if we need headers
            file_exists = os.path.exists(csv_filename)
            
            with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'timestamp', 'scan_number', 'total_points', 'clusters_found',
                    'cluster_id', 'x_m', 'y_m', 'z_m', 'distance_m', 'angle_deg', 'point_count'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # Write header if file is new
                if not file_exists:
                    writer.writeheader()
                
                # Write data rows
                base_row = {
                    'timestamp': scan_data['timestamp'],
                    'scan_number': scan_data['scan_number'],
                    'total_points': scan_data['total_points'],
                    'clusters_found': scan_data['clusters_found']
                }
                
                if scan_data['clusters']:
                    # Write one row per cluster
                    for cluster_id, cluster_data in scan_data['clusters'].items():
                        row = base_row.copy()
                        row.update({
                            'cluster_id': cluster_id,
                            'x_m': cluster_data['x_m'],
                            'y_m': cluster_data['y_m'],
                            'z_m': cluster_data['z_m'],
                            'distance_m': cluster_data['distance_m'],
                            'angle_deg': cluster_data['angle_deg'],
                            'point_count': cluster_data['point_count']
                        })
                        writer.writerow(row)
                else:
                    # Write one row even if no clusters found
                    row = base_row.copy()
                    for field in ['cluster_id', 'x_m', 'y_m', 'z_m', 'distance_m', 'angle_deg', 'point_count']:
                        row[field] = ''
                    writer.writerow(row)
                    
        except Exception as e:
            print(f"[!] Error saving CSV data: {e}")
            success = False
    
    return success

def main():
    lidar = None
    print("=== RPLIDAR A1 Real-Time Clustering Viewer ===")
    print(f"[📝] JSON output: {OUTPUT_FILE} {'(enabled)' if SAVE_TO_JSON else '(disabled)'}")
    print(f"[📊] CSV output: {OUTPUT_CSV} {'(enabled)' if SAVE_TO_CSV else '(disabled)'}")
    print(f"[🔧] Clustering parameters: eps={CLUSTERING_EPS}mm, min_samples={MIN_SAMPLES}")
    
    try:
        # Connect with timeout
        lidar = RPLidar(PORT, timeout=3)
        print(f"[✓] Connected to {PORT}")
        
        # Get device info first
        try:
            info = lidar.get_info()
            print(f"[✓] Device Info: {info}")
        except Exception as e:
            print(f"[!] Could not get device info: {e}")
        
        # Get health status
        try:
            health = lidar.get_health()
            print(f"[✓] Health: {health}")
        except Exception as e:
            print(f"[!] Could not get health: {e}")
        
        time.sleep(1)
        
        print("[...] Starting motor...")
        lidar.start_motor()
        time.sleep(3)  # Give more time for motor to spin up
        
        print("[...] Clearing buffer and starting scan...")
        # Clear any existing data
        try:
            lidar.clear_input()
        except Exception as e:
            print(f"[!] Clear input warning: {e}")
        
        time.sleep(1)
        
        # Try different scan methods
        scan_iterator = None
        try:
            # Method 1: Default scan with larger buffer
            scan_iterator = lidar.iter_scans(max_buf_meas=1000)
            print("[✓] Using default scan mode")
        except Exception as e:
            print(f"[!] Default scan failed: {e}")
            try:
                # Method 2: Force scan
                lidar.start_scan(force=True)
                scan_iterator = lidar.iter_scans(max_buf_meas=1000)
                print("[✓] Using force scan mode")
            except Exception as e2:
                print(f"[!] Force scan also failed: {e2}")
                raise e2
        
        if SHOW_PLOT:
            plt.ion()
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(PLOT_SIZE*2, PLOT_SIZE))
        
        valid_scans = 0
        print(f"[🔄] Starting continuous scan mode...")
        
        for i, scan in enumerate(scan_iterator):
            # Validate scan data
            if not scan or len(scan) == 0:
                continue
            
            points = []
            raw_angles = []
            raw_distances = []
            valid_points = 0
            
            for measurement in scan:
                # Handle different measurement formats
                if len(measurement) == 3:
                    quality, angle, distance = measurement
                elif len(measurement) == 4:
                    quality, angle, distance, _ = measurement
                else:
                    continue
                
                # Filter valid measurements
                if 0 < distance < MAX_DISTANCE and quality > 0:
                    x, y, z = polar_to_cartesian(angle, distance)
                    points.append((x, y, z, distance, angle))
                    raw_angles.append(x)
                    raw_distances.append(y)
                    valid_points += 1
            
            if valid_points > 0:
                valid_scans += 1
                
                # Perform clustering
                clusters = cluster_points(points, eps=CLUSTERING_EPS, min_samples=MIN_SAMPLES)
                
                # Create scan data structure
                scan_data = {
                    'timestamp': datetime.now().isoformat(),
                    'scan_number': valid_scans,
                    'total_points': valid_points,
                    'clusters_found': len(clusters),
                    'clusters': {cluster['cluster_id']: {
                        'x_m': round(cluster['x_m'], 3),
                        'y_m': round(cluster['y_m'], 3),
                        'z_m': round(cluster['z_m'], 3),
                        'distance_m': round(cluster['distance_m'], 3),
                        'angle_deg': round(cluster['angle_deg'], 1),
                        'point_count': cluster['point_count']
                    } for cluster in clusters}
                }
                
                # Save data every SAVE_INTERVAL scans
                if valid_scans % SAVE_INTERVAL == 0:
                    if save_scan_data(scan_data, OUTPUT_FILE, OUTPUT_CSV):
                        save_msg = []
                        if SAVE_TO_JSON: save_msg.append(f"JSON to {OUTPUT_FILE}")
                        if SAVE_TO_CSV: save_msg.append(f"CSV to {OUTPUT_CSV}")
                        print(f"[💾] Saved scan {valid_scans} data: {', '.join(save_msg)}")
                
                # Display results
                print(f"[✓] Scan {valid_scans}: {valid_points} points → {len(clusters)} clusters")
                if clusters:
                    for cluster in clusters:
                        print(f"    {cluster['cluster_id']}: ({cluster['x_m']:.2f}m, {cluster['y_m']:.2f}m) "
                              f"dist={cluster['distance_m']:.2f}m, angle={cluster['angle_deg']:.1f}°, "
                              f"points={cluster['point_count']}")
                
                # Plotting
                if SHOW_PLOT and len(raw_angles) > 0:
                    # Plot 1: Raw points
                    ax1.clear()
                    ax1.plot(raw_angles, raw_distances, 'r.', markersize=1.5, alpha=0.6)
                    ax1.set_xlim(-MAX_DISTANCE, MAX_DISTANCE)
                    ax1.set_ylim(-MAX_DISTANCE, MAX_DISTANCE)
                    ax1.set_title(f"Raw Points - Scan #{valid_scans}")
                    ax1.set_xlabel("X (mm)")
                    ax1.set_ylabel("Y (mm)")
                    ax1.grid(True, alpha=0.3)
                    ax1.set_aspect('equal')
                    
                    # Plot 2: Clustered points
                    ax2.clear()
                    if clusters:
                        colors = plt.cm.tab10(np.linspace(0, 1, len(clusters)))
                        for idx, cluster in enumerate(clusters):
                            x_mm = cluster['x_m'] * 1000
                            y_mm = cluster['y_m'] * 1000
                            ax2.plot(x_mm, y_mm, 'o', color=colors[idx], 
                                   markersize=8, label=cluster['cluster_id'])
                            ax2.annotate(cluster['cluster_id'], 
                                       (x_mm, y_mm), 
                                       xytext=(5, 5), 
                                       textcoords='offset points',
                                       fontsize=8)
                    
                    ax2.set_xlim(-MAX_DISTANCE, MAX_DISTANCE)
                    ax2.set_ylim(-MAX_DISTANCE, MAX_DISTANCE)
                    ax2.set_title(f"Clusters - Scan #{valid_scans} ({len(clusters)} clusters)")
                    ax2.set_xlabel("X (mm)")
                    ax2.set_ylabel("Y (mm)")
                    ax2.grid(True, alpha=0.3)
                    ax2.set_aspect('equal')
                    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    
                    # Add distance circles for reference
                    for ax in [ax1, ax2]:
                        for radius in [1000, 2000, 3000, 4000, 5000]:
                            circle = plt.Circle((0, 0), radius, fill=False, alpha=0.3, linestyle='--')
                            ax.add_patch(circle)
                    
                    plt.tight_layout()
                    plt.pause(0.01)
                
                # Add small delay to prevent overwhelming output
                time.sleep(0.1)
            
            # Handle buffer clearing
            if "Too many measurements" in str(scan):
                print("[!] Buffer overflow detected, clearing...")
                try:
                    lidar.clear_input()
                except:
                    pass
    
    except KeyboardInterrupt:
        print(f"\n[✋] Stopped by user after {valid_scans} scans.")
        save_msg = []
        if SAVE_TO_JSON: save_msg.append(f"JSON: {OUTPUT_FILE}")
        if SAVE_TO_CSV: save_msg.append(f"CSV: {OUTPUT_CSV}")
        if save_msg:
            print(f"[📊] Data saved to: {', '.join(save_msg)}")
    except RPLidarException as e:
        print(f"[✗] RPLidar Error: {e}")
        print("[💡] Troubleshooting tips:")
        print("  - Try unplugging and reconnecting the LiDAR")
        print("  - Check if another program is using the port")
        print("  - Verify the LiDAR model matches the library version")
        print("  - Try a different USB cable")
    except Exception as e:
        print(f"[✗] General Error: {e}")
    finally:
        if lidar:
            print("[...] Stopping and cleaning up...")
            try:
                lidar.stop()
                lidar.stop_motor()
                lidar.disconnect()
            except Exception as e:
                print(f"[!] Cleanup warning: {e}")
        if SHOW_PLOT:
            try:
                plt.ioff()
                plt.show()
            except:
                pass
        print("[✓] Finished.")

if __name__ == '__main__':
    main()