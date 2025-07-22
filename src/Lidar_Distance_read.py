from rplidar import RPLidar, RPLidarException
import matplotlib.pyplot as plt
import numpy as np
import time

# ------------------ Config ------------------
PORT = 'COM3'          # Change if needed
MAX_DISTANCE = 6000    # in mm
SCAN_COUNT_LIMIT = 1000 # Max scans to show
PLOT_SIZE = 6          # inches
SHOW_PLOT = True       # Set False to disable plotting
# --------------------------------------------

def polar_to_cartesian(angle_deg, distance_mm):
    angle_rad = np.radians(angle_deg)
    x = distance_mm * np.cos(angle_rad)
    y = distance_mm * np.sin(angle_rad)
    return x, y

def main():
    lidar = None
    print("=== RPLIDAR A1 Real-Time Scan Viewer ===")
    
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
            # Method 1: Default scan
            scan_iterator = lidar.iter_scans(max_buf_meas=500)
            print("[✓] Using default scan mode")
        except Exception as e:
            print(f"[!] Default scan failed: {e}")
            try:
                # Method 2: Force scan
                lidar.start_scan(force=True)
                scan_iterator = lidar.iter_scans(max_buf_meas=500)
                print("[✓] Using force scan mode")
            except Exception as e2:
                print(f"[!] Force scan also failed: {e2}")
                raise e2
        
        if SHOW_PLOT:
            plt.ion()
            fig, ax = plt.subplots(figsize=(PLOT_SIZE, PLOT_SIZE))
        
        valid_scans = 0
        for i, scan in enumerate(scan_iterator):
            if valid_scans >= SCAN_COUNT_LIMIT:
                break
            
            # Validate scan data
            if not scan or len(scan) == 0:
                print(f"[!] Empty scan {i+1}, retrying...")
                continue
            
            angles = []
            distances = []
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
                    x, y = polar_to_cartesian(angle, distance)
                    angles.append(x)
                    distances.append(y)
                    valid_points += 1
            
            if valid_points > 0:
                valid_scans += 1
                
                if SHOW_PLOT and len(angles) > 0:
                    ax.clear()
                    ax.plot(angles, distances, 'r.', markersize=1.5)
                    ax.set_xlim(-MAX_DISTANCE, MAX_DISTANCE)
                    ax.set_ylim(-MAX_DISTANCE, MAX_DISTANCE)
                    ax.set_title(f"RPLIDAR Scan #{valid_scans} ({valid_points} points)")
                    ax.set_xlabel("X (mm)")
                    ax.set_ylabel("Y (mm)")
                    ax.grid(True, alpha=0.3)
                    ax.set_aspect('equal')
                    
                    # Add distance circles for reference
                    for radius in [1000, 2000, 3000, 4000, 5000]:
                        circle = plt.Circle((0, 0), radius, fill=False, alpha=0.3, linestyle='--')
                        ax.add_patch(circle)
                    
                    plt.pause(0.01)
                
                print(f"[✓] Scan {valid_scans}: {valid_points} valid points (total: {len(scan)})")
                
                # Show some sample measurements
                if valid_scans == 1 and valid_points > 0:
                    print("Sample measurements:")
                    sample_count = min(5, len(scan))
                    for j, measurement in enumerate(scan[:sample_count]):
                        if len(measurement) >= 3:
                            quality, angle, distance = measurement[:3]
                            print(f"  Point {j+1}: {angle:.1f}° = {distance:.1f}mm (Q:{quality})")
            else:
                print(f"[!] Scan {i+1}: No valid points")
    
    except KeyboardInterrupt:
        print("\n[✋] Stopped by user.")
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