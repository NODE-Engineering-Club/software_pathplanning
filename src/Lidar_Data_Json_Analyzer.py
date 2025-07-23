import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pandas as pd

# Font size configuration
TITLE_FONT_SIZE = 15
LABEL_FONT_SIZE = 11
TICK_FONT_SIZE = 8
LEGEND_FONT_SIZE = 6

# Set global font sizes
plt.rcParams.update({
    'font.size': TICK_FONT_SIZE,
    'axes.titlesize': TITLE_FONT_SIZE,
    'axes.labelsize': LABEL_FONT_SIZE,
    'xtick.labelsize': TICK_FONT_SIZE,
    'ytick.labelsize': TICK_FONT_SIZE,
    'legend.fontsize': LEGEND_FONT_SIZE
})

def set_font_sizes(title=16, label=14, tick=12, legend=11):
    """
    Set custom font sizes for all plots
    
    Args:
        title: Font size for plot titles
        label: Font size for axis labels
        tick: Font size for tick labels
        legend: Font size for legend text
    """
    plt.rcParams.update({
        'font.size': tick,
        'axes.titlesize': title,
        'axes.labelsize': label,
        'xtick.labelsize': tick,
        'ytick.labelsize': tick,
        'legend.fontsize': legend
    })
    print(f"[✓] Font sizes updated: title={title}, label={label}, tick={tick}, legend={legend}")

def load_lidar_data(filename='lidar_data.json'):
    """Load LiDAR data from JSON file"""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        print(f"[✓] Loaded {len(data)} scan records from {filename}")
        return data
    except FileNotFoundError:
        print(f"[✗] File {filename} not found!")
        return []
    except Exception as e:
        print(f"[✗] Error loading data: {e}")
        return []

def analyze_data(data):
    """Analyze the LiDAR data and print statistics"""
    if not data:
        return
    
    print("\n=== DATA ANALYSIS ===")
    
    # Overall statistics
    total_scans = len(data)
    total_clusters = sum(scan['clusters_found'] for scan in data)
    total_points = sum(scan['total_points'] for scan in data)
    
    print(f"Total scans: {total_scans}")
    print(f"Total clusters detected: {total_clusters}")
    print(f"Total points processed: {total_points}")
    print(f"Average clusters per scan: {total_clusters/total_scans:.2f}")
    print(f"Average points per scan: {total_points/total_scans:.2f}")
    
    # Time analysis
    if total_scans > 1:
        start_time = datetime.fromisoformat(data[0]['timestamp'])
        end_time = datetime.fromisoformat(data[-1]['timestamp'])
        duration = (end_time - start_time).total_seconds()
        print(f"Duration: {duration:.2f} seconds")
        print(f"Scan rate: {total_scans/duration:.2f} Hz")
    
    # Cluster analysis
    all_distances = []
    cluster_counts = []
    
    for scan in data:
        cluster_counts.append(scan['clusters_found'])
        for cluster_id, cluster_data in scan['clusters'].items():
            all_distances.append(cluster_data['distance_m'])
    
    if all_distances:
        print(f"\nCluster distance statistics:")
        print(f"  Min distance: {min(all_distances):.3f}m")
        print(f"  Max distance: {max(all_distances):.3f}m")
        print(f"  Mean distance: {np.mean(all_distances):.3f}m")
        print(f"  Std distance: {np.std(all_distances):.3f}m")

def plot_cluster_positions(data, scan_range=None):
    """Plot cluster positions over time"""
    if not data:
        return
    
    # Select scan range
    if scan_range:
        start_idx, end_idx = scan_range
        data = data[start_idx:end_idx]
    
    plt.figure(figsize=(12, 8))
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Cluster positions (top view)
    all_x, all_y = [], []
    colors = []
    scan_numbers = []
    
    for scan_idx, scan in enumerate(data):
        for cluster_id, cluster_data in scan['clusters'].items():
            all_x.append(cluster_data['x_m'])
            all_y.append(cluster_data['y_m'])
            colors.append(scan_idx)
            scan_numbers.append(scan['scan_number'])
    
    if all_x:
        scatter = ax1.scatter(all_x, all_y, c=colors, cmap='viridis', alpha=0.6, s=30)
        ax1.set_xlabel('X (meters)')
        ax1.set_ylabel('Y (meters)')
        ax1.set_title('Cluster Positions Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        plt.colorbar(scatter, ax=ax1, label='Scan Index')
        
        # Add circle references with custom font size
        for radius in [1, 2, 3, 4, 5]:  # meters
            circle = plt.Circle((0, 0), radius, fill=False, alpha=0.3, linestyle='--', color='gray')
            ax1.add_patch(circle)
            ax1.text(radius*0.7, radius*0.7, f'{radius}m', fontsize=TICK_FONT_SIZE-2, 
                    alpha=0.7, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    # Plot 2: Distance distribution
    all_distances = []
    for scan in data:
        for cluster_id, cluster_data in scan['clusters'].items():
            all_distances.append(cluster_data['distance_m'])
    
    if all_distances:
        ax2.hist(all_distances, bins=30, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Distance (meters)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Cluster Distance Distribution')
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Clusters per scan over time
    scan_nums = [scan['scan_number'] for scan in data]
    cluster_counts = [scan['clusters_found'] for scan in data]
    
    ax3.plot(scan_nums, cluster_counts, 'b-', alpha=0.7, linewidth=1)
    ax3.set_xlabel('Scan Number')
    ax3.set_ylabel('Number of Clusters')
    ax3.set_title('Clusters Detected Over Time')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Point count distribution
    point_counts = [scan['total_points'] for scan in data]
    
    ax4.hist(point_counts, bins=20, alpha=0.7, edgecolor='black', color='orange')
    ax4.set_xlabel('Points per Scan')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Point Count Distribution')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def export_to_csv(data, filename='lidar_clusters.csv'):
    """Export cluster data to CSV format"""
    rows = []
    
    for scan in data:
        base_row = {
            'timestamp': scan['timestamp'],
            'scan_number': scan['scan_number'],
            'total_points': scan['total_points'],
            'clusters_found': scan['clusters_found']
        }
        
        if scan['clusters']:
            for cluster_id, cluster_data in scan['clusters'].items():
                row = base_row.copy()
                row['cluster_id'] = cluster_id
                row['x_m'] = cluster_data['x_m']
                row['y_m'] = cluster_data['y_m']
                row['z_m'] = cluster_data['z_m']
                row['distance_m'] = cluster_data['distance_m']
                row['angle_deg'] = cluster_data['angle_deg']
                row['point_count'] = cluster_data['point_count']
                rows.append(row)
        else:
            # Include scans with no clusters
            row = base_row.copy()
            for key in ['cluster_id', 'x_m', 'y_m', 'z_m', 'distance_m', 'angle_deg', 'point_count']:
                row[key] = None
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"[📊] Exported {len(rows)} records to {filename}")

def print_recent_scans(data, count=5):
    """Print the most recent scans in a readable format"""
    if not data:
        return
    
    print(f"\n=== LAST {count} SCANS ===")
    recent_data = data[-count:] if len(data) >= count else data
    
    for scan in recent_data:
        timestamp = datetime.fromisoformat(scan['timestamp']).strftime("%H:%M:%S")
        print(f"\n[{timestamp}] Scan #{scan['scan_number']}: {scan['total_points']} points → {scan['clusters_found']} clusters")
        
        if scan['clusters']:
            for cluster_id, cluster_data in scan['clusters'].items():
                print(f"  {cluster_id}: ({cluster_data['x_m']:6.3f}m, {cluster_data['y_m']:6.3f}m) "
                      f"dist={cluster_data['distance_m']:5.3f}m, "
                      f"angle={cluster_data['angle_deg']:6.1f}°, "
                      f"pts={cluster_data['point_count']}")

def main():
    """Main analysis function"""
    print("=== LiDAR Data Analyzer ===")
    
    # Uncomment and modify to change font sizes:
    # set_font_sizes(title=20, label=16, tick=14, legend=12)  # Larger fonts
    # set_font_sizes(title=12, label=10, tick=8, legend=9)    # Smaller fonts
    
    # Load data
    data = load_lidar_data()
    if not data:
        return
    
    # Analyze data
    analyze_data(data)
    
    # Print recent scans
    print_recent_scans(data)
    
    # Export to CSV
    export_to_csv(data)
    
    # Create visualizations
    print("\n[📈] Creating visualizations...")
    plot_cluster_positions(data)
    
    print("\n[✓] Analysis complete!")

if __name__ == '__main__':
    main()