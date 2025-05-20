import serial
import time
import math
import csv
import matplotlib.pyplot as plt

PORT = 'COM3'  # Update this to your LiDAR COM port
BAUDRATE = 115200
MAX_POINTS = 360
csv_file_path = "rplidar_scan_data.csv"

# RPLIDAR commands
SYNC_BYTE = b'\xA5'
SCAN_CMD = b'\x20'
STOP_CMD = b'\x25'
MOTOR_ON = b'\xA5\xF0\x02\xFF\xFF'
MOTOR_OFF = b'\xA5\xF0\x02\x00\x00'

def parse_measurement_packet(data):
    if len(data) != 5:
        return None
    quality = data[0] >> 2
    angle = ((data[2] >> 1) | (data[3] << 7)) / 64.0
    distance = ((data[4] << 8) | data[1]) / 4.0
    return angle, distance, quality

ser = serial.Serial(PORT, BAUDRATE, timeout=1)
ser.write(MOTOR_ON)
time.sleep(0.1)
ser.write(SYNC_BYTE + SCAN_CMD)
time.sleep(0.1)

plt.ion()
fig, ax = plt.subplots(figsize=(6, 6))
scan_plot, = ax.plot([], [], 'bo', markersize=2)
ax.set_xlim(-10000, 10000)
ax.set_ylim(-10000, 10000)
ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
ax.set_title("Live RPLIDAR Scan and Logging")
ax.grid(True)

angles = []
distances = []

with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["timestamp", "angle_deg", "distance_mm", "quality"])

    try:
        print("Logging and visualizing... Press Ctrl+C to stop.")
        while True:
            sync = ser.read(1)
            if sync != b'\xA5':
                continue
            response = ser.read(1)
            if response != b'\x5A':
                continue

            ser.read(5)
            data_packet = ser.read(5)

            if len(data_packet) == 5:
                parsed = parse_measurement_packet(data_packet)
                if parsed:
                    angle, distance, quality = parsed
                    if quality > 0 and 0 < distance < 10000:
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S.%f")
                        writer.writerow([timestamp, angle, distance, quality])
                        angles.append(math.radians(angle))
                        distances.append(distance)

            if len(angles) >= MAX_POINTS:
                xs = [d * math.cos(a) for a, d in zip(angles, distances)]
                ys = [d * math.sin(a) for a, d in zip(angles, distances)]
                scan_plot.set_data(xs, ys)
                fig.canvas.draw()
                fig.canvas.flush_events()
                angles.clear()
                distances.clear()

    except KeyboardInterrupt:
        print("Stopped by user.")

    finally:
        ser.write(MOTOR_OFF)
        ser.write(SYNC_BYTE + STOP_CMD)
        ser.close()
        print("Serial port closed and data saved.")
