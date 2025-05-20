import serial
import time
import csv
from datetime import datetime

# === CONFIGURATION ===
PORT = 'COM3'  # Replace with the correct COM port (e.g., COM4, COM5)
BAUDRATE = 115200  # Default RPLIDAR A2M8 speed
csv_file_path = "rplidar_scan_data.csv"  # Output path for the logged data

# === COMMAND BYTES ===
SYNC_BYTE = b'\xA5'
SCAN_CMD = b'\x20'         # Start scan command
STOP_CMD = b'\x25'         # Stop scan command
MOTOR_ON = b'\xA5\xF0\x02\xFF\xFF'  # Max PWM (motor spins at full speed)
MOTOR_OFF = b'\xA5\xF0\x02\x00\x00'  # Turn off motor

# === PARSE 5-BYTE MEASUREMENT PACKET ===
def parse_measurement_packet(data):
    if len(data) != 5:
        return None
    quality = data[0] >> 2
    angle = ((data[2] >> 1) | (data[3] << 7)) / 64.0
    distance = ((data[4] << 8) | data[1]) / 4.0
    return angle, distance, quality

# === OPEN SERIAL CONNECTION ===
ser = serial.Serial(PORT, BAUDRATE, timeout=1)

# === START MOTOR AND SCAN MODE ===
ser.write(MOTOR_ON)                  # Start the motor
time.sleep(0.1)
ser.write(SYNC_BYTE + SCAN_CMD)     # Begin scanning
time.sleep(0.1)

# === START LOGGING TO CSV ===
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["timestamp", "angle_deg", "distance_mm", "quality"])  # CSV headers

    try:
        print("Logging started. Press Ctrl+C to stop.")
        while True:
            sync = ser.read(1)
            if sync != b'\xA5':
                continue
            response = ser.read(1)
            if response != b'\x5A':
                continue

            ser.read(5)  # Skip 5-byte descriptor
            data_packet = ser.read(5)

            if len(data_packet) == 5:
                parsed = parse_measurement_packet(data_packet)
                if parsed:
                    angle, distance, quality = parsed
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                    writer.writerow([timestamp, angle, distance, quality])

    except KeyboardInterrupt:
        print("Logging stopped by user.")

    finally:
        # === STOP MOTOR AND CLOSE CONNECTION ===
        ser.write(MOTOR_OFF)              # Stop the motor
        ser.write(SYNC_BYTE + STOP_CMD)   # End scan mode
        ser.close()
        print("Serial port closed.")
