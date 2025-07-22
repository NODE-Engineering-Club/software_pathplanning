import serial
import time
import sys

# -------------------------------
# Configuration
# -------------------------------
PORT = 'COM3'         # Change if needed
BAUD = 115200         # Try 9600, 57600, etc.
TIMEOUT = 1           # 1 second timeout for read

# -------------------------------
# Connect to Serial Port
# -------------------------------
try:
    ser = serial.Serial(PORT, BAUD, timeout=TIMEOUT)
    print(f"[✓] Connected to {PORT} at {BAUD} baud.")
except Exception as e:
    print(f"[✗] Could not open serial port {PORT}: {e}")
    sys.exit(1)

print("\n--- Reading LiDAR Data ---\n(Press Ctrl+C to stop)\n") 

# -------------------------------
# Read Loop
# -------------------------------
try:
    while True:
        bytes_waiting = ser.in_waiting
        if bytes_waiting > 0:
            raw_data = ser.read(bytes_waiting)

            # Try decoding to text
            try:
                decoded = raw_data.decode('utf-8', errors='ignore').strip()
                if decoded:
                    print(f"[Text] {decoded}")
                else:
                    print(f"[Info] Data received, but no readable text.")
            except Exception as decode_error:
                print(f"[Decode Error] {decode_error}")

            # Always show raw hex output for debugging
            print(f"[Raw] {raw_data.hex()}")
        else:
            print("[Info] No data available...")

        time.sleep(0.5)

except KeyboardInterrupt:
    print("\n[✋] Interrupted by user. Exiting...")

finally:
    ser.close()
    print("[✓] Serial port closed.")
