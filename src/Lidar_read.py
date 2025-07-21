import serial
import time

PORT = 'COM3'        # Confirm this is the CP2102 port
BAUD = 115200        # Try 9600, 57600, or 115200 if unsure
TIMEOUT = 1          # So it doesn’t freeze indefinitely

try:
    ser = serial.Serial(PORT, BAUD, timeout=TIMEOUT)
    print(f"Connected to {PORT} at {BAUD} baud")
except Exception as e:
    print(f"Could not open serial port: {e}")
    exit(1)

print("Reading data... Press Ctrl+C to stop.")

try:
    while True:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if line:
            print(f"Received: {line}")
        else:
            print("No data...")
        time.sleep(0.5)
except KeyboardInterrupt:
    print("Stopped.")
finally:
    ser.close()
