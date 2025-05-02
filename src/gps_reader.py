import serial
import pynmea2

def get_live_gps(port="/dev/ttyACM0", baudrate=115200):
    try:
        gps_serial = serial.Serial(port, baudrate=baudrate, timeout=1)
        print(f"✅ Connected to GPS on {port} at {baudrate} baud")

        while True:
            line = gps_serial.readline().decode("ascii", errors="replace").strip()
            if line.startswith("$GNGGA") or line.startswith("$GPGGA"):
                try:
                    msg = pynmea2.parse(line)
                    lat = msg.latitude
                    lon = msg.longitude
                    if lat and lon:
                        print(f"📍 Latitude: {lat}, Longitude: {lon}")
                        return lat, lon
                except pynmea2.ParseError:
                    continue
    except Exception as e:
        print("❌ GPS Error:", e)
        return None, None

