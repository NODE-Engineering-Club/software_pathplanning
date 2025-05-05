from gps_reader import get_live_gps

while True:
    lat, lon = get_live_gps("/dev/ttyACM0") # Update if needed
    print("Live GPS →", lat, lon)
