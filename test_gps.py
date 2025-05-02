from gps_reader import get_live_gps

# Adjust the port if using UART GPIO
lat, lon = get_live_gps("/dev/ttyUSB0")
print("Returned GPS →", lat, lon)
