import serial
import pynmea2

def get_live_gps(port_name="COM3", baudrate=9600): #  def get_heading(port="/dev/ttyACM0", baud=57600):
    try:
        port = serial.Serial(port_name, baudrate=baudrate, timeout=1)
        while True:
            line = port.readline().decode("ascii", errors="replace")
            if line.startswith("$GNGGA") or line.startswith("$GPGGA"):
                msg = pynmea2.parse(line)
                return msg.latitude, msg.longitude
    except Exception as e:
        print("GPS Error:", e)
        return None, None
