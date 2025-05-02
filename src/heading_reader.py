from pymavlink import mavutil
import math

def get_heading(port="/dev/ttyACM0", baud=57600):
    try:
        master = mavutil.mavlink_connection(port, baud=baud)
        master.wait_heartbeat()
        msg = master.recv_match(type='ATTITUDE', blocking=True)
        yaw = (math.degrees(msg.yaw) + 360) % 360
        print("🧭 Heading:", yaw)
        return yaw
    except Exception as e:
        print("❌ Heading Error:", e)
        return None
