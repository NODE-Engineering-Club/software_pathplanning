from pymavlink import mavutil
import math

def get_heading(port="/dev/ttyACM0", baud=57600): # Update if needed
    """
    Connects to Pixhawk and returns heading in degrees.
    """
    try:
        master = mavutil.mavlink_connection(port, baud=baud)
        print("📡 Waiting for Pixhawk heartbeat...")
        master.wait_heartbeat()
        print("✅ Pixhawk connected")

        # Wait for a valid ATTITUDE message
        msg = master.recv_match(type='ATTITUDE', blocking=True)
        yaw_rad = msg.yaw
        heading_deg = (math.degrees(yaw_rad) + 360) % 360
        return heading_deg
    except Exception as e:
        print("❌ Heading error:", e)
        return None
