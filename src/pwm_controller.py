from pymavlink import mavutil

master = None

def connect_pixhawk(port="/dev/ttyACM0", baudrate=57600):
    global master
    master = mavutil.mavlink_connection(port, baud=baudrate)
    master.wait_heartbeat()
    print("✅ Pixhawk Connected")

def send_pwm(left, right, front):
    if master:
        master.mav.command_long_send(
            master.target_system, master.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
            0, 1, left, 2, right, 3, front, 0
        )
        print(f"⚙️ PWM Sent: Left={left}, Right={right}, Front={front}")
