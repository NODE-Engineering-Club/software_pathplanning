from pymavlink import mavutil

master = None

def connect_pixhawk(port="/dev/ttyACM0", baudrate=57600):
    global master
    master = mavutil.mavlink_connection(port, baud=baudrate)
    master.wait_heartbeat()
    print("✅ Pixhawk connected")

def send_single_pwm(pwm_percent):
    """
    Sends PWM to only one motor (e.g., left propeller on SERVO1).
    pwm_percent should be between 0–100.
    """
    if master is None:
        print("❌ Pixhawk not connected")
        return

    # Convert 0–100% duty cycle to RC PWM value (1100–1900)
    pwm_value = int(1100 + (pwm_percent / 100) * 800)

    print(f"⚙️ Sending PWM to Left Motor: {pwm_value}")

    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
        0,
        1, pwm_value,  # SERVO1 = left motor
        0, 0, 0, 0, 0   # Others unused
    )
