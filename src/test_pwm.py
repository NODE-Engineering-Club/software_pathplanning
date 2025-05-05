from pwm_controller import connect_pixhawk, send_single_pwm
import time

connect_pixhawk("/dev/ttyACM0")

# Spin motor at 40% for 2 seconds
send_single_pwm(40)
time.sleep(2)

# Stop motor
send_single_pwm(0)

#send_single_pwm(40) sends ~1420 PWM → moderate speed
#send_single_pwm(0) sends 1100 PWM → stop