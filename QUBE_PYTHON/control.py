from PID import *
import numpy as np

# Arduino COM port
COM_PORT = "COM9"

# Using mac?
USING_MAC = False

# Target values
MOTOR_TARGET_ANGLE = 180  # degrees
PENDULUM_TARGET_ANGLE = 0  # degrees
MOTOR_TARGET_RPM = 0  # rpm (max 3500)

pid = PID()


# Main function to control. Must return the voltage (control signal) to apply to the motor.
def control_system(current_time, dt, motor_angle, pendulum_angle, rpm):
    if current_time < 0.1:
        return 10.0
    return 0.0


# This function is used to tune the PID controller with the GUI.
def setPidParams(_pid):
    #pid.copy(_pid)  # Uncomment this line if you want to use the GUI to tune your PID live.
    return 0
