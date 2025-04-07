from PID import *
import numpy as np
from stable_baselines3 import SAC

# Arduino COM port
COM_PORT = "COM9"

# Using mac?
USING_MAC = False

# Target values
MOTOR_TARGET_ANGLE = 180  # degrees
PENDULUM_TARGET_ANGLE = 0  # degrees
MOTOR_TARGET_RPM = 0  # rpm (max 3500)

pid = PID()

model = SAC.load("../Trained Models/sac_7.zip", device="cpu")
last_angle = 0.0

# Main function to control. Must return the voltage (control signal) to apply to the motor.
def control_system(current_time, dt, motor_angle, pendulum_angle, rpm):
    global last_angle

    motor_angle = np.radians(motor_angle)
    pendulum_angle = np.radians(pendulum_angle)

    s0 = np.sin(motor_angle)
    c0 = np.cos(motor_angle)
    d0 = rpm * 2.0 * np.pi / 60.0
    s1 = np.sin(pendulum_angle)
    c1 = np.cos(pendulum_angle)


    # if (pendulum_angle - last_angle) < 1e-6 or dt < 1e-6:
    #  d1 = 0.0
    # else:
    #   d1 = (pendulum_angle - last_angle) / dt

    d1 = (pendulum_angle - last_angle) / dt

    last_angle = pendulum_angle #radians

    current_state = np.array([s0, c0, d0, s1, c1, d1], dtype=np.float32)

    # Stack and flatten for model input
    stacked_obs = np.stack(current_state, axis=0).flatten()

    # Predict action (voltage)
    action, _ = model.predict(stacked_obs, deterministic=True)
    action = np.clip(action[0], -6.0, 6.0)

    return action


# This function is used to tune the PID controller with the GUI.
def setPidParams(_pid):
    #pid.copy(_pid)  # Uncomment this line if you want to use the GUI to tune your PID live.
    return 0
