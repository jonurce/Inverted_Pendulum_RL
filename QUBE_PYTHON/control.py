from PID import *
import numpy as np
from stable_baselines3 import SAC

# Arduino COM port
COM_PORT = "COM8"

# Using mac?
USING_MAC = False

# Target values
MOTOR_TARGET_ANGLE = 0  # degrees
PENDULUM_TARGET_ANGLE = 180  # degrees
MOTOR_TARGET_RPM = 0  # rpm (max 3500)

pid = PID()

model_full = SAC.load("../Trained Models/sac_21.zip", device="cpu")
last_angle = 0.0

# Main function to control. Must return the voltage (control signal) to apply to the motor.
def control_system_RL_full(current_time, dt, motor_angle, pendulum_angle, rpm):
    global last_angle

    # Degrees to radians
    motor_angle = motor_angle * np.pi / 180
    pendulum_angle = pendulum_angle * np.pi / 180

    # Sin and cos of both angles
    s0 = np.sin(motor_angle)
    c0 = np.cos(motor_angle)
    s1 = np.sin(pendulum_angle)
    c1 = np.cos(pendulum_angle)

    #RPM to rad/s
    d0 = rpm * 2.0 * np.pi / 60.0

    if abs(pendulum_angle - last_angle) < 0.000001 or dt < 0.000001:
        d1 = 0.0
    else:
        # d1 estimation in rad/s
        d1 = (pendulum_angle - last_angle) / dt

    # Save last angle in radian
    last_angle = pendulum_angle

    # Full state for model inference
    current_state = np.array([s0, c0, d0, s1, c1, d1], dtype=np.float32)
    stacked_obs = np.stack(current_state, axis=0).flatten()

    # Predict action (voltage)
    action, _ = model_full.predict(stacked_obs, deterministic=True)
    action = np.clip(action[0], -6.0, 6.0)
    return action

model_partial = SAC.load("../Trained Models/sac_12.zip", device="cpu")
n_frames = 5
frame_history = [np.array([0.0, 1.0, 0.0, 0.0, 1.0])] * n_frames

def control_system_RL_partial(current_time, dt, motor_angle, pendulum_angle, rpm):
    # Degrees to radians
    motor_angle = motor_angle * np.pi / 180
    pendulum_angle = pendulum_angle * np.pi / 180

    # Sin and cos of both angles
    s0 = np.sin(motor_angle)
    c0 = np.cos(motor_angle)
    s1 = np.sin(pendulum_angle)
    c1 = np.cos(pendulum_angle)

    #RPM to rad/s
    d0 = rpm * 2.0 * np.pi / 60.0

    # Current partial state
    current_state = np.array([s0, c0, d0, s1, c1], dtype=np.float32)

    # Update frame history, kick out older state, insert current state
    frame_history.pop(0)
    frame_history.append(current_state)
    stacked_obs = np.stack(frame_history, axis=0).flatten()

    # Predict action (voltage)
    action, _ = model_partial.predict(stacked_obs, deterministic=True)
    action = np.clip(action[0], -6.0, 6.0)
    return action

def control_system(current_time, dt, motor_angle, pendulum_angle, rpm):
    if current_time < 0.2:
        return 3
    return 0


# This function is used to tune the PID controller with the GUI.
def setPidParams(_pid):
    #pid.copy(_pid)  # Uncomment this line if you want to use the GUI to tune your PID live.
    return 0
