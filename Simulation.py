import sys
from math import gamma

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from numpy.ma.core import arctan2
import math

from stable_baselines3 import PPO, SAC


m_1 = 0.024  # Pendulum mass (kg)
l_1 = 0.128 / 2  # Pendulum length to CoM (m)
I_1 = 0.0000235  # Pendulum inertia about pivot (kg·m²)
m_0 = 0.053  # Arm mass (kg)
L_0 = 0.086  # Arm length (m)
I_0 = 0.0000572 + 0.00006  # Arm inertia about pivot (kg·m²)
g = 9.81  # Gravity (m/s²)
b_0 = 0.0004  # Viscous friction coefficient for arm (N·m·s/rad)
b_1 = 0.000003  # Viscous friction coefficient for pendulum (N·m·s/rad)
k = 0.002  # Torsional spring constant for the cable effect (N·m/rad)
K_m = 0.0431
R_m = 8.94

run_time = 20
dt = 0.01

times = []
voltages = []

# Load the trained model
model = SAC.load("Trained Models/sac_6.zip", device="cpu")

# Initialize state history for frame stacking. Start with arm and pendulum at 0º, zero velocity
frame_history = [np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0])]

def voltage(t, s0, c0, d0, s1, c1, d1):
    global last_state, dt
    # Current state
    #t0_dot = (arctan2(s0, c0) - arctan2(frame_history[-1][0],frame_history[-1][1])) / dt
    #t1_dot = (arctan2(s1, c1) - arctan2(frame_history[-1][3], frame_history[-1][4])) / dt
    current_state = np.array([s0, c0, d0, s1, c1, d1], dtype=np.float32)

    # Update last_state
    frame_history.pop(0)
    frame_history.append(current_state)

    # Stack and flatten for model input
    stacked_obs = np.stack(frame_history, axis=0).flatten()

    # Predict action (voltage)
    action, _ = model.predict(stacked_obs, deterministic=True)
    action = np.clip(action[0], -6.0, 6.0)

    times.append(t)
    voltages.append(action)

    return action  # Ensure within action limits

# System dynamics
def dynamics(t, x):
    s0, c0, d0, s1, c1, d1 = x

    torque = K_m*(voltage(t,s0, c0, d0, s1, c1, d1)-K_m*d0)/R_m

    #Based on Euler-Lagrange differential equation (original)
    #theta_0 is arm angle ; theta_1 is pendulum angle (0 upwards)

    # Mass matrix (left-hand side)
    #alpha = I_0 + m_1*L_0**2 + m_1*l_1**2*s1**2
    #beta = m_1*l_1**2*(2*s1*c1)
    #gamma = m_1*L_0*l_1*c1
    #sigma = m_1*L_0*l_1*s1

    #M = np.array([
    #    [alpha,gamma],
    #    [gamma, I_1 + m_1*l_1**2],
    #])

    # Right-hand side
    #f = np.array([
    #    torque - b_0*d0 + sigma*d1**2 - beta*d0*d1,
    #    -b_1*d1 + m_1*g*l_1*s1 + 0.5*beta*d0**2,
    #])

    # Based on Euler-Lagrange differential equation (adapted to real pendulum)
    #theta_0 is arm angle ; theta_1 is pendulum angle (0 downwards)
    alpha = I_0 + m_1 * L_0 ** 2 + m_1 * l_1 ** 2 * s1 ** 2
    beta = -m_1 * l_1 ** 2 * (2 * s1 * c1)
    gamma = -m_1 * L_0 * l_1 * c1
    sigma = m_1 * L_0 * l_1 * s1

    M = np.array([
        [-alpha, -gamma],
        [-gamma, -(I_1 + m_1 * l_1 ** 2)],
    ])

    # Right-hand side
    f = np.array([
        -torque + b_0 * d0 + k * arctan2(s0, c0) + sigma * d1 ** 2 - beta * d0 * d1,
        b_1 * d1 + m_1 * g * l_1 * s1 + 0.5 * beta * d0 ** 2,
    ])

    # Solve for accelerations: M * [theta_ddot, alpha_ddot] = f
    acc = np.linalg.solve(M, f)

    # State derivatives
    return [d0 * c0, -d0 * s0, acc[0], d1 * c1, -d1 * s1, acc[1]]

# Initial conditions: [s0, c0, d0, s1, c1, d1]
# Start with pendulum near upright (theta_1 = 0) and small perturbation
theta_0 = 0.0
theta_1 = 0.0
x0 = [np.sin(theta_0), np.cos(theta_0), 0.0, np.sin(theta_1), np.cos(theta_1), 0.0]

# Time span
t_span = (0, run_time)
t_eval = np.linspace(0, run_time, math.ceil(run_time/dt))

# Solve ODE
sol = solve_ivp(dynamics, t_span, x0, t_eval=t_eval, method='RK45')


# Extract results
s0 = sol.y[0]
c0 = sol.y[1]
theta_0 = np.arctan2(s0,c0)
theta_0_dot = sol.y[2]
s1 = sol.y[3]
c1 = sol.y[4]
theta_1 = np.arctan2(s1,c1)
theta_1_dot = sol.y[5]
t = sol.t

# After solving ODE
x_cm_dot = -L_0 * s0 * theta_0_dot - l_1 * c1 * theta_1_dot * s0 - l_1 * theta_0_dot * s1 * c0
y_cm_dot = L_0 * c0 * theta_0_dot + l_1 * c1 * theta_1_dot * c0 - l_1 * theta_0_dot * s1 * s0
z_cm_dot = -theta_1_dot * l_1 * s1
T = 0.5 * I_0 * theta_0_dot**2 + 0.5 * m_1 * (x_cm_dot**2 + y_cm_dot**2 + z_cm_dot**2) + 0.5 * I_1 * theta_1_dot**2
V = m_1 * g * l_1 * (c1+1)
E = T + V

plt.figure()
plt.scatter(t, E, label='Total Energy', s=1)
plt.xlabel('Time (s)')
plt.ylabel('Energy (J)')
plt.legend()
plt.grid(True)
plt.show()

# Plotting
plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.scatter(t, np.degrees(theta_0), label=r'$\theta_0$ (arm angle)', s=1)
plt.xlabel('Time (s)')
plt.ylabel('Angle (degrees)')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 2)
plt.scatter(t, np.degrees(theta_1), label=r'$\theta_1$ (pendulum angle)', s=1, color = "orange")
plt.xlabel('Time (s)')
plt.ylabel('Angle (degrees)')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 3)
plt.scatter(t, theta_0_dot * 60 / (2*np.pi), label=r'$\dot{\theta_0}$ (arm velocity)', s=1)
plt.scatter(t, theta_1_dot * 60 / (2*np.pi), label=r'$\dot{\theta_1}$ (pendulum velocity)', s=1)
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (rpm)')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(times, voltages, label='Voltage (Control Input)')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()