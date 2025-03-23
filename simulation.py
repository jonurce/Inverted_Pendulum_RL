import sys
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO


# Parameters (approximate values from QUBE-Servo 2 specs)
m_1 = 0.024  # Pendulum mass (kg)
l_1 = 0.128  # Pendulum length to CoM (m)
I_1 = 0.000131  # Pendulum inertia about pivot (kg·m²)
m_0 = 0.053  # Arm mass (kg)
L_0 = 0.086  # Arm length (m)
I_0 = 0.0000572  # Arm inertia about pivot (kg·m²)
g = 9.81  # Gravity (m/s²)
b_0 = 0.0003  # Viscous friction coefficient for arm (N·m·s/rad)
b_1 = 0.0005  # Viscous friction coefficient for pendulum (N·m·s/rad)
K_m = 0.0431
R_m = 8.94

# Motor voltage (set to 0 for free motion; can be a function of time or control input)
def voltage(t,s0, c0, s1, c1):
    return 5.0

# System dynamics
def dynamics(t, x):
    s0, c0, d0, s1, c1, d1 = x

    torque = K_m*(voltage(t,s0, c0, s1, c1)-K_m*d0)/R_m

    #Based on Euler-Lagrange differential equation
    # Mass matrix (left-hand side)
    M = np.array([
        [m_1 * L_0 ** 2 + m_1*l_1**2 * s1**2 + I_0,
         +m_1 * L_0 * l_1 * c1],
        [m_1*L_0*l_1*c1,
         I_1 + m_1*l_1**2],
    ])

    # Right-hand side
    f = np.array([
        torque - b_0*d0 + m_1*L_0*l_1*s1*d1**2 - m_1*l_1**2*s1*c1*d0*d1,
        -b_1*d1 + m_1*g*l_1*s1 + m_1*l_1**2*d0**2*s1*c1,
    ])

    # Solve for accelerations: M * [theta_ddot, alpha_ddot] = f
    acc = np.linalg.solve(M, f)

    # State derivatives
    return [d0*c0, -d0*s0, acc[0], d1*c1, -d1*s1, acc[1]]

# Initial conditions: [theta, theta_dot, alpha, alpha_dot]
# Start with pendulum near upright (alpha ≈ π) and small perturbation
theta_0 = 0.0
theta_1 = 1.0
x0 = [np.sin(theta_0), np.cos(theta_0), 0.0, np.sin(theta_1), np.cos(theta_1), 0.0]

# Time span
t_span = (0, 20)  # 5 seconds
t_eval = np.linspace(0, 20, 5000)

# Solve ODE
sol = solve_ivp(dynamics, t_span, x0, t_eval=t_eval, method='RK45')


# Extract results
s0 = sol.y[0]
c0 = sol.y[1]
theta = np.arctan2(s0,c0)
theta_dot = sol.y[2]
s1 = sol.y[3]
c1 = sol.y[4]
alpha = np.arctan2(s1,c1)
alpha_dot = sol.y[5]
t = sol.t

# After solving ODE
x_cm_dot = -L_0 * s0 * theta_dot - l_1 * c1 * alpha_dot * s0 - l_1 * theta_dot * s1 * c0
y_cm_dot = L_0 * c0 * theta_dot + l_1 * c1 * alpha_dot * c0 - l_1 * theta_dot * s1 * s0
z_cm_dot = -alpha_dot * l_1 * s1
T = 0.5 * I_0 * theta_dot**2 + 0.5 * m_1 * (x_cm_dot**2 + y_cm_dot**2 + z_cm_dot**2) + 0.5 * I_1 * alpha_dot**2
V = m_1 * g * l_1 * (c1+1)
E = T + V

plt.figure()
plt.plot(t, E, label='Total Energy')
plt.xlabel('Time (s)')
plt.ylabel('Energy (J)')
plt.legend()
plt.grid(True)
plt.show()

# Plotting
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(t, theta, label=r'$\theta$ (arm angle)')
plt.plot(t, alpha, label=r'$\alpha$ (pendulum angle)')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t, theta_dot, label=r'$\dot{\theta}$ (arm velocity)')
plt.plot(t, alpha_dot, label=r'$\dot{\alpha}$ (pendulum velocity)')
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (rad/s)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()