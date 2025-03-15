import sys
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# Parameters (approximate values from QUBE-Servo 2 specs)
m_p = 0.024  # Pendulum mass (kg)
L_p = 0.129  # Pendulum length to CoM (m)
I_p = 0.0000995  # Pendulum inertia about pivot (kg·m²)
m_a = 0.094  # Arm mass (kg)
L_a = 0.085  # Arm length (m)
I_a = 0.000534  # Arm inertia about pivot (kg·m²)
g = 9.81  # Gravity (m/s²)
b_a = 0.001  # Viscous friction coefficient for arm (N·m·s/rad)
b_p = 0.0005  # Viscous friction coefficient for pendulum (N·m·s/rad)

# Motor torque (set to 0 for free motion; can be a function of time or control input)
def torque(t,theta,alpha):
    return np.sin(t)*0.1

# System dynamics
def dynamics(t, x):
    x1, x2, x3, x4 = x  # theta, theta_dot, alpha, alpha_dot

    # Mass matrix (left-hand side)
    M = np.array([
       [I_a + m_p * L_a**2 + 2*m_p*L_a*L_p*np.sin(x3)*np.sin(x1)*np.cos(x1), m_p * L_a * L_p * np.cos(x3)],
        [m_p*L_a*L_p*np.cos(x3)+2*m_p*(L_p**2)*np.sin(x3)*np.cos(x3)*np.sin(x1)*np.cos(x1), I_p + m_p*L_p**2],
    ])

    # Right-hand side
    f = np.array([
       torque(t,x1,x3) + m_p * L_a * L_p * np.sin(x3) * x4**2 + m_p*L_a*L_p*np.sin(x3)*x2**2*np.cos(2*x1) - 2*L_p*x3*x2*np.cos(x3)*np.cos(x1)*m_p*L_a*np.sin(x1) - b_a*x2,
        -2*m_p*g*L_p*np.sin(x3) + m_p*L_p**2*x2**2*np.sin(x3)*np.cos(x3)*np.cos(2*x1) - 2*m_p*L_p**2*x2*x4*np.cos(x3)**2*np.sin(x1)*np.cos(x1) - b_p*x4,
    ])

    # Solve for accelerations: M * [theta_ddot, alpha_ddot] = f
    accelerations = np.linalg.solve(M, f)
    theta_ddot = accelerations[0]
    alpha_ddot = accelerations[1]

    # State derivatives
    dxdt = [x2, theta_ddot, x4, alpha_ddot]
    return dxdt

# Initial conditions: [theta, theta_dot, alpha, alpha_dot]
# Start with pendulum near upright (alpha ≈ π) and small perturbation
x0 = [0.0, 0.0, np.pi/3, 0.0]

# Time span
t_span = (0, 10)  # 5 seconds
t_eval = np.linspace(0, 10, 1000)

# Solve ODE
sol = solve_ivp(dynamics, t_span, x0, t_eval=t_eval, method='RK45')


# Extract results
theta = sol.y[0]
theta_dot = sol.y[1]
alpha = sol.y[2]
alpha_dot = sol.y[3]
t = sol.t

# After solving ODE
x_cm_dot = -L_a * np.sin(theta) * theta_dot - L_p * np.cos(alpha) * alpha_dot * np.sin(theta) - L_p * theta_dot * np.sin(alpha) * np.cos(theta)
y_cm_dot = L_a * np.cos(theta) * theta_dot + L_p * np.cos(alpha) * alpha_dot * np.cos(theta) + L_p * theta_dot * np.sin(alpha) * np.sin(theta)
T = 0.5 * I_a * theta_dot**2 + 0.5 * m_p * (x_cm_dot**2 + y_cm_dot**2) + 0.5 * I_p * alpha_dot**2
V = m_p * g * L_p * (1 - np.cos(alpha))
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