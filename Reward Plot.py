import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
import matplotlib
matplotlib.use('TkAgg')  # Set interactive backend explicitly

# Define constants
g = 9.81
l_1 = 0.128 / 2
m_1 = 0.024
I_1 = 0.0000235
max_theta_0 = 5.0 * np.pi / 6.0

# Reward function
def compute_reward(theta0, theta1, d0, d1):
    s0, c0 = np.sin(theta0), np.cos(theta0)
    s1, c1 = np.sin(theta1), np.cos(theta1)

    upright_reward = -2.0 * c1

    velocity_penalty = -0.3 * np.tanh((d0**2 + d1**2) / 10.0)

    pos_penalty = -0.1 * np.tanh(theta0**2 / 2.0)

    upright_closeness = np.exp(-10.0 * (abs(theta1) - np.pi)**2)
    stability_factor = np.exp(-1.0 * d1**2)
    bonus = 10.0 * upright_closeness * stability_factor

    downright_closeness = np.exp(-1.0 * theta1**2)
    stability_factor = np.exp(-1.0 * d1 ** 2)
    bonus += -10.0 * downright_closeness * stability_factor

    limit_distance = np.clip(1.0 - 0.5*(max_theta_0 - abs(theta0)), 0, 1)
    limit_penalty = -20.0 * limit_distance**3

    energy_reward = 2 - 0.15 * abs(m_1 * g * l_1 * (c1 + 1.0) + 0.5 * I_1 * d1**2)

    return upright_reward + pos_penalty + bonus + limit_penalty + energy_reward

# Create meshgrid
theta0 = np.linspace(-np.pi, np.pi, 720)  # Reduced for responsiveness
theta1 = np.linspace(-np.pi, np.pi, 720)
theta0, theta1 = np.meshgrid(theta0, theta1)

# Initial values
d0_init = 0.0
d1_init = 0.0
rewards = compute_reward(theta0, theta1, d0_init, d1_init)

# Create figure
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(theta0, theta1, rewards, cmap='viridis')
ax.set_xlabel('Theta0 (rad)')
ax.set_ylabel('Theta1 (rad)')
ax.set_zlabel('Reward')
ax.set_title('Reward Function with Velocity Sliders')
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.view_init(elev=30, azim=45)

# Adjust layout to make room for sliders
plt.subplots_adjust(bottom=0.25)

# Add sliders
ax_d0 = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_d1 = plt.axes([0.25, 0.10, 0.65, 0.03])
s_d0 = Slider(ax_d0, 'd0 (rad/s)', -10.0, 10.0, valinit=d0_init)
s_d1 = Slider(ax_d1, 'd1 (rad/s)', -10.0, 10.0, valinit=d1_init)

# Update function
def update(val):
    ax.clear()
    d0 = s_d0.val
    d1 = s_d1.val
    rewards = compute_reward(theta0, theta1, d0, d1)
    ax.plot_surface(theta0, theta1, rewards, cmap='viridis')
    ax.set_xlabel('Theta0 (rad)')
    ax.set_ylabel('Theta1 (rad)')
    ax.set_zlabel('Reward')
    ax.set_title('Reward Function with Velocity Sliders')
    ax.view_init(elev=30, azim=45)
    fig.canvas.draw_idle()

# Connect sliders
s_d0.on_changed(update)
s_d1.on_changed(update)

plt.show()