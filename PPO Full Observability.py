import numpy as np
from scipy.integrate import solve_ivp
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

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

class LossTrackingCallback(BaseCallback):
    def __init__(self, check_freq=1000, patience=10, loss_threshold=0.01, verbose=1):
        super(LossTrackingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.patience = patience
        self.loss_threshold = loss_threshold
        self.total_losses = []

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            logger = self.model.logger
            total_loss = logger.name_to_value.get('train/loss', np.nan)

            if not np.isnan(total_loss):
                self.total_losses.append(total_loss)

                if len(self.total_losses) > self.patience:
                    recent_losses = self.total_losses[-self.patience:]
                    loss_change = np.abs(np.mean(np.diff(recent_losses)))
                    avg_loss = np.mean(recent_losses)
                    if avg_loss < self.loss_threshold or loss_change < 0.005:
                        print(f"Loss converged: Avg = {avg_loss:.4f} < {self.loss_threshold} "
                              f"or Change = {loss_change:.4f} < 0.005. Stopping.")
                        return False
        return True

class QubeServo2Env(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_limit = 10.0
        self.action_space = spaces.Box(low=-self.action_limit, high=self.action_limit, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)  # Full state
        self.state = None
        self.dt = 0.01
        self.max_steps = 1000
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        alpha = np.random.uniform(-np.pi, np.pi)
        self.state = np.array([0.0, 1.0, 0.0, np.sin(alpha), np.cos(alpha), 0.0], dtype=np.float32)
        self.step_count = 0
        return self.state, {}  # Return full state: [theta, theta_dot, alpha, alpha_dot]

    def step(self, action):
        voltage = np.clip(action[0], -self.action_limit, self.action_limit)
        torque = K_m * (voltage - K_m * self.state[2]) / R_m
        def dynamics(t, x):
            s0, c0, d0, s1, c1, d1 = x

            # Based on Euler-Lagrange differential equation
            # Left-hand side
            M = np.array([
                [m_1 * L_0 ** 2 + m_1 * l_1 ** 2 * s1 ** 2 + I_0,
                 +m_1 * L_0 * l_1 * c1],
                [m_1 * L_0 * l_1 * c1,
                 I_1 + m_1 * l_1 ** 2],
            ])

            # Right-hand side
            f = np.array([
                torque - b_0 * d0 + m_1 * L_0 * l_1 * s1 * d1 ** 2 - m_1 * l_1 ** 2 * s1 * c1 * d0 * d1,
                -b_1 * d1 + m_1 * g * l_1 * s1 + m_1 * l_1 ** 2 * d0 ** 2 * s1 * c1,
            ])

            acc = np.linalg.solve(M, f)

            return [d0*c0, -d0*s0, acc[0], d1*c1, -d1*s1, acc[1]]
        sol = solve_ivp(dynamics, [0, self.dt], self.state, method='RK45')
        self.state = sol.y[:, -1]
        dyn = dynamics(0, self.state)

        s0 = self.state[0]
        c0 = self.state[1]
        d0 = self.state[2]
        dd0 = dyn[2]
        s1 = self.state[3]
        c1 = self.state[4]
        d1 = self.state[5]
        dd1 = dyn[5]

        x_cm_dot = -L_0 * s0 * d0 - l_1 * c1 * d1 * s0 - l_1 * d0 * s1 * c0
        y_cm_dot = L_0 * c0 * d0 + l_1 * c1 * d1 * c0 - l_1 * d0 * s1 * s0
        z_cm_dot = -d1 * l_1 * s1
        T = 0.5 * I_0 * d0 ** 2 + 0.5 * m_1 * (x_cm_dot ** 2 + y_cm_dot ** 2 + z_cm_dot ** 2) + 0.5 * I_1 * d1 ** 2
        V = m_1 * g * l_1 * (c1 + 1)
        E = T + V
        E_r = 2 * m_1 * g * l_1

        reward = (
            -np.arctan2(s1,c1)**2
            - 0.7*np.arctan2(s0,c0)**2
            # - 1500*abs(torque_value)
            # - 0.1 * (theta_dot**2 + alpha_dot**2)
            #- 0.05 * (theta_ddot**2 + alpha_ddot**2)
            #-(E-E_r)**2
        )
        self.step_count += 1
        #done = abs(theta) > 2 * np.pi / 3 or self.step_count >= self.max_steps
        done = self.step_count >= self.max_steps
        truncated = self.step_count >= self.max_steps
        return self.state, reward, done, truncated, {}  # Return full state

# Train with frame stacking
env = DummyVecEnv([lambda: QubeServo2Env()])
env = VecFrameStack(env, n_stack=2)  # Stacks 6D states: 6 × 2 = 12D

# Train PPO
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0001, n_steps=2048)
callback = LossTrackingCallback(check_freq=5000, patience=10, loss_threshold=0.01, verbose=1)
model.learn(total_timesteps=400000, callback=callback)
model.save("pendulum_ppo")

# Test and collect data
env = QubeServo2Env()
obs, _ = env.reset()
frame_history = [obs] * 2  # obs is now 6D
rewards = []
thetas = []
alphas = []
voltages = []
times = []

for i in range(1000):
    stacked_obs = np.stack(frame_history, axis=0).flatten()  # 6 × 2 = 12D flattened
    action, _states = model.predict(stacked_obs, deterministic=True)
    obs, reward, done, truncated, _ = env.step(action)
    frame_history.pop(0)
    frame_history.append(obs)
    rewards.append(reward)
    thetas.append(np.arctan2(obs[0],obs[1]))
    alphas.append(np.arctan2(obs[3],obs[4]))
    voltages.append(action[0])
    times.append(i * env.dt)
    print(f"Step {i}: Theta = {np.arctan2(obs[0],obs[1]):.3f}, Alpha = {np.arctan2(obs[3],obs[4]):.3f}, Voltage = {action[0]:.3f}, Reward = {reward:.3f}")
    if done or truncated:
        obs, _ = env.reset()
        frame_history = [obs] * 2

# Plotting results
plt.figure(figsize=(12, 10))
plt.subplot(3, 1, 1)
plt.plot(times, thetas, label=r'$\theta_0$ (arm angle)')
plt.plot(times, alphas, label=r'$\theta_1$ (pendulum angle)')
plt.axhline(y=np.pi, color='r', linestyle='--', label=r'$\theta_1 = +-\pi$ (down)')
plt.axhline(y=-np.pi, color='r', linestyle='--')
plt.axhline(y=2*np.pi/3, color='k', linestyle='--', label=r'$\theta_0$ limits')
plt.axhline(y=-2*np.pi/3, color='k', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(times, rewards, label='Reward')
plt.xlabel('Time (s)')
plt.ylabel('Reward')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(times, voltages, label='Torque')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()