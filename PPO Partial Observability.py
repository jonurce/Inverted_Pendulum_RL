import numpy as np
from scipy.integrate import solve_ivp
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

# Parameters (approximate values from QUBE-Servo 2 specs)
m_p = 0.024  # Pendulum mass (kg)
L_p = 0.128  # Pendulum length to CoM (m)
I_p = 0.000131  # Pendulum inertia about pivot (kg·m²)
m_a = 0.053  # Arm mass (kg)
L_a = 0.086  # Arm length (m)
I_a = 0.0000572  # Arm inertia about pivot (kg·m²)
g = 9.81  # Gravity (m/s²)
b_a = 0.0003  # Viscous friction coefficient for arm (N·m·s/rad)
b_p = 0.0005  # Viscous friction coefficient for pendulum (N·m·s/rad)

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
        self.action_space = spaces.Box(low=-0.3, high=0.3, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        self.state = None
        self.dt = 0.01
        self.max_steps = 1000
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([0.0, 0.0, np.random.uniform(-1.0, 1.0), 0.0], dtype=np.float32)
        self.step_count = 0
        return self.state[[0, 2]], {}

    def step(self, action):
        torque_value = np.clip(action[0], -0.3, 0.3)
        def dynamics(t, x):
            x1, x2, x3, x4 = x
            x2 = np.clip(x2, -5, 5)
            x4 = np.clip(x4, -5, 5)
            # Based on Euler-Lagrange differential equation (copied from document)
            # Mass matrix (left-hand side)
            M = np.array([
                [m_p * L_a ** 2 + 0.25 * m_p * L_p ** 2 - 0.25 * m_p * L_p ** 2 * np.cos(x3) ** 2 + I_a,
                 -0.5 * m_p * L_a * L_p * np.cos(x3)],
                [0.5 * m_p * L_a * L_p * np.cos(x3),
                 I_p + 0.25 * m_p * L_p ** 2],
            ])

            # Right-hand side
            f = np.array([
                torque_value - b_a * x2 - 0.5 * m_p * L_a * L_p * np.sin(
                    x3) * x4 ** 2 - 0.5 * m_p * L_p ** 2 * np.sin(x3) * np.cos(x3) * x2 * x4,
                -b_p * x4 - 0.5 * m_p * g * L_p * np.sin(x3) + 0.25 * m_p * L_p ** 2 * x2 ** 2 * np.sin(x3) * np.cos(
                    x3),
            ])
            acc = np.linalg.solve(M, f)
            acc[0] = np.clip(acc[0], -5, 5)
            acc[1] = np.clip(acc[1], -5, 5)
            return [x2, acc[0], x4, acc[1]]
        sol = solve_ivp(dynamics, [0, self.dt], self.state, method='RK45')
        self.state = sol.y[:, -1]
        dyn = dynamics(0, self.state)
        dyn = np.clip(dyn, -5, 5)
        theta = self.state[0]
        theta_dot = dyn[0]
        theta_ddot = dyn[1]
        alpha = self.state[2]
        alpha_dot = dyn[2]
        alpha_ddot = dyn[3]
        reward = (
            - 2*(np.pi - abs(alpha))**2
            - 0.5 * theta**2
            - 0.01 * torque_value**2
            - 0.0005 * (theta_dot**2 + alpha_dot**2)
            - 0.0005 * (theta_ddot**2 + alpha_ddot**2)
        )
        self.step_count += 1
        done = abs(theta) > 5 * np.pi / 6 or self.step_count >= self.max_steps
        truncated = self.step_count >= self.max_steps
        return self.state[[0, 2]], reward, done, truncated, {}

# Train with frame stacking
env = DummyVecEnv([lambda: QubeServo2Env()])
env = VecFrameStack(env, n_stack=8)

# Train PPO
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.00003, n_steps=2048, clip_range_vf=0.2)
callback = LossTrackingCallback(check_freq=5000, patience=10, loss_threshold=0.05, verbose=1)
model.learn(total_timesteps=100000, callback=callback)
model.save("pendulum_ppo_angles_loss_stop")

# Test and collect data
env = QubeServo2Env()
obs, _ = env.reset()
frame_history = [obs] * 8
rewards = []
thetas = []
alphas = []
torques = []
times = []

for i in range(1000):
    stacked_obs = np.stack(frame_history, axis=0).flatten()
    action, _states = model.predict(stacked_obs, deterministic=True)
    obs, reward, done, truncated, _ = env.step(action)
    frame_history.pop(0)
    frame_history.append(obs)
    rewards.append(reward)
    thetas.append(obs[0])
    alphas.append(obs[1])
    torques.append(action[0])
    times.append(i * env.dt)
    print(f"Step {i}: Theta = {obs[0]:.3f}, Alpha = {obs[1]:.3f}, Torque = {action[0]:.3f}, Reward = {reward:.3f}")
    if done or truncated:
        obs, _ = env.reset()
        frame_history = [obs] * 8

# Plotting results
plt.figure(figsize=(12, 10))
plt.subplot(3, 1, 1)
plt.plot(times, thetas, label=r'$\theta$ (arm angle)')
plt.plot(times, alphas, label=r'$\alpha$ (pendulum angle)')
plt.axhline(y=np.pi, color='r', linestyle='--', label=r'$\alpha = \pi$ (upright)')
plt.axhline(y=5*np.pi/6, color='k', linestyle='--', label=r'$\theta$ limit')
plt.axhline(y=-5*np.pi/6, color='k', linestyle='--')
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
plt.plot(times, torques, label='Torque')
plt.xlabel('Time (s)')
plt.ylabel('Torque (N·m)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()