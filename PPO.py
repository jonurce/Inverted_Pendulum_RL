import numpy as np
from scipy.integrate import solve_ivp
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt

# Parameters
m_p = 0.024
L_p = 0.129
I_p = 0.0000995
m_a = 0.094
L_a = 0.085
I_a = 0.000534
g = 9.81
b_a = 0.001
b_p = 0.0005


# Custom Gym Environment
class QubeServo2Env(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.state = None
        self.dt = 0.02  # 50 Hz
        self.max_steps = 500  # 10 seconds
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([0.0, 0.0, np.pi + np.random.uniform(-0.1, 0.1), 0.0], dtype=np.float32)
        self.step_count = 0
        return self.state, {}

    def step(self, action):
        torque_value = np.clip(action[0], -1.0, 1.0)

        def dynamics(t, x):
            x1, x2, x3, x4 = x
            M = np.array([
                [I_a + m_p * L_a ** 2 + 2 * m_p * L_a * L_p * np.sin(x3) * np.sin(x1) * np.cos(x1),
                 m_p * L_a * L_p * np.cos(x3)],
                [m_p * L_a * L_p * np.cos(x3) + 2 * m_p * (L_p ** 2) * np.sin(x3) * np.cos(x3) * np.sin(x1) * np.cos(
                    x1), I_p + m_p * L_p ** 2]
            ])
            f = np.array([
                torque_value + m_p * L_a * L_p * np.sin(x3) * x4 ** 2 + m_p * L_a * L_p * np.sin(x3) * x2 ** 2 * np.cos(
                    2 * x1) -
                2 * L_p * x3 * x2 * np.cos(x3) * np.cos(x1) * m_p * L_a * np.sin(x1) - b_a * x2,
                -2 * m_p * g * L_p * np.sin(x3) + m_p * L_p ** 2 * x2 ** 2 * np.sin(x3) * np.cos(x3) * np.cos(2 * x1) -
                2 * m_p * L_p ** 2 * x2 * x4 * np.cos(x3) ** 2 * np.sin(x1) * np.cos(x1) - b_p * x4
            ])
            acc = np.linalg.solve(M, f)
            return [x2, acc[0], x4, acc[1]]

        sol = solve_ivp(dynamics, [0, self.dt], self.state, method='RK45')
        self.state = sol.y[:, -1]

        reward = np.cos(self.state[2]) - 0.1 * (self.state[1] ** 2 + self.state[3] ** 2)
        self.step_count += 1
        done = abs(self.state[2] - np.pi) > np.pi / 2 or self.step_count >= self.max_steps
        truncated = self.step_count >= self.max_steps

        return self.state, reward, done, truncated, {}

    def render(self, mode='human'):
        pass


# Verify environment
env = QubeServo2Env()
check_env(env)

# Train PPO
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048)
model.learn(total_timesteps=100000)
model.save("qube_servo2_ppo")

# Test and collect data
env = QubeServo2Env()
obs, _ = env.reset()
rewards = []
thetas = []
alphas = []
times = []

for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, _ = env.step(action)
    rewards.append(reward)
    thetas.append(obs[0])
    alphas.append(obs[2])
    times.append(i * env.dt)
    # print(f"Step {i}: Theta = {obs[0]:.3f}, Alpha = {obs[2]:.3f}, Torque = {action[0]:.3f}, Reward = {reward:.3f}")
    if done or truncated:
        obs, _ = env.reset()

# Plotting
plt.figure(figsize=(12, 8))

# Angles
plt.subplot(2, 1, 1)
plt.plot(times, thetas, label=r'$\theta$ (arm angle)')
plt.plot(times, alphas, label=r'$\alpha$ (pendulum angle)')
plt.axhline(y=np.pi, color='r', linestyle='--', label=r'$\alpha = \pi$ (upright)')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.legend()
plt.grid(True)

# Reward
plt.subplot(2, 1, 2)
plt.plot(times, rewards, label='Reward')
plt.xlabel('Time (s)')
plt.ylabel('Reward')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()