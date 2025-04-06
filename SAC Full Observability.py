import numpy as np
from numpy.ma.core import arctan2
from scipy.integrate import solve_ivp
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class TrainingMonitorCallback(BaseCallback):
    def __init__(self, check_freq=1000, patience=10, loss_threshold=0.01, verbose=1):
        super(TrainingMonitorCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.patience = patience
        self.loss_threshold = loss_threshold
        self.loss_change = 0.005  # Threshold for loss change
        self.total_losses = []
        self.rewards = []
        self.episode_rewards = []
        self.episode_lengths = []
        self.theta1_values = []

    def _on_step(self):
        # Collect step data
        reward = self.locals['rewards'][0]
        self.rewards.append(reward)

        # Extract theta_1 from observation
        obs = self.locals['new_obs'][0]
        s1, c1 = obs[3], obs[4]
        theta1 = np.arctan2(s1, c1)
        self.theta1_values.append(theta1)

        # Track episode completion
        if self.locals['dones'][0]:
            self.episode_rewards.append(sum(self.rewards))
            self.episode_lengths.append(len(self.rewards))
            self.rewards = []

        # Periodic logging and early stopping check
        if self.n_calls % self.check_freq == 0:
            logger = self.model.logger
            actor_loss = logger.name_to_value.get('train/actor_loss', np.nan)
            critic_loss = logger.name_to_value.get('train/critic_loss', np.nan)
            total_loss = np.nanmean([actor_loss, critic_loss]) if not (np.isnan(actor_loss) or np.isnan(critic_loss)) else np.nan

            # Loss tracking for early stopping
            if not np.isnan(total_loss):
                self.total_losses.append(total_loss)
                if len(self.total_losses) > self.patience:
                    recent_losses = self.total_losses[-self.patience:]
                    loss_change = np.abs(np.mean(np.diff(recent_losses)))
                    avg_loss = np.mean(recent_losses)
                    if abs(avg_loss) < self.loss_threshold or loss_change < self.loss_change:
                        print(f"Loss converged: Avg = {avg_loss:.4f} < {self.loss_threshold} "
                              f"or Change = {loss_change:.4f} < 0.005. Stopping.")
                        return False

            # Additional monitoring metrics
            avg_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
            avg_theta1 = np.mean(np.abs(np.degrees(self.theta1_values[-self.check_freq:])))
            success_rate = np.mean(np.abs(self.theta1_values[-self.check_freq:]) > np.pi / 2)

            # Print to terminal
            print(f"Step {self.n_calls}")

            # Log to TensorBoard
            self.logger.record('custom/avg_episode_reward', avg_reward)
            self.logger.record('custom/avg_theta1_degrees', avg_theta1)
            self.logger.record('custom/success_rate', success_rate)
            self.logger.record('custom/combined_loss', total_loss)
            self.logger.dump(step=self.n_calls)

        return True  # Continue training unless stopped

class QubeServo2Env(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_limit = 6.0
        self.action_space = spaces.Box(low=-self.action_limit, high=self.action_limit, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)  # Full state
        self.state = None
        self.dt = 0.01
        self.max_steps = 2000
        self.step_count = 0
        self.max_theta_0 = 5.0 * np.pi / 6.0

        # Nominal parameters (moved from global scope)
        self.nominal_params = {
            'm_1': 0.024,  # Pendulum mass (kg)
            'l_1': 0.128 / 2,  # Pendulum length to CoM (m)
            'I_1': 0.0000235,  # Pendulum inertia (kg·m²)
            'm_0': 0.053,  # Arm mass (kg)
            'L_0': 0.086,  # Arm length (m)
            'I_0': 0.0000572 + 0.00006,  # Arm inertia (kg·m²)
            'g': 9.81,  # Gravity (m/s²)
            'b_0': 0.0004,  # Arm friction (N·m·s/rad)
            'b_1': 0.000003,  # Pendulum friction (N·m·s/rad)
            'k': 0.002,  # Torsional spring constant (N·m/rad)
            'K_m': 0.0431,  # Motor constant
            'R_m': 8.94  # Motor resistance
        }
        # Current parameters (will be randomized)
        self.params = self.nominal_params.copy()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)  # Ensure reproducibility per environment

            # Randomize parameters at the start of each episode
        self.params = self.nominal_params.copy()
        self.params['m_1'] = self.nominal_params['m_1'] * np.random.uniform(0.8, 1.2)  # ±20%
        self.params['l_1'] = self.nominal_params['l_1'] * np.random.uniform(0.8, 1.2)  # ±20%
        self.params['I_1'] = self.nominal_params['I_1'] * np.random.uniform(0.8, 1.2)  # ±20%
        self.params['m_0'] = self.nominal_params['m_0'] * np.random.uniform(0.8, 1.2)  # ±20%
        self.params['L_0'] = self.nominal_params['L_0'] * np.random.uniform(0.8, 1.2)  # ±20%
        self.params['I_0'] = self.nominal_params['I_0'] * np.random.uniform(0.8, 1.2)  # ±20%
        self.params['g'] = self.nominal_params['g'] * np.random.uniform(0.95, 1.05)  # ±5%
        self.params['b_0'] = self.nominal_params['b_0'] * np.random.uniform(0.8, 1.2)  # ±20%
        self.params['b_1'] = self.nominal_params['b_1'] * np.random.uniform(0.8, 1.2)  # ±20%
        self.params['k'] = self.nominal_params['k'] * np.random.uniform(0.8, 1.2)  # ±20%
        self.params['K_m'] = self.nominal_params['K_m'] * np.random.uniform(0.8, 1.2)  # ±20%
        self.params['R_m'] = self.nominal_params['R_m'] * np.random.uniform(0.8, 1.2)  # ±20%

        theta0 = np.random.uniform(-np.pi / 6, np.pi / 6)  # ±30° for arm
        theta1 = np.random.uniform(-np.pi, np.pi)  # Anywhere for pendulum
        self.state = np.array([np.sin(theta0), np.cos(theta0), 0.0, np.sin(theta1), np.cos(theta1), 0.0], dtype=np.float32)
        self.step_count = 0
        return self.state, {}

    def step(self, action):

        m_1 = self.params['m_1']
        l_1 = self.params['l_1']
        I_1 = self.params['I_1']
        m_0 = self.params['m_0']
        L_0 = self.params['L_0']
        I_0 = self.params['I_0']
        g = self.params['g']
        b_0 = self.params['b_0']
        b_1 = self.params['b_1']
        k = self.params['k']
        K_m = self.params['K_m']
        R_m = self.params['R_m']

        voltage = np.clip(action[0], -self.action_limit, self.action_limit)
        torque = K_m * (voltage - K_m * self.state[2]) / R_m
        def dynamics(t, x):
            s0, c0, d0, s1, c1, d1 = x

            # Based on Euler-Lagrange differential equation (adapted to real pendulum)
            # theta_0 is arm angle ; theta_1 is pendulum angle (0 downwards)
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
                -torque + b_0 * d0 + k * np.arctan2(s0, c0) + sigma * d1 ** 2 - beta * d0 * d1,
                b_1 * d1 + m_1 * g * l_1 * s1 + 0.5 * beta * d0 ** 2,
            ])

            acc = np.linalg.solve(M, f)

            return [d0*c0, -d0*s0, acc[0], d1*c1, -d1*s1, acc[1]]
        sol = solve_ivp(dynamics, [0, self.dt], self.state, method='RK45')
        dyn = dynamics(0, self.state)
        self.state = sol.y[:, -1]

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
        V = m_1 * g * l_1 * (1 - c1)
        E = T + V
        E_r = 2 * m_1 * g * l_1

        # COMPONENT 1: Base reward for pendulum being upright (range: -1 to 1)
        # Uses cosine which is a naturally smooth function
        upright_reward = - 2.0 * c1

        # COMPONENT 2: Smooth penalty for high velocities - quadratic falloff
        # Use tanh to create a smoother penalty that doesn't grow excessively large
        velocity_penalty = -0.3 * np.tanh((d0 ** 2 + d1 ** 2) / 10.0)

        # COMPONENT 3: Smooth penalty for arm position away from center
        # Again using tanh for smooth bounded penalties
        pos_penalty = -0.1 * np.tanh(arctan2(s0,c0) ** 2 / 2.0)

        # COMPONENT 4: Smoother bonus for being close to upright position
        upright_closeness = np.exp(-10.0 * (abs(arctan2(s1,c1))-np.pi) ** 2)  # Close to 1 when near upright, falls off quickly
        stability_factor = np.exp(-1.0 * d1 ** 2)  # Close to 1 when velocity is low
        bonus = 10.0 * upright_closeness * stability_factor  # Smoothly scales based on both factors

        # COMPONENT 4.1: Smoother cost for being close to downright position
        downright_closeness = np.exp(-1.0 * abs(arctan2(s1,c1)) ** 2) # Close to 1 when is near down
        bonus += -10.0 * downright_closeness * stability_factor  # Smoothly scales based on both factors

        # COMPONENT 4.2: Bonus when pendulum is up and theta0 not moving
        stability_0 = np.exp(-1.0 * d0 ** 2)
        bonus += 5.0 * upright_closeness * stability_0

        # COMPONENT 5: Smoother penalty for approaching limits
        # Create a continuous penalty that increases as the arm approaches limits
        # Map the distance to limits to a 0-1 range, with 1 being at the limit
        limit_distance = np.clip(0.8 - 0.2 * (self.max_theta_0 - abs(arctan2(s0,c0))), 0, 1)

        # Apply a nonlinear function to create gradually increasing penalty
        # The penalty grows more rapidly as the arm gets very close to limits
        limit_penalty = -15.0 * limit_distance ** 3

        # COMPONENT 6: Energy management reward
        # This component is already quite smooth, just adjust scaling
        energy_reward = 2 - 0.15 * abs(m_1 * g * l_1 * (c1 + 1.0) + 0.5 * I_1 * d1 ** 2)

        # Combine all components
        reward = (
            20.0 +
                # 2 * (1.0 - c1)
             + upright_reward
            # + velocity_penalty
             + pos_penalty
             + bonus
             + limit_penalty
             + energy_reward
        )

        self.step_count += 1
        done = abs(np.arctan2(s0, c0)) > self.max_theta_0 or self.step_count >= self.max_steps
        #done = self.step_count >= self.max_steps
        truncated = self.step_count >= self.max_steps
        return self.state, reward, done, truncated, {}  # Return full state

# Train with frame stacking
env = DummyVecEnv([lambda: QubeServo2Env()])

# Train PPO
model = SAC(
    "MlpPolicy",
    env,
    learning_rate=3e-4,  # SAC typically uses a slightly higher default LR than PPO
    buffer_size=400000,  # Replay buffer size
    batch_size=1024,  # Number of samples per update
    tau=0.005,  # Soft update coefficient for target network
    gamma=0.99,  # Discount factor
    ent_coef="auto",  # Automatic entropy tuning
    verbose=1,
    tensorboard_log="./tensorboard_logs/",
    device=device
)
callback = TrainingMonitorCallback(check_freq=1000, patience=10, loss_threshold=0.01, verbose=1)
try:
    model.learn(total_timesteps=2000000, callback=callback)
except KeyboardInterrupt:
    print("Training interrupted! Model saved")
model.save("Trained Models/pendulum_sac_random_ultrareward")

# Test and collect data
env = QubeServo2Env()
obs, _ = env.reset()
rewards = []
thetas = []
alphas = []
voltages = []
times = []

for i in range(1000):
    stacked_obs = np.stack(obs, axis=0).flatten()
    action, _states = model.predict(stacked_obs, deterministic=True)
    obs, reward, done, truncated, _ = env.step(action)
    rewards.append(reward)
    thetas.append(np.degrees(np.arctan2(obs[0],obs[1])))
    alphas.append(np.degrees(np.arctan2(obs[3],obs[4])))
    voltages.append(action[0])
    times.append(i * env.dt)
    if done or truncated:
        obs, _ = env.reset()
        frame_history = [obs] * 2

# Plotting results
plt.figure(figsize=(12, 10))
plt.subplot(3, 1, 1)
plt.scatter(times, thetas, label=r'$\theta_0$ (arm angle)')
plt.scatter(times, alphas, label=r'$\theta_1$ (pendulum angle)')
plt.axhline(y=180, color='r', linestyle='--', label=r'$\theta_1 = +-180º$ (up)')
plt.axhline(y=-180, color='r', linestyle='--')
plt.axhline(y=150, color='k', linestyle='--', label=r'$\theta_0$ limits')
plt.axhline(y=-150, color='k', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Angle (degrees)')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.scatter(times, rewards, label='Reward')
plt.xlabel('Time (s)')
plt.ylabel('Reward')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.scatter(times, voltages, label='Voltage')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()