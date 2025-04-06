import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
from tqdm import tqdm
import numba as nb
from time import time
from numpy.ma.core import arctan2

# Define device globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ====== System Constants ======
g = 9.81
base_max_voltage = 4.0
THETA_MIN = -2.2
THETA_MAX = 2.2

# ====== Base Parameter Values ======
base_Rm = 8.94
base_Km = 0.0431
base_DA = 0.0004
base_DL = 0.000003
base_mA = 0.053
base_mL = 0.024
base_LA = 0.086
base_LL = 0.128
base_JA = 0.0000572 + 0.00006
base_JL = 0.0000235
base_k = 0.002

# ====== Hyperparameters ======
batch_size = 256 * 8

# ====== Helper Functions ======
@nb.njit(fastmath=True, cache=True)
def clip_value(value, min_value, max_value):
    return min_value if value < min_value else (max_value if value > max_value else value)

@nb.njit(fastmath=True, cache=True)
def apply_voltage_deadzone(vm):
    return 0.0 if -0.2 <= vm <= 0.2 else vm

@nb.njit(fastmath=True, cache=True)
def normalize_angle(angle):
    angle = angle % (2 * np.pi)
    return angle - 2 * np.pi if angle > np.pi else angle

@nb.njit(fastmath=True, cache=True)
def enforce_theta_limits(state):
    theta_0, theta_1, theta_0_dot, theta_1_dot = state
    if theta_0 > THETA_MAX:
        theta_0 = THETA_MAX
        if theta_0_dot > 0:
            theta_0_dot = 0.0
    elif theta_0 < THETA_MIN:
        theta_0 = THETA_MIN
        if theta_0_dot < 0:
            theta_0_dot = 0.0
    return np.array([theta_0, theta_1, theta_0_dot, theta_1_dot])

# ====== Parameter Manager (unchanged) ======
class ParameterManager:
    def __init__(self, variation_pct=0.1, fixed_params=False, voltage_range=None):
        self.variation_pct = variation_pct
        self.fixed_params = fixed_params
        self.voltage_range = voltage_range
        self.base_params = {
            'Rm': base_Rm, 'Km': base_Km, 'DA': base_DA, 'DL': base_DL,
            'mA': base_mA, 'mL': base_mL, 'LA': base_LA, 'LL': base_LL,
            'JA': base_JA, 'JL': base_JL, 'k': base_k, 'max_voltage': base_max_voltage
        }
        self.current_params = {}
        self.param_history = []
        self.reset()

    def reset(self):
        if self.fixed_params:
            self.current_params = self.base_params.copy()
        else:
            self.current_params = {}
            for name, base_value in self.base_params.items():
                if name == 'max_voltage' and self.voltage_range is not None:
                    min_v, max_v = self.voltage_range
                    self.current_params[name] = np.random.uniform(min_v, max_v)
                else:
                    variation_factor = 1.0 + np.random.uniform(-self.variation_pct, self.variation_pct)
                    self.current_params[name] = base_value * variation_factor
        self.param_history.append(self.current_params.copy())
        self.current_params['l_1'] = self.current_params['LL'] / 2
        self.current_params['half_mL_l1_g'] = self.current_params['mL'] * self.current_params['l_1'] * g
        return self.current_params

    def get_current_params(self):
        return self.current_params.copy()

# ====== Time Step Generator (unchanged) ======
class VariableTimeGenerator:
    def __init__(self, mean=0.005, std_dev=0.002, min_dt=0.0025, max_dt=0.01):
        self.mean = mean
        self.std_dev = std_dev
        self.min_dt = min_dt
        self.max_dt = max_dt

    def get_next_dt(self):
        dt = np.random.normal(self.mean, self.std_dev)
        return clip_value(dt, self.min_dt, self.max_dt)

# ====== Dynamics Function (unchanged) ======
def dynamics_step(state, t, vm, params):
    theta_0, theta_1, theta_0_dot, theta_1_dot = state
    Rm, Km, DA, DL, mL, LA, l_1, JA, JL, k = [params[p] for p in ['Rm', 'Km', 'DA', 'DL', 'mL', 'LA', 'l_1', 'JA', 'JL', 'k']]
    s0, c0 = np.sin(theta_0), np.cos(theta_0)
    s1, c1 = np.sin(theta_1), np.cos(theta_1)
    if (theta_0 >= THETA_MAX and theta_0_dot > 0) or (theta_0 <= THETA_MIN and theta_0_dot < 0):
        theta_0_dot = 0.0
    vm = apply_voltage_deadzone(vm)
    torque = Km * (vm - Km * theta_0_dot) / Rm
    alpha = JA + mL * LA ** 2 + mL * l_1 ** 2 * s1 ** 2
    beta = -mL * l_1 ** 2 * (2 * s1 * c1)
    gamma = -mL * LA * l_1 * c1
    sigma = mL * LA * l_1 * s1
    M = np.array([[-alpha, -gamma], [-gamma, -(JL + mL * l_1 ** 2)]])
    f = np.array([
        -torque + DA * theta_0_dot + k * arctan2(s0, c0) + sigma * theta_1_dot ** 2 - beta * theta_0_dot * theta_1_dot,
        DL * theta_1_dot + mL * g * l_1 * s1 + 0.5 * beta * theta_0_dot ** 2
    ])
    det_M = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    if abs(det_M) < 1e-10:
        theta_0_ddot = 0
        theta_1_ddot = 0
    else:
        theta_0_ddot = (M[1, 1] * f[0] - M[0, 1] * f[1]) / det_M
        theta_1_ddot = (M[0, 0] * f[1] - M[1, 0] * f[0]) / det_M
    return np.array([theta_0_dot, theta_1_dot, theta_0_ddot, theta_1_ddot])

# ====== Environment with POMDP ======
class PendulumEnv:
    def __init__(self, dt=0.005, max_steps=2000, variable_dt=False, param_variation=0.1, fixed_params=False, voltage_range=None):
        self.fixed_dt = dt
        self.variable_dt = variable_dt
        self.voltage_range = voltage_range
        if variable_dt:
            self.time_generator = VariableTimeGenerator()
            self.dt = self.time_generator.get_next_dt()
        else:
            self.dt = self.fixed_dt
        self.param_manager = ParameterManager(variation_pct=param_variation, fixed_params=fixed_params, voltage_range=voltage_range)
        self.max_steps = max_steps
        self.step_count = 0
        self.state = None
        self.time_history = []
        self.params = None

    def reset(self, random_init=True):
        if self.variable_dt:
            self.dt = self.time_generator.get_next_dt()
        else:
            self.dt = self.fixed_dt
        self.params = self.param_manager.reset()
        if random_init:
            self.state = np.array([
                np.random.uniform(-0.1, 0.1),
                np.random.uniform(-0.1, 0.1),
                np.random.uniform(-0.05, 0.05),
                np.random.uniform(-0.05, 0.05)
            ])
        else:
            self.state = np.array([0.0, 0.1, 0.0, 0.0])
        self.step_count = 0
        self.time_history = []
        return self._get_observation()

    def _get_observation(self):
        theta_0, theta_1, _, _ = self.state
        theta_1_norm = normalize_angle(theta_1 + np.pi)
        return np.array([
            np.sin(theta_0), np.cos(theta_0),
            np.sin(theta_1_norm), np.cos(theta_1_norm)
        ], dtype=np.float32)

    def step(self, action):
        max_voltage = self.params['max_voltage']
        voltage = float(action) * max_voltage
        self.last_voltage = voltage
        self.time_history.append(self.dt)
        self.state = self._rk4_step(self.state, voltage)
        if self.variable_dt:
            self.dt = self.time_generator.get_next_dt()
        self.state = enforce_theta_limits(self.state)
        reward = self._compute_reward()
        self.step_count += 1
        done = self.step_count >= self.max_steps
        return self._get_observation(), reward, done, {}

    def _rk4_step(self, state, vm):
        state = enforce_theta_limits(state)
        k1 = dynamics_step(state, 0, vm, self.params)
        state_k2 = enforce_theta_limits(state + 0.5 * self.dt * k1)
        k2 = dynamics_step(state_k2, 0, vm, self.params)
        state_k3 = enforce_theta_limits(state + 0.5 * self.dt * k2)
        k3 = dynamics_step(state_k3, 0, vm, self.params)
        state_k4 = enforce_theta_limits(state + self.dt * k3)
        k4 = dynamics_step(state_k4, 0, vm, self.params)
        new_state = state + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return enforce_theta_limits(new_state)

    def _compute_reward(self):
        theta_0, theta_1, theta_0_dot, theta_1_dot = self.state
        last_voltage = getattr(self, 'last_voltage', 0.0)
        theta_1_norm = normalize_angle(theta_1 + np.pi)
        upright_reward = 1.0 * np.cos(theta_1_norm)
        arm_center = np.exp(-1.0 * theta_0 ** 2)
        upright_closeness = np.exp(-10.0 * theta_1_norm ** 2)
        stability_factor = np.exp(-0.6 * theta_1_dot ** 2)
        bonus = 3.0 * upright_closeness * stability_factor * arm_center
        downright_theta_1 = normalize_angle(theta_1)
        downright_closeness = np.exp(-10.0 * downright_theta_1 ** 2)
        bonus += -3.0 * downright_closeness * stability_factor
        theta_max_dist = np.clip(1.0 - abs(theta_0 - THETA_MAX) / 0.5, 0, 1)
        theta_min_dist = np.clip(1.0 - abs(theta_0 - THETA_MIN) / 0.5, 0, 1)
        limit_distance = max(theta_max_dist, theta_min_dist)
        limit_penalty = -10.0 * limit_distance ** 3
        JL = self.params['JL']
        mL = self.params['mL']
        l_1 = self.params['l_1']
        energy_reward = 2 - 0.15 * abs(mL * g * l_1 * (np.cos(theta_1_norm)) + 0.5 * JL * theta_1_dot ** 2 - mL * g * l_1)
        return upright_reward + bonus + limit_penalty + energy_reward

    def get_current_parameters(self):
        return self.params

# ====== RNN Feature Extractor ======
class RNNExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, features_dim=64):
        super(RNNExtractor, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, features_dim)
        self.hidden = None
        self.to(device)

    def reset_hidden(self):
        self.hidden = None

    def forward(self, obs):
        if len(obs.shape) == 2:  # (batch, obs_dim) -> (batch, 1, obs_dim)
            obs = obs.unsqueeze(1)
        batch_size, seq_len, _ = obs.shape
        if self.hidden is None or self.hidden[0].shape[1] != batch_size:
            self.hidden = (torch.zeros(1, batch_size, self.hidden_dim).to(device),
                           torch.zeros(1, batch_size, self.hidden_dim).to(device))
        out, self.hidden = self.rnn(obs, self.hidden)
        out = out[:, -1, :]
        return self.fc(out)

# ====== Actor Network with RNN ======
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256, features_dim=64):
        super(Actor, self).__init__()
        self.extractor = RNNExtractor(obs_dim, hidden_dim=64, features_dim=features_dim)
        self.net = nn.Sequential(
            nn.Linear(features_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.to(device)

    def forward(self, state):
        features = self.extractor(state)
        x = self.net(features)
        mean = torch.tanh(self.mean(x))
        log_std = torch.clamp(self.log_std(x), -20, 2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x = normal.rsample()
        action = torch.tanh(x)
        log_prob = normal.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

    def reset(self):
        self.extractor.reset_hidden()

# ====== Critic Network with RNN ======
class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256, features_dim=64):
        super(Critic, self).__init__()
        self.extractor = RNNExtractor(obs_dim, hidden_dim=64, features_dim=features_dim)
        self.q1_net = nn.Sequential(
            nn.Linear(features_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.q2_net = nn.Sequential(
            nn.Linear(features_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.to(device)

    def forward(self, state, action):
        features = self.extractor(state)
        x = torch.cat([features, action], dim=1)
        q1 = self.q1_net(x)
        q2 = self.q2_net(x)
        return q1, q2

    def reset(self):
        self.extractor.reset_hidden()

# ====== Replay Buffer (unchanged) ======
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = map(np.array, zip(*[self.buffer[i] for i in batch]))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# ====== SAC Agent with RNN ======
class SACAgent:
    def __init__(self, obs_dim, action_dim, hidden_dim=256, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2, automatic_entropy_tuning=True):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.actor = Actor(obs_dim, action_dim, hidden_dim)
        self.critic = Critic(obs_dim, action_dim, hidden_dim)
        self.critic_target = Critic(obs_dim, action_dim, hidden_dim)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        if automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor([action_dim])).item()
            # Use nn.Parameter to ensure log_alpha is a leaf tensor
            self.log_alpha = nn.Parameter(torch.zeros(1, device=device))
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        else:
            self.log_alpha = None  # Avoid undefined variable if not tuning
        self.device = device
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            if evaluate:
                action, _ = self.actor(state)
            else:
                action, _ = self.actor.sample(state)
            return action.cpu().numpy()[0]

    def update_parameters(self, memory, batch_size=batch_size):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.sample(batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state_batch)
            target_q1, target_q2 = self.critic_target(next_state_batch, next_action)
            target_q = torch.min(target_q1, target_q2)
            alpha = self.log_alpha.exp() if self.automatic_entropy_tuning else self.alpha
            target_q = target_q - alpha * next_log_prob
            target_q = reward_batch + (1 - done_batch) * self.gamma * target_q

        current_q1, current_q2 = self.critic(state_batch, action_batch)
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actions, log_probs = self.actor.sample(state_batch)
        q1, q2 = self.critic(state_batch, actions)
        min_q = torch.min(q1, q2)
        alpha = self.log_alpha.exp() if self.automatic_entropy_tuning else self.alpha
        actor_loss = (alpha * log_probs - min_q).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()  # Update self.alpha for consistency

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {'critic_loss': critic_loss.item(), 'actor_loss': actor_loss.item(), 'alpha': self.alpha}

    def reset(self):
        self.actor.reset()
        self.critic.reset()
        self.critic_target.reset()

# ====== Training Function (adapted for POMDP) ======
def train(variable_dt=False, param_variation=0.1, fixed_params=False, voltage_range=None, max_episodes=1000, eval_interval=10):
    env = PendulumEnv(variable_dt=variable_dt, param_variation=param_variation, fixed_params=fixed_params, voltage_range=voltage_range)
    obs_dim = 4  # POMDP observation: [sin(theta_0), cos(theta_0), sin(theta_1_norm), cos(theta_1_norm)]
    action_dim = 1
    agent = SACAgent(obs_dim, action_dim)
    replay_buffer = ReplayBuffer(100000)
    episode_rewards = []
    avg_rewards = []

    for episode in tqdm(range(max_episodes), desc="Training"):
        state = env.reset()
        episode_reward = 0
        agent.reset()  # Reset RNN hidden states

        for step in range(env.max_steps):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if len(replay_buffer) > batch_size:
                update_info = agent.update_parameters(replay_buffer)

            if done:
                break

        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        avg_rewards.append(avg_reward)

        if (episode + 1) % eval_interval == 0 or episode == 0:
            print(f"Episode {episode + 1}/{max_episodes} | Reward: {episode_reward:.2f} | Avg Reward: {avg_reward:.2f}")

        if avg_reward > 10000 and episode > 500:
            print(f"Environment solved in {episode + 1} episodes!")
            break

    torch.save(agent.actor.state_dict(), "actor_pomdp.pth")
    torch.save(agent.critic.state_dict(), "critic_pomdp.pth")

    plt.plot(episode_rewards, label='Episode Reward')
    plt.plot(avg_rewards, label='Moving Average')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.show()

    return agent

# ====== Main Execution ======
if __name__ == "__main__":
    agent = train(variable_dt=True, param_variation=0.15, max_episodes=1000, eval_interval=10)