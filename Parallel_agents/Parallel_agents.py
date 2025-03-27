import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque, namedtuple
from torch.amp import GradScaler, autocast
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
from gymnasium.vector import SyncVectorEnv
from tqdm import tqdm
import ale_py
import matplotlib.pyplot as plt
import os

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

# FrameStack wrapper
class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(k, *shp),
            dtype=np.uint8,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.stack(self.frames, axis=0)

# DQN Model
class DQN(nn.Module):
    def __init__(self, input_channels, num_actions):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc_net = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = x / 255.0
        x = self.conv_net(x)
        x = x.view(x.size(0), -1)
        return self.fc_net(x)

    def custom_dump(self):
        return self.state_dict()

# Epsilon schedule
class ExponentialSchedule:
    def __init__(self, value_from, value_to, num_steps):
        self.value_from = value_from
        self.value_to = value_to
        self.num_steps = num_steps
        self.a = value_from
        self.b = np.log(value_to / value_from) / (num_steps - 1)

    def value(self, step):
        if step <= 0:
            return self.value_from
        elif step >= self.num_steps - 1:
            return self.value_to
        else:
            return max(self.a * np.exp(self.b * step), self.value_to)

# Environment factory
def make_env(seed=None):
    def thunk():
        env = gym.make("ALE/Pong-v5", frameskip=1)
        env = AtariPreprocessing(env, frame_skip=4, scale_obs=False)
        env = FrameStack(env, k=4)
        if seed is not None:
            env.reset(seed=seed)
        return env
    return thunk

# Training loop

def train_dqn_parallel(
    vec_env,
    num_steps,
    replay_size,
    replay_prepopulate_steps,
    batch_size,
    exploration,
    gamma=0.99,
    target_update_freq=5000,
    save_dir='checkpoints_pong_parallel'
):
    os.makedirs(save_dir, exist_ok=True)

    n_actions = vec_env.single_action_space.n
    policy_net = DQN(4, n_actions).to(device)
    target_net = DQN(4, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
    scaler = GradScaler()
    replay_buffer = ReplayBuffer(replay_size)

    obs, _ = vec_env.reset()
    state = obs
   

    # Prepopulate
    for _ in tqdm(range(replay_prepopulate_steps // vec_env.num_envs), desc="Prepopulating"):
        actions = [vec_env.single_action_space.sample() for _ in range(vec_env.num_envs)]
        next_obs, rewards, terminations, truncations, _ = vec_env.step(actions)
        dones = np.logical_or(terminations, truncations)
        for i in range(vec_env.num_envs):
            replay_buffer.push(state[i], actions[i], rewards[i], next_obs[i], dones[i])
        state = next_obs

    step_count = 0
    episode_rewards = np.zeros(vec_env.num_envs)
    episode_lengths = np.zeros(vec_env.num_envs)
    rewards, lengths, losses = [], [], []
    losses_per_episode = []
    current_episode_losses = [[] for _ in range(vec_env.num_envs)]
    save_checkpoints_at = [int(num_steps * x) for x in [0.02, 0.30, 0.75, 1.0]]

    pbar = tqdm(total=num_steps)
    while step_count < num_steps:
        epsilon = exploration.value(step_count)
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            q_values = policy_net(state_tensor)
            greedy_actions = q_values.argmax(dim=1).cpu().numpy()

        actions = [
            a if random.random() > epsilon else vec_env.single_action_space.sample()
            for a in greedy_actions
        ]
        next_obs, rewards_vec, terminations, truncations, _ = vec_env.step(actions)
        dones = np.logical_or(terminations, truncations)

        for i in range(vec_env.num_envs):
            replay_buffer.push(state[i], actions[i], rewards_vec[i], next_obs[i], dones[i])

        episode_rewards += rewards_vec
        episode_lengths += 1

        # Training
        transitions = replay_buffer.sample(batch_size)
        batch = Transition(*zip(*transitions))
        s = torch.tensor(np.array(batch.state), dtype=torch.float32, device=device)
        a = torch.tensor(batch.action, dtype=torch.long, device=device).unsqueeze(1)
        r = torch.tensor(batch.reward, dtype=torch.float32, device=device).unsqueeze(1)
        ns = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=device)
        d = torch.tensor(batch.done, dtype=torch.float32, device=device).unsqueeze(1)

        with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            q_values = policy_net(s).gather(1, a)
            with torch.no_grad():
                next_q = target_net(ns).max(1)[0].unsqueeze(1)
                target = r + gamma * next_q * (1 - d)
            loss = F.mse_loss(q_values, target)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_value = loss.item()
        losses.append(loss_value)

        for i in range(vec_env.num_envs):
            current_episode_losses[i].append(loss_value)

        for i in range(vec_env.num_envs):
            if dones[i]:
                rewards.append(episode_rewards[i])
                lengths.append(episode_lengths[i])
                if current_episode_losses[i]:
                    losses_per_episode.append(np.mean(current_episode_losses[i]))
                    current_episode_losses[i] = []
                episode_rewards[i] = 0
                episode_lengths[i] = 0

        if step_count % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if step_count in save_checkpoints_at:
            torch.save(policy_net.custom_dump(), os.path.join(save_dir, f"model_step_{step_count}.pt"))
            torch.save({
                'rewards': rewards.copy(),
                'lengths': lengths.copy(),
                'losses': losses.copy(),
                'losses_per_episode': losses_per_episode.copy()
            }, os.path.join(save_dir, f"metrics_step_{step_count}.pt"))
            print(f"\nCheckpoint saved at step {step_count}")

        avg_return = np.mean(rewards[-50:]) if len(rewards) >= 50 else np.mean(rewards)
        pbar.set_description(f"Step: {step_count:7} | AvgRet: {avg_return:6.2f} | Eps: {epsilon:.2f}")
        step_count += vec_env.num_envs
        state = next_obs
        pbar.update(vec_env.num_envs)

    pbar.close()

    torch.save(policy_net.custom_dump(), os.path.join(save_dir, f"final_model.pt"))
    torch.save({
        'rewards': rewards,
        'lengths': lengths,
        'losses_per_episode': losses_per_episode
    }, os.path.join(save_dir, f"final_metrics.pt"))

    return rewards, lengths, losses_per_episode

# Plotting
def moving_average(data, window_size=50):
    data = np.asarray(data)
    if data.ndim != 1:
        data = data.squeeze()
    kernel = np.ones(window_size)
    smooth = np.convolve(data, kernel, mode="valid") / window_size
    return smooth

def save_plots(rewards, lengths, losses_per_episode, plot_dir="training_plots_parallel"):
    os.makedirs(plot_dir, exist_ok=True)

    plt.plot(rewards)
    plt.plot(moving_average(rewards))
    plt.title("Episode Returns")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "episode_returns.png"))
    plt.show()

    plt.plot(lengths)
    plt.plot(moving_average(lengths))
    plt.title("Episode Lengths")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "episode_lengths.png"))
    plt.show()

    plt.plot(losses_per_episode, label="Avg Loss per Episode")
    plt.plot(moving_average(losses_per_episode), label="Smoothed", linestyle="--")
    plt.title("Loss per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "loss_per_episode.png"))
    plt.show()




num_envs = 8
vec_env = SyncVectorEnv([make_env(i) for i in range(num_envs)])
exploration = ExponentialSchedule(1.0, 0.01, 2_000_000)

rewards, lengths, losses, losses_per_episode = train_dqn_parallel(
    vec_env,
    num_steps=10_000_000,
    replay_size=200_000,
    replay_prepopulate_steps=50_000,
    batch_size=128,
    exploration=exploration
)

save_plots(rewards, lengths, losses_per_episode)

