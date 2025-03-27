import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque, namedtuple
import cv2
import gymnasium as gym
import os
from tqdm import tqdm
import ale_py
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from gymnasium.wrappers import AtariPreprocessing

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

    def __len__(self):
        return len(self.buffer)

# FrameStack
class FrameStack:
    def __init__(self, env, k):
        self.env = env
        self.k = k
        self.frames = deque([], maxlen=k)

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
        return np.stack(list(self.frames), axis=0)

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.env, name)

# CNN DQN
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

# Epsilon decay schedule
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
            return self.a * np.exp(self.b * step)


def train_dqn_atari(
    env,
    num_steps,
    num_saves,
    replay_size,
    replay_prepopulate_steps,
    batch_size,
    exploration,
    gamma=0.99,
    target_update_freq=10_000,
    save_dir='checkpoints_pong'
):
    os.makedirs(save_dir, exist_ok=True)

    env = AtariPreprocessing(env, frame_skip=4, scale_obs=False)
    env = FrameStack(env, k=4)

    n_actions = env.action_space.n
    policy_net = DQN(4, n_actions).to(device)
    target_net = DQN(4, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
    replay_buffer = ReplayBuffer(replay_size)
    scaler = GradScaler()

    obs, _ = env.reset()
    state = obs
    for _ in range(replay_prepopulate_steps):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_state = next_obs
        done = terminated or truncated
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state if not done else env.reset()[0]

    rewards, lengths, losses = [], [], []
    models = {}
    step_count = 0
    i_episode = 0
    save_checkpoints_at = [int(num_steps * x) for x in [0.02, 0.30, 0.75, 1.0]]

    pbar = tqdm(total=num_steps)
    while step_count < num_steps:
        obs, _ = env.reset()
        state = obs
        episode_reward = 0
        episode_length = 0
        done = False

        while not done and step_count < num_steps:
            epsilon = exploration.value(step_count)
            if random.random() > epsilon:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = policy_net(state_tensor).argmax(1).item()
            else:
                action = env.action_space.sample()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = next_obs
            done = terminated or truncated

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            episode_length += 1

            transitions = replay_buffer.sample(batch_size)
            batch = Transition(*zip(*transitions))

            s = torch.FloatTensor(np.array(batch.state)).to(device)
            a = torch.LongTensor(batch.action).unsqueeze(1).to(device)
            r = torch.FloatTensor(batch.reward).unsqueeze(1).to(device)
            ns = torch.FloatTensor(np.array(batch.next_state)).to(device)
            d = torch.FloatTensor(batch.done).unsqueeze(1).to(device)

            with autocast():
                q_values = policy_net(s).gather(1, a)
                with torch.no_grad():
                    next_q = target_net(ns).max(1)[0].unsqueeze(1)
                    future_q = r + gamma * next_q
                    target = future_q * (1 - d) - d
                loss = F.mse_loss(q_values, target)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            losses.append(loss.item())

            if step_count % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if step_count in save_checkpoints_at:
                models[step_count] = policy_net

            pbar.set_description(
                f'Episode: {i_episode} | Steps: {episode_length:3} | Return: {episode_reward:5.2f} | Epsilon: {epsilon:.2f}'
            )
            pbar.update(1)
            step_count += 1

        rewards.append(episode_reward)
        lengths.append(episode_length)
        i_episode += 1

    pbar.close()
    return models, rewards, lengths, losses


# Atari Pong environment
base_env = gym.make("ALE/Pong-v5", frameskip=1)

# Optimized Parameters
num_steps = 2_500_000
num_saves = 4
replay_size = 70_000
replay_prepopulate_steps = 50_000
batch_size = 32
gamma = 0.99
exploration = ExponentialSchedule(1.0, 0.05, 1_000_000)

dqn_models, returns, lengths, losses = train_dqn_atari(
    base_env,
    num_steps=num_steps,
    num_saves=num_saves,
    replay_size=replay_size,
    replay_prepopulate_steps=replay_prepopulate_steps,
    batch_size=batch_size,
    exploration=exploration,
    gamma=gamma,
)

# Save model checkpoints
os.makedirs('checkpoint_ALE', exist_ok=True)
checkpoint = {key: model.custom_dump() for key, model in dqn_models.items()}
torch.save(checkpoint, f'checkpoint_ALE/Pong-v5.pt')

# Plotting

def moving_average(data, *, window_size = 50):
    assert data.ndim == 1
    kernel = np.ones(window_size)
    smooth_data = np.convolve(data, kernel) / np.convolve(np.ones_like(data), kernel)
    return smooth_data[: -window_size + 1]

plt.plot(returns)
plt.plot(moving_average(np.array(returns)))
plt.title("Episode Returns")
plt.xlabel("Episode")
plt.ylabel("Return")
plt.grid(True)
plt.show()

plt.plot(lengths)
plt.plot(moving_average(np.array(lengths)))
plt.title("Episode Lengths")
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.grid(True)
plt.show()

plt.plot(losses)
plt.plot(moving_average(np.array(losses)))
plt.title("Training Loss")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
