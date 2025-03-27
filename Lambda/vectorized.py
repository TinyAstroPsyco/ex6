import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque, namedtuple
import gym
from gym.wrappers import AtariPreprocessing, FrameStack
import os
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
# import gym.envs.atari




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

# Training loop (vectorized envs)
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

    n_envs = env.num_envs
    n_actions = env.single_action_space.n
    policy_net = DQN(4, n_actions).to(device)
    target_net = DQN(4, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
    replay_buffer = ReplayBuffer(replay_size)
    scaler = GradScaler()

    obs, _ = env.reset()
    state = obs
    for _ in range(replay_prepopulate_steps // n_envs):
        actions = [env.single_action_space.sample() for _ in range(n_envs)]
        next_obs, rewards, terminateds, truncateds, _ = env.step(actions)
        dones = np.logical_or(terminateds, truncateds)
        for i in range(n_envs):
            replay_buffer.push(state[i], actions[i], rewards[i], next_obs[i], dones[i])
        state = next_obs

    rewards, lengths, losses = [], [], []
    models = {}
    step_count = 0
    i_episode = 0
    episode_rewards = np.zeros(n_envs)
    episode_lengths = np.zeros(n_envs)
    save_checkpoints_at = [int(num_steps * x) for x in [0.02, 0.30, 0.75, 1.0]]

    pbar = tqdm(total=num_steps)
    obs, _ = env.reset()
    state = obs

    while step_count < num_steps:
        epsilon = exploration.value(step_count)
        actions = []
        for i in range(n_envs):
            if random.random() > epsilon:
                state_tensor = torch.FloatTensor(state[i]).unsqueeze(0).to(device)
                action = policy_net(state_tensor).argmax(1).item()
            else:
                action = env.single_action_space.sample()
            actions.append(action)

        next_obs, rewards_step, terminateds, truncateds, _ = env.step(actions)
        dones = np.logical_or(terminateds, truncateds)

        for i in range(n_envs):
            replay_buffer.push(state[i], actions[i], rewards_step[i], next_obs[i], dones[i])
            episode_rewards[i] += rewards_step[i]
            episode_lengths[i] += 1

            if dones[i]:
                rewards.append(episode_rewards[i])
                lengths.append(episode_lengths[i])
                episode_rewards[i] = 0
                episode_lengths[i] = 0

        state = next_obs
        step_count += n_envs

        # Sample and train
        if len(replay_buffer) >= batch_size:
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
            f'Steps: {step_count} | AvgReturn: {np.mean(rewards[-10:]) if rewards else 0:.2f} | Eps: {epsilon:.2f}'
        )
        pbar.update(n_envs)

    pbar.close()
    return models, rewards, lengths, losses


# ---- CONFIG & RUN ---- #

num_envs = 4
# env = gym.vector.make("ALE/Pong-v5", num_envs=num_envs, asynchronous=True)
# env = gym.vector.make("PongNoFrameskip-v4", num_envs=num_envs, asynchronous=True)
import gym

# env = gym.vector.make("PongNoFrameskip-v4", num_envs=4, asynchronous=True)
# env = AtariPreprocessing(env, frame_skip=4, scale_obs=False)
# env = FrameStack(env, num_stack=4)


def make_env():
    def thunk():
        env = gym.make("ALE/Pong-v5", frameskip=1)
        env = AtariPreprocessing(env, frame_skip=1, scale_obs=False)
        env = FrameStack(env, num_stack=4)
        return env
    return thunk

env = gym.vector.AsyncVectorEnv([make_env() for _ in range(num_envs)])


# Hyperparams
num_steps = 100000
num_saves = 4
replay_size = 70_000
replay_prepopulate_steps = 50_000
batch_size = 32
gamma = 0.99
exploration = ExponentialSchedule(1.0, 0.05, 50000)

# Train
dqn_models, returns, lengths, losses = train_dqn_atari(
    env,
    num_steps=num_steps,
    num_saves=num_saves,
    replay_size=replay_size,
    replay_prepopulate_steps=replay_prepopulate_steps,
    batch_size=batch_size,
    exploration=exploration,
    gamma=gamma,
)

# Save
os.makedirs('checkpoint_ALE', exist_ok=True)
checkpoint = {
    "models": {key: model.custom_dump() for key, model in dqn_models.items()},
    "rewards": returns,
    "lengths": lengths,
    "losses": losses
}
torch.save(checkpoint, 'checkpoint_ALE/Pong-v5.pt')
