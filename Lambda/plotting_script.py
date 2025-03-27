import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def moving_average(data, window_size=50):
    data = np.array(data)
    kernel = np.ones(window_size)
    smoothed = np.convolve(data, kernel, mode='valid') / np.convolve(np.ones_like(data), kernel, mode='valid')
    return smoothed

# ---- Load Checkpoint ----
checkpoint_path = 'checkpoint_ALE/Pong-v5.pt'
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

rewards = checkpoint['rewards']
lengths = checkpoint['lengths']
losses = checkpoint['losses']

# ---- Create Save Directory ----
plot_dir = 'plots'
os.makedirs(plot_dir, exist_ok=True)

# ---- Plot & Save ----

# Rewards
plt.figure()
plt.plot(rewards, label='Return')
plt.plot(moving_average(rewards), label='Moving Avg')
plt.title('Episode Returns')
plt.xlabel('Episode')
plt.ylabel('Return')
plt.legend()
plt.grid(True)
plt.savefig(f'{plot_dir}/episode_returns.png')
plt.close()

# Lengths
plt.figure()
plt.plot(lengths, label='Episode Length')
plt.plot(moving_average(lengths), label='Moving Avg')
plt.title('Episode Lengths')
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.legend()
plt.grid(True)
plt.savefig(f'{plot_dir}/episode_lengths.png')
plt.close()

# Losses
plt.figure()
plt.plot(losses, label='Loss')
plt.plot(moving_average(losses), label='Moving Avg')
plt.title('Training Loss')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(f'{plot_dir}/training_loss.png')
plt.close()

print("✅ Plots saved in 'plots/' folder.")
