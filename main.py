import gymnasium as gym
import numpy as np
import torch

# Environment Setup
env_name = "Seaquest-ram-v4"  # RAM observation, no sticky actions [web:5]
env = gym.make(env_name, render_mode="human")

obs_size = env.observation_space.shape[0]  # 128 for RAM [web:16]
n_actions = env.action_space.n  # 18 for Seaquest [web:7]

print(f"Observation size: {obs_size}, Action space: {n_actions}")


# ===== HERE: Define your custom DQN class =====
# class DQN(torch.nn.Module):
#     def __init__(self, obs_size, n_actions):
#         ...
#     def forward(self, x):
#         ...
# ===== END =====

# ===== HERE: Initialize your DQN agent, target network, optimizer, replay buffer =====
# q_net = DQN(obs_size, n_actions)
# target_net = DQN(obs_size, n_actions)
# optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-4)
# replay_buffer = YourReplayBuffer(capacity=100000)
# ===== END =====

# Training Loop Template
def train():
    obs, _ = env.reset()
    obs = obs.astype(np.float32) / 255.0  # Normalize RAM [0,255] -> [0,1] [web:16]

    for step in range(100000):  # Adjust total steps as needed
        # ===== HERE: Epsilon-greedy action selection using your DQN =====
        # action = ...
        # ===== END =====

        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_obs = next_obs.astype(np.float32) / 255.0
        done = terminated or truncated

        # ===== HERE: Store transition in replay buffer =====
        # replay_buffer.push(obs, action, reward, next_obs, done)
        # ===== END =====

        # ===== HERE: Sample batch and train your DQN =====
        # if len(replay_buffer) > batch_size:
        #     ...
        # ===== END =====

        obs = next_obs

        if done:
            obs, _ = env.reset()
            obs = obs.astype(np.float32) / 255.0

        # ===== HERE: Update target network periodically =====
        # if step % target_update_freq == 0:
        #     ...
        # ===== END =====


# Evaluation Template
def evaluate(model):  # Pass your trained DQN model
    obs, _ = env.reset()
    obs = obs.astype(np.float32) / 255.0
    total_reward = 0

    for _ in range(1000):
        # ===== HERE: Deterministic action from your DQN =====
        # with torch.no_grad():
        #     action = ...
        # ===== END =====

        obs, reward, terminated, truncated, _ = env.step(action)
        obs = obs.astype(np.float32) / 255.0
        total_reward += reward

        if terminated or truncated:
            obs, _ = env.reset()
            obs = obs.astype(np.float32) / 255.0

    print(f"Average reward: {total_reward / 1000:.2f}")
    env.close()


if __name__ == "__main__":
    # train()  # Uncomment to train
    # evaluate(your_trained_model)  # Uncomment to evaluate
    pass
