import gymnasium as gym
import numpy as np
import os
import torch
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from dqn import DQN, ReplayBuffer
from config import SeqQuestConfig
import ale_py

gym.register_envs(ale_py)

# Environment Setup
env_name = "ALE/Seaquest-v5"  # RAM observation, no sticky actions [web:5]
env = gym.make(env_name, render_mode=None, obs_type="ram")

obs_size = env.observation_space.shape[0]  # 128 for RAM [web:16]
n_actions = env.action_space.n  # 18 for Seaquest [web:7]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = "results"


def export_plot(ys, ylabel, title, filename):
    plt.figure()
    plt.plot(range(len(ys)), ys)
    plt.xlabel("Training Episode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename)
    plt.close()


# Training Loop Template
def train():
    obs, _ = env.reset()
    obs = obs.astype(np.float32) / 255.0  # Normalize RAM [0,255] -> [0,1] [web:16]
    config = SeqQuestConfig()
    model = DQN(env=env, config=config)

    # Initialize the target action value as the model.
    target_model = DQN(env=env, config=config)
    target_model.load_state_dict(model.state_dict())
    replay_buffer = ReplayBuffer(state_dim=obs_size, capacity=config.replay_buffer_size, device=device)

    episode_reward = 0
    episode_rewards = []

    for step in range(100000):

        action = model.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_obs = next_obs.astype(np.float32) / 255.0
        done = terminated or truncated

        replay_buffer.push(obs, action, reward, next_obs, done)
        episode_reward += reward

        obs = next_obs

        if replay_buffer.size >= replay_buffer.capacity:
            # Train the model:x
            (state, next_state, action, rewards, terminal) = replay_buffer.sample(batch_size=config.batch_size)
            with torch.no_grad():
                # Get the max Q values for the next state:
                next_state_Q = target_model.forward(next_state)
                max_Q = next_state_Q.max(dim=1).values
                max_Q = max_Q.reshape((max_Q.shape[0], 1))

            # Get the obtained Q for the action:
            obtained_Q = model.forward(state)
            q_action = torch.gather(obtained_Q, dim=1, index=action)
            td_target = rewards + (config.gamma * max_Q * (1 - terminal))

            # Compute the loss function:
            model.compute_loss(q_action.squeeze(), td_target.squeeze())

        if done:
            episode_rewards.append(episode_reward)
            print(f"Episode {len(episode_rewards)}, Reward: {episode_reward}")
            episode_reward = 0
            obs, _ = env.reset()
            obs = obs.astype(np.float32) / 255.0

        # Update the target model:
        target_model = soft_update(model, target_model, tau=config.tau_weight)

    # Save scores and plot
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "scores.npy"), episode_rewards)
    export_plot(episode_rewards, "Score", "Seaquest DQN", os.path.join(output_dir, "scores.png"))
    print(f"Saved scores.npy and scores.png to {output_dir}/")

    return model


def soft_update(model, target_model, tau=0.005):
    # Perform polyak averaging instead of updating the model every C steps:
    policy_state_dict = model.state_dict()
    target_state_dict = target_model.state_dict()

    # Iterate over all the parameters in the model
    for k, v in policy_state_dict.items():
        target_state_dict[k] = tau * v + (1 - tau) * target_state_dict[k]

    target_model.load_state_dict(target_state_dict)

    return target_model


# Evaluation Template
def evaluate(model: DQN):  # Pass your trained DQN model
    obs, _ = env.reset()
    obs = obs.astype(np.float32) / 255.0
    total_reward = 0

    # Trying to set the epsilon to a minimum value to avoid epsilon greedy action
    model.epsilon = 0.0001

    for _ in range(1000):
        with torch.no_grad():
            action = model.select_action(obs)

        obs, reward, terminated, truncated, _ = env.step(action)
        obs = obs.astype(np.float32) / 255.0
        total_reward += reward

        if terminated or truncated:
            obs, _ = env.reset()
            obs = obs.astype(np.float32) / 255.0

    print(f"Average reward: {total_reward / 1000:.2f}")
    env.close()


if __name__ == "__main__":
    trained_model = train()
    evaluate(trained_model)  # Uncomment to evaluate
    pass
