import argparse
import gymnasium as gym
import numpy as np
import os
import torch
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from dqn_her import DQN, ReplayBuffer
from config import SeqQuestConfig
import ale_py
from goal_wrapper import SeaQWrapper

gym.register_envs(ale_py)

# Environment Setup
env_name = "ALE/Seaquest-v5"  # RAM observation, no sticky actions [web:5]
env = gym.make(env_name, render_mode=None, obs_type="ram")
env = SeaQWrapper(env)

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


def save_checkpoint(model, target_model, step, episode_rewards):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'target_model_state_dict': target_model.state_dict(),
        'optimizer_state_dict': model.optimizer.state_dict(),
        'step': step,
        'epsilon': model.epsilon,
        'episode_rewards': episode_rewards,
    }
    torch.save(checkpoint, os.path.join(output_dir, "checkpoint.pt"))


def load_checkpoint(model, target_model):
    path = os.path.join(output_dir, "checkpoint.pt")
    if not os.path.exists(path):
        return None
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    target_model.load_state_dict(checkpoint['target_model_state_dict'])
    model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.epsilon = checkpoint['epsilon']
    return checkpoint


# Training Loop Template
def train(resume=False, seed=0):
    obs, _ = env.reset(seed=seed)
    obs = obs.astype(np.float32) / 255.0  # Normalize RAM [0,255] -> [0,1] [web:16]
    time_step = 0
    config = SeqQuestConfig()
    model = DQN(env=env, config=config)
    obtained_goals = []

    # Initialize the target action value as the model.
    target_model = DQN(env=env, config=config)
    target_model.load_state_dict(model.state_dict())
    replay_buffer = ReplayBuffer(state_dim=obs_size, capacity=config.replay_buffer_size, device=device, goal_dim=env.num_goal_dimension)

    start_step = 0
    episode_rewards = []

    if resume:
        checkpoint = load_checkpoint(model, target_model)
        if checkpoint:
            start_step = checkpoint['step'] + 1
            episode_rewards = checkpoint['episode_rewards']
            print(f"Resumed from step {start_step}, {len(episode_rewards)} episodes, epsilon={model.epsilon:.4f}")
        else:
            print("No checkpoint found, starting fresh.")

    episode_reward = 0

    for step in range(start_step, 5000000):
        action = model.select_action(obs, goal=env.normalize_goals(env.desired_goal))
        next_obs, reward, terminated, truncated, reward_her = env.step(action)
        next_obs = next_obs.astype(np.float32) / 255.0
        done = terminated or truncated
        obtained_goals.append(env.get_achieved_goal())

        replay_buffer.push(obs, action, reward_her, next_obs, done, env.desired_goal)
        time_step += 1 # Update the number of timesteps in the episodes

        episode_reward += reward

        obs = next_obs

        if replay_buffer.size >= 10000:
            # Train the model:x
            (state, next_state, action, rewards, terminal, goal) = replay_buffer.sample(batch_size=config.batch_size)
            with torch.no_grad():
                # Get the max Q values for the next state:
                # Applying Double DQN to avoid overestimation bias to be propagated.
                next_state_Q = model.forward(next_state, goal=env.normalize_goals(goal))
                target_next_state_Q = target_model.forward(next_state, goal=env.normalize_goals(goal))
                max_action = next_state_Q.max(dim=1).indices
                max_action = max_action.reshape((max_action.shape[0], 1))
                max_Q = torch.gather(target_next_state_Q, dim=1, index=max_action)

            # Get the obtained Q for the action:
            obtained_Q = model.forward(state, goal=env.normalize_goals(goal))
            q_action = torch.gather(obtained_Q, dim=1, index=action)
            td_target = rewards + (config.gamma * max_Q * (1 - terminal.float()))

            # Compute the loss function:
            model.compute_loss(q_action.squeeze(), td_target.squeeze())

        if done:
            episode_rewards.append(episode_reward)
            env.update_history()

            print(f"Episode {len(episode_rewards)}, Reward: {episode_reward}, Steps: {step},"
                  f"Achieved goal: {obtained_goals[-1]}, Desired goal: {env.desired_goal}, reward_her: {reward_her}")


            # If the terminal state still has reward_her set to -1,
            # then we need to update the goal based on the terminal state:
            # Update the replay buffer with N transitions:
            (state, next_state, action, _, terminal, goal) = replay_buffer.fetch_last_N_samples(time_step)
            # Update the goal state:
            replay_buffer.push_batch(state, next_state, action, terminal, obtained_goals)

            # Reset the environment:
            time_step = 0
            episode_reward = 0
            obtained_goals = []
            obs, _ = env.reset()
            if env.updated_max_goals:
                model.epsilon = max(model.epsilon, 0.5) # Restart the exploration phase to learn a new goal
            obs = obs.astype(np.float32) / 255.0

            # Exponential decay of epsilon value:
            # model.epsilon = max(1.0 - step / 1_000_000, 0.05)
            model.epsilon = max(0.05, model.epsilon * 0.9995)

        # Update the target model:
        # target_model = soft_update(model, target_model, tau=config.tau_weight)
        if step % config.target_update_frequency == 0:
            target_model.load_state_dict(model.state_dict())


    # Save scores, plot, and model checkpoint
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "scores.npy"), episode_rewards)
    export_plot(episode_rewards, "Score", "Seaquest DQN", os.path.join(output_dir, "scores.png"))
    save_checkpoint(model, target_model, step, episode_rewards)
    print(f"Saved scores.npy, scores.png, and checkpoint to {output_dir}/")

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

    # Set an extremely high goal during evaluation
    env.desired_goal = torch.tensor([5000, 480, 60])

    # Trying to set the epsilon to a minimum value to avoid epsilon greedy action
    model.epsilon = 0.0001

    for _ in range(1000):
        with torch.no_grad():
            action = model.select_action(obs, goal=env.normalize_goals(env.desired_goal))

        obs, reward, terminated, truncated, _ = env.step(action)
        obs = obs.astype(np.float32) / 255.0
        total_reward += reward

        if terminated or truncated:
            obs, _ = env.reset()
            obs = obs.astype(np.float32) / 255.0

    print(f"Average reward: {total_reward / 1000:.2f}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    trained_model = train(resume=args.resume, seed=args.seed)
    evaluate(trained_model)
