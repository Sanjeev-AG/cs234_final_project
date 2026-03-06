from stable_baselines3 import DQN, HerReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import gymnasium as gym
from goal_wrapper_sb3 import SeaQWrapperSB3
from config import SeqQuestConfig
import ale_py
import torch
import argparse


class EpisodeRewardCallback(BaseCallback):
    """Prints the reward at the end of each training episode."""
    def __init__(self):
        super().__init__()
        self.episode_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_count += 1
                print(f"Episode {self.episode_count}: "
                      f"reward = {info['episode']['r']:.2f}, "
                      f"length = {info['episode']['l']}")
        return True

# Environment Setup
env_name = "ALE/Seaquest-v5"  # RAM observation, no sticky actions [web:5]
env = gym.make(env_name, render_mode=None, obs_type="ram")
env = SeaQWrapperSB3(env, SeqQuestConfig())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = "results"


def train():
    model = DQN(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"),
        buffer_size=int(1e6),
        learning_starts=1000,
        verbose=0,
    )

    model.learn(total_timesteps=1000000, callback=EpisodeRewardCallback())

    return model


def evaluate(model: DQN, num_episodes=10):
    obs, _ = env.reset()
    episode_reward = 0
    episode_rewards = []

    # Set an extremely high goal during evaluation
    env.desired_goal = torch.tensor([5000, 480, 60])

    while len(episode_rewards) < num_episodes:
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward

        if terminated or truncated:
            episode_rewards.append(episode_reward)
            print(f"Episode {len(episode_rewards)}: reward = {episode_reward:.2f}")
            episode_reward = 0
            obs, _ = env.reset()

    print(f"\nMean reward over {num_episodes} episodes: {np.mean(episode_rewards):.2f}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    trained_model = train()
    evaluate(trained_model)


